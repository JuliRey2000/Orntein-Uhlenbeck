"""Estudios: configuración congelada y registro persistente de la prueba (sección 8.1, decisión S10).

Un estudio es la ÚNICA vía de la API para ejecutar el segmento de prueba. Se guarda en
``<studies_dir>/<id>.json`` y sobrevive a reinicios (a diferencia de ``RunConfig.test_history``,
que viaja dentro del archivo; ver S10a/S10b).

Reglas (CONTRATO):
- ``create(config, data, name)``: ejecuta ``run_pair(config, data, include_test=False)`` para
  fijar instantáneas (``snapshot_id``) y el split congelado; guarda el RunConfig devuelto (con
  ``include_test=False`` y ``test_history`` vacío). Rechaza (ConfigError en español) un config con
  ``include_test=True``: la prueba se ejecuta solo con ``run_test``.
- ``update_config(id, strategy=None, costs=None, analysis=None)``: solo cambian parámetros; los
  datos (pair, snapshots, frozen_split) NO cambian (para otro par se crea otro estudio). Cada
  cambio queda en ``changes`` (qué campos, hash antes/después, fecha). Si la prueba ya se vio
  (``test_runs`` no vacío) y el hash cambia → ``contaminated = True`` PERMANENTE con
  ``contamination_reason`` en español ("se modificaron umbrales después de ver la prueba").
  Cambiar ``analysis`` (p. ej. α, ventana) también cuenta.
- ``run_test(id, data)``: ejecuta ``run_pair`` con la configuración actual e ``include_test=True``
  y ``test_history`` = hashes de ``test_runs``; registra la ejecución (hash, fecha); si el hash
  difiere de ejecuciones anteriores → contaminado (el runner ya lo detecta con test_history;
  combina ambas señales). Devuelve (Study, RunResult) con ``result.test_contaminated`` igual al
  estado del estudio (True si contaminado por cualquiera de las dos vías).
- ``result(id, data, include_test)``: re-ejecuta (determinista, desde caché) para exportar o
  mostrar; ``include_test`` solo permitido si ya se ejecutó la prueba (si no → ConfigError
  "ejecuta primero la prueba"). El RunResult refleja ``contaminated``.
- ``get``, ``list`` (más recientes primero), ``delete``. Id = uuid4 hex; ids con otro formato →
  DataError(code="ESTUDIO_NO_ENCONTRADO") (evita rutas peligrosas). Escritura atómica
  (temporal + os.replace) y lock por proceso. Marcas de tiempo UTC ISO.

Precisiones de la implementación (decisiones de la fase 2 sobre estudios; ver DECISIONS.md):
- Identidad: los hashes de ``changes`` (``hash_before``/``hash_after``) son los de la
  configuración guardada (``include_test=False``). Los de ``test_runs`` son los de la
  configuración CON la prueba (``include_test=True``), los mismos que registra
  ``RunConfig.test_history`` y que compara el runner. ``StudyChange.test_hash_after`` guarda la
  versión «con prueba» del hash posterior a un cambio.
- Historial que se pasa al runner (``history_hashes``): hashes de ``test_runs`` más los
  ``test_hash_after`` de los cambios hechos DESPUÉS de ver la prueba. Solo un estudio
  contaminado tiene estos últimos, así que no crea falsos positivos, y la marca se conserva al
  reproducir el ``run_config.json`` exportado fuera del estudio (incluso si se volvió a la
  configuración original).
- ``create`` importa el ``test_history`` que traiga la configuración (la prueba ya se vio, p. ej.
  con la demo de la fase 1): esas ejecuciones quedan en ``test_runs`` con ``source =
  "run_config"`` y, si alguna es de otra configuración, el estudio nace contaminado. Así un
  estudio nuevo nunca «lava» un historial existente.
- ``create`` también importa las ejecuciones de la prueba de OTROS estudios (incluidos los
  eliminados) del mismo par (mismas series A y B, en cualquier orden, sin distinguir mayúsculas)
  cuyo periodo de prueba se solapa con el del estudio nuevo, con ``source = "otro_estudio"``. Si
  alguna es de otra configuración, el estudio nace contaminado: crear un estudio nuevo (o borrar
  el anterior y volver a crearlo) con umbrales ajustados después de ver la prueba no «lava» la
  marca. Con la misma configuración (mismo hash) no hay contaminación.
- ``delete`` de un estudio con ejecuciones propias de la prueba lo quita de la lista pero conserva
  su registro en ``<studies_dir>/eliminados/`` (solo para la regla anterior); los demás se borran.
- ``update_config`` rechaza cambiar ``analysis.split`` (``DIVISION_CONGELADA``): la división
  está congelada en ``frozen_split`` y cambiarla no tendría efecto. Un cambio sin diferencias no
  se registra.
- ``run_test`` exige que haya algo que probar: periodo de prueba no vacío (``SIN_PERIODO_PRUEBA``)
  y estrategia permitida (``AnalysisBlocked PRUEBA_NO_SIMULABLE``, con los motivos). En esos
  casos no registra ninguna ejecución.
- ``result(include_test=True)`` con una configuración cuyo hash aún no figura en ``test_runs``
  (se cambió después de ver la prueba) es una ejecución nueva de la prueba: se registra con
  ``source = "exportacion"`` (el estudio ya está contaminado por el cambio).
- El RunResult de ``run_test`` y ``result`` lleva ``test_contamination_note`` con el origen y
  el motivo de la marca (la exportación lo usa en el LEEME).
- Lock: uno por proceso y carpeta para leer y escribir archivos, y otro por estudio que
  serializa las operaciones de lectura-modificación-escritura (incluido el cálculo de
  ``run_test``) sin bloquear los demás estudios.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from mrlab.analysis.pair_pipeline import STRATEGY_BLOCKING_CODES, AnalysisConfig, PairStatus
from mrlab.analysis.runner import RunResult, run_pair
from mrlab.data.service import DataService
from mrlab.domain.errors import AnalysisBlocked, ConfigError, DataError, LabError
from mrlab.domain.models import Segment
from mrlab.reporting.run_config import RunConfig, config_hash
from mrlab.strategy.params import CostParams, StrategyParams

logger = logging.getLogger("mrlab.studies")

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SUFFIX = ".json"
#: Subcarpeta con el registro de los estudios eliminados cuya prueba ya se vio.
DELETED_DIR = "eliminados"

TestRunSource = Literal["prueba", "exportacion", "run_config", "otro_estudio"]

#: Regla de la prueba (para la interfaz y la API).
STUDY_RULE_NOTE = (
    "La prueba fuera de muestra se ejecuta solo dentro de un estudio, con la configuración "
    "congelada. Ver la prueba y después cambiar cualquier parámetro (umbrales, costes o análisis) "
    "marca el estudio como «prueba contaminada» de forma permanente, también en las "
    "exportaciones: sus resultados dejan de ser una evaluación fuera de muestra."
)


class StudyChange(BaseModel):
    """Cambio de configuración registrado en un estudio."""

    at: datetime = Field(description="Fecha y hora del cambio (UTC)")
    fields: list[str] = Field(description="Campos modificados (p. ej. strategy.entry_z)")
    hash_before: str = Field(description="Hash de la configuración guardada antes del cambio")
    hash_after: str = Field(description="Hash de la configuración guardada después del cambio")
    after_test: bool = Field(description="True si el cambio se hizo después de ver la prueba")
    test_hash_after: str | None = Field(
        None,
        description=(
            "Hash de la nueva configuración CON la prueba incluida (el que registra "
            "test_history); solo en cambios hechos después de ver la prueba"
        ),
    )


class StudyTestRun(BaseModel):
    """Ejecución registrada de la prueba fuera de muestra."""

    at: datetime = Field(description="Fecha y hora de la ejecución (UTC)")
    config_hash: str = Field(description="Hash de la configuración con la que se ejecutó")
    source: TestRunSource = Field(
        "prueba",
        description=(
            "prueba: ejecución pedida; exportacion: ejecución al exportar con una configuración "
            "nueva; run_config: ejecución anterior importada del test_history del run_config.json; "
            "otro_estudio: ejecución de otro estudio del mismo par con un periodo de prueba que "
            "se solapa"
        ),
    )
    study_id: str | None = Field(
        None, description="Estudio donde se ejecutó (solo para source = otro_estudio)"
    )


class Study(BaseModel):
    id: str = Field(description="Identificador del estudio (32 caracteres hexadecimales)")
    name: str = Field(description="Nombre del estudio")
    created_at: datetime = Field(description="Creación (UTC)")
    updated_at: datetime = Field(description="Última modificación (UTC)")
    config: RunConfig = Field(
        description=(
            "Configuración congelada: instantáneas de datos y fechas de corte fijas; los "
            "parámetros solo cambian con PUT /api/studies/{id}/config (y quedan registrados)"
        )
    )
    test_runs: list[StudyTestRun] = Field(
        default_factory=list, description="Ejecuciones registradas de la prueba"
    )
    changes: list[StudyChange] = Field(
        default_factory=list, description="Cambios de configuración registrados"
    )
    contaminated: bool = Field(
        False, description="True si la prueba está contaminada (marca permanente)"
    )
    contamination_reason: str | None = Field(
        None, description="Motivo de la contaminación, en español"
    )

    @property
    def test_viewed(self) -> bool:
        """True si la prueba de este par y periodo ya se vio (aquí, en otro estudio o según el
        historial del run_config.json)."""
        return bool(self.test_runs)

    @property
    def test_run_here(self) -> bool:
        """True si la prueba se ejecutó en ESTE estudio (pedida o al exportar)."""
        return any(r.source in ("prueba", "exportacion") for r in self.test_runs)

    @property
    def symbols(self) -> tuple[str, str]:
        return self.config.pair.a.symbol, self.config.pair.b.symbol

    def history_hashes(self) -> list[str]:
        """Historial para el runner: ejecuciones de la prueba y configuraciones posteriores."""
        hashes = [r.config_hash for r in self.test_runs]
        hashes += [c.test_hash_after for c in self.changes if c.after_test and c.test_hash_after]
        return list(dict.fromkeys(hashes))


# --------------------------------------------------------------------------- utilidades


def _now() -> datetime:
    return datetime.now(UTC)


def _date_text(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def hash_with_test(config: RunConfig) -> str:
    """Hash de la configuración con la prueba incluida (el que registra ``test_history``)."""
    return config_hash(config.model_copy(update={"include_test": True, "config_hash": None}))


def _stored_hash(config: RunConfig) -> str:
    return config_hash(config.model_copy(update={"include_test": False, "config_hash": None}))


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}"))
        return out
    return {prefix: value}


def _changed_fields(before: RunConfig, after: RunConfig) -> list[str]:
    fields: list[str] = []
    for section in ("analysis", "strategy", "costs"):
        old = _flatten(getattr(before, section).model_dump(mode="json"), section)
        new = _flatten(getattr(after, section).model_dump(mode="json"), section)
        fields += [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
    return fields


def _not_found(study_id: Any) -> DataError:
    shown = str(study_id)
    if len(shown) > 64:
        shown = shown[:63] + "…"
    return DataError(
        f"No existe el estudio «{shown}». Los identificadores de estudio tienen 32 caracteres "
        "hexadecimales; consulta la lista de estudios.",
        code="ESTUDIO_NO_ENCONTRADO",
        details={"study_id": shown},
    )


def _pair_key(config: RunConfig) -> frozenset[tuple[str, str]]:
    """Identidad del par analizado (series A y B, en cualquier orden)."""
    return frozenset(
        (ref.provider.strip().lower(), ref.symbol.strip().upper())
        for ref in (config.pair.a, config.pair.b)
    )


def _test_overlaps(x: RunConfig, y: RunConfig) -> bool:
    sx, sy = x.frozen_split, y.frozen_split
    if sx is None or sy is None:
        return False
    if None in (sx.test_start, sx.test_end, sy.test_start, sy.test_end):
        return False
    return sx.test_start <= sy.test_end and sy.test_start <= sx.test_end  # type: ignore[operator]


def _config_fixed_at(study: Study) -> datetime:
    """Momento en que se fijó la configuración actual (creación o último cambio)."""
    return study.changes[-1].at if study.changes else study.created_at


_locks: dict[tuple[str, str], threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock_for(directory: Path, key: str) -> threading.RLock:
    """Lock por proceso: uno por carpeta (archivos) y uno por estudio (operaciones)."""
    ident = (str(directory.resolve()), key)
    with _locks_guard:
        return _locks.setdefault(ident, threading.RLock())


# --------------------------------------------------------------------------- almacén


class StudyStore:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    # ------------------------------------------------------------------ archivos

    @property
    def _files_lock(self) -> threading.RLock:
        return _lock_for(self.directory, "")

    @contextlib.contextmanager
    def _study_lock(self, study_id: str) -> Iterator[None]:
        self._path(study_id)  # valida el id antes de crear un lock con él
        with _lock_for(self.directory, study_id):
            yield

    def _path(self, study_id: Any) -> Path:
        if not isinstance(study_id, str) or not _ID_RE.fullmatch(study_id):
            raise _not_found(study_id)
        return self.directory / f"{study_id}{_SUFFIX}"

    def _read(self, path: Path, study_id: str) -> Study:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise _not_found(study_id) from None
        except (OSError, UnicodeDecodeError) as exc:
            raise self._corrupt(study_id, str(exc)) from exc
        try:
            study = Study.model_validate_json(raw)
        except (ValidationError, ValueError, LabError) as exc:
            raise self._corrupt(study_id, type(exc).__name__) from exc
        if study.id != study_id:
            raise self._corrupt(study_id, "el identificador interno no coincide con el archivo")
        return study

    @staticmethod
    def _corrupt(study_id: str, reason: str) -> DataError:
        return DataError(
            f"El archivo del estudio «{study_id}» está dañado o no tiene el formato esperado, así "
            "que no se puede usar. Si lo editaste a mano, restaura una copia; si no, crea un "
            "estudio nuevo (la prueba ya vista con esta configuración debe considerarse "
            "contaminada).",
            code="ESTUDIO_CORRUPTO",
            details={"study_id": study_id, "motivo": reason},
        )

    def _write(self, study: Study) -> None:
        path = self._path(study.id)
        payload = study.model_dump_json(indent=2) + "\n"
        with self._files_lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=f".{study.id}.", suffix=".tmp", dir=self.directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise

    # ------------------------------------------------------------------ API pública

    def get(self, study_id: str) -> Study:
        path = self._path(study_id)
        with self._files_lock:
            return self._read(path, study_id)

    def list(self) -> list[Study]:
        studies: list[Study] = []
        with self._files_lock:
            if not self.directory.is_dir():
                return []
            for path in self.directory.glob(f"*{_SUFFIX}"):
                study_id = path.name[: -len(_SUFFIX)]
                if not _ID_RE.fullmatch(study_id):
                    continue
                try:
                    studies.append(self._read(path, study_id))
                except DataError as exc:
                    logger.warning("Se omite el estudio %s: %s", study_id, exc.message)
        return sorted(studies, key=lambda s: (s.created_at, s.id), reverse=True)

    def delete(self, study_id: str) -> None:
        """Quita el estudio. Si su prueba ya se vio, su registro pasa a ``eliminados/``."""
        path = self._path(study_id)
        with self._study_lock(study_id), self._files_lock:
            try:
                study = self._read(path, study_id)
            except DataError as exc:
                if exc.code != "ESTUDIO_CORRUPTO":
                    raise
                study = None
            # Solo hace falta conservar las ejecuciones PROPIAS (las copiadas de otros estudios
            # siguen en su estudio de origen).
            own = study is not None and any(r.source != "otro_estudio" for r in study.test_runs)
            if own:
                trash = self.directory / DELETED_DIR
                trash.mkdir(parents=True, exist_ok=True)
                os.replace(path, trash / path.name)
                return
            try:
                path.unlink()
            except FileNotFoundError:
                raise _not_found(study_id) from None

    # ------------------------------------------------------------------ otros estudios

    def _study_files(self) -> Iterator[tuple[Path, str]]:
        for folder in (self.directory, self.directory / DELETED_DIR):
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob(f"*{_SUFFIX}")):
                study_id = path.name[: -len(_SUFFIX)]
                if _ID_RE.fullmatch(study_id):
                    yield path, study_id

    def _foreign_runs(self, study: Study, until: datetime) -> list[StudyTestRun]:
        """Ejecuciones de la prueba de OTROS estudios (también eliminados) del mismo par con un
        periodo de prueba que se solapa, ocurridas hasta ``until``."""
        key = _pair_key(study.config)
        runs: list[StudyTestRun] = []
        with self._files_lock:
            for path, other_id in self._study_files():
                if other_id == study.id:
                    continue
                try:
                    other = self._read(path, other_id)
                except DataError:
                    continue
                if not other.test_runs or _pair_key(other.config) != key:
                    continue
                if not _test_overlaps(other.config, study.config):
                    continue
                runs += [
                    StudyTestRun(
                        at=run.at,
                        config_hash=run.config_hash,
                        source="otro_estudio",
                        study_id=other.id,
                    )
                    for run in other.test_runs
                    # Las copias de otros estudios ya se leen en su estudio de origen.
                    if run.source != "otro_estudio" and run.at <= until
                ]
        return runs

    def _sync_foreign_runs(self, study: Study, until: datetime) -> bool:
        """Añade a ``study.test_runs`` las ejecuciones ajenas que aún no tenía. True si añadió."""
        known = {
            (r.study_id, r.config_hash, r.at) for r in study.test_runs if r.source == "otro_estudio"
        }
        new = [
            r
            for r in self._foreign_runs(study, until)
            if (r.study_id, r.config_hash, r.at) not in known
        ]
        study.test_runs.extend(sorted(new, key=lambda r: r.at))
        return bool(new)

    def _mark_other_config(self, study: Study, current: str, now: datetime) -> None:
        """Contamina si alguna ejecución registrada usó otra configuración (con su motivo)."""
        if study.contaminated:
            return
        differing = [r for r in study.test_runs if r.config_hash != current]
        if not differing:
            return
        study.contaminated = True
        foreign = next((r for r in differing if r.source == "otro_estudio"), None)
        imported = any(r.source == "run_config" for r in differing)
        if imported:
            study.contamination_reason = (
                "La configuración con la que se creó el estudio ya registraba ejecuciones de la "
                "prueba con otra configuración (historial del run_config.json): la prueba ya se "
                "había visto antes de congelar estos parámetros."
            )
        elif foreign is not None:
            other_name = self._name_of(foreign.study_id)
            study.contamination_reason = (
                f"La prueba de este mismo par, en un periodo que se solapa, ya se había visto en "
                f"el estudio {other_name} el {_date_text(foreign.at)} con otra configuración: "
                "los parámetros de este estudio se fijaron después de verla, así que sus "
                "resultados de prueba no son una evaluación fuera de muestra."
            )
        else:
            study.contamination_reason = (
                f"La prueba se volvió a ejecutar el {_date_text(now)} con una configuración "
                f"distinta (hash {current[:12]}…) de la que ya se había evaluado: los "
                "resultados de la prueba ya no son una evaluación fuera de muestra."
            )

    def _name_of(self, study_id: str | None) -> str:
        if study_id is None:
            return "(desconocido)"
        for folder in (self.directory, self.directory / DELETED_DIR):
            path = folder / f"{study_id}{_SUFFIX}"
            if path.is_file():
                try:
                    other = self._read(path, study_id)
                except DataError:
                    break
                deleted = " (eliminado)" if folder.name == DELETED_DIR else ""
                return f"«{other.name}» ({study_id}){deleted}"
        return f"({study_id})"

    def create(self, config: RunConfig, data: DataService, name: str = "") -> Study:
        if config.include_test:
            raise ConfigError(
                "Un estudio se crea sin la prueba (include_test debe ser false): primero se "
                "congela la configuración y después la prueba se ejecuta una sola vez, desde el "
                "propio estudio.",
                code="PRUEBA_SOLO_EN_ESTUDIO",
                details={"include_test": True},
            )
        imported = list(dict.fromkeys(config.test_history))
        result = run_pair(config, data, include_test=False)
        frozen = result.config.model_copy(update={"include_test": False, "test_history": []})
        frozen = frozen.model_copy(update={"config_hash": config_hash(frozen)})
        now = _now()
        a, b = frozen.pair.a.symbol, frozen.pair.b.symbol
        study = Study(
            id=uuid.uuid4().hex,
            name=name.strip() or f"{a} / {b}",
            created_at=now,
            updated_at=now,
            config=frozen,
            test_runs=[StudyTestRun(at=now, config_hash=h, source="run_config") for h in imported],
        )
        with self._files_lock:
            self._sync_foreign_runs(study, now)
            self._mark_other_config(study, hash_with_test(frozen), now)
            self._write(study)
        return study

    def update_config(
        self,
        study_id: str,
        *,
        strategy: StrategyParams | None = None,
        costs: CostParams | None = None,
        analysis: AnalysisConfig | None = None,
    ) -> Study:
        with self._study_lock(study_id):
            study = self.get(study_id)
            old = study.config
            if analysis is not None and (
                analysis.split.model_dump() != old.analysis.split.model_dump()
            ):
                raise ConfigError(
                    "La división formación/validación/prueba de un estudio está congelada (sus "
                    "fechas de corte se fijaron al crearlo) y no se puede cambiar. Para otra "
                    "división, crea un estudio nuevo.",
                    code="DIVISION_CONGELADA",
                    details={"actual": old.analysis.split.model_dump(mode="json")},
                )
            updates: dict[str, Any] = {}
            if strategy is not None:
                updates["strategy"] = strategy
            if costs is not None:
                updates["costs"] = costs
            if analysis is not None:
                updates["analysis"] = analysis
            new = old.model_copy(update={**updates, "config_hash": None})
            new = new.model_copy(update={"config_hash": config_hash(new)})
            fields = _changed_fields(old, new)
            hash_before, hash_after = _stored_hash(old), _stored_hash(new)
            if not fields and hash_before == hash_after:
                return study
            now = _now()
            # Una prueba del mismo par vista en otro estudio antes de este cambio también cuenta.
            self._sync_foreign_runs(study, now)
            after_test = study.test_viewed
            study.changes.append(
                StudyChange(
                    at=now,
                    fields=fields,
                    hash_before=hash_before,
                    hash_after=hash_after,
                    after_test=after_test,
                    test_hash_after=hash_with_test(new) if after_test else None,
                )
            )
            study.config = new
            study.updated_at = now
            if after_test and hash_before != hash_after and not study.contaminated:
                shown = ", ".join(fields[:6]) + (
                    f" y {len(fields) - 6} más" if len(fields) > 6 else ""
                )
                study.contaminated = True
                study.contamination_reason = (
                    f"Se modificaron parámetros ({shown}) el {_date_text(now)}, después de ver "
                    "la prueba: los resultados de la prueba ya no son una evaluación fuera de "
                    "muestra."
                )
            self._write(study)
            return study

    def run_test(self, study_id: str, data: DataService) -> tuple[Study, RunResult]:
        with self._study_lock(study_id):
            study = self.get(study_id)
            return self._execute_test(study, data, source="prueba")

    def result(self, study_id: str, data: DataService, include_test: bool = False) -> RunResult:
        with self._study_lock(study_id):
            study = self.get(study_id)
            if include_test:
                if not study.test_viewed:
                    raise ConfigError(
                        "La prueba de este estudio aún no se ha ejecutado: ejecuta primero la "
                        "prueba (una sola vez, con la configuración congelada) para poder "
                        "mostrarla o exportarla.",
                        code="PRUEBA_NO_EJECUTADA",
                        details={"study_id": study_id},
                    )
                return self._execute_test(study, data, source="exportacion")[1]
            config = study.config.model_copy(update={"test_history": study.history_hashes()})
            result = run_pair(config, data, include_test=False)
            result.test_contamination_note = contamination_note(study, included=False)
            return result

    # ------------------------------------------------------------------ prueba

    def _execute_test(
        self, study: Study, data: DataService, *, source: TestRunSource
    ) -> tuple[Study, RunResult]:
        config = study.config
        split = config.frozen_split
        if split is not None and split.n_test == 0:
            raise ConfigError(
                "La división de este estudio no tiene periodo de prueba (0 observaciones): no "
                "hay nada que evaluar fuera de muestra.",
                code="SIN_PERIODO_PRUEBA",
                details={"study_id": study.id},
            )
        history = study.history_hashes()
        result = run_pair(
            config.model_copy(update={"test_history": history}), data, include_test=True
        )
        analysis = result.analysis
        attempted = Segment.TEST in result.backtests or Segment.TEST in result.segment_errors
        if not analysis.strategy_allowed or not attempted:
            raise self._not_testable(result)
        current = result.config.config_hash
        assert current is not None
        now = _now()
        with self._files_lock:
            # Ejecuciones de otros estudios del mismo par vistas ANTES de fijar esta configuración.
            synced = self._sync_foreign_runs(study, _config_fixed_at(study))
            recorded = {r.config_hash for r in study.test_runs}
            if source == "prueba" or current not in recorded or synced:
                self._mark_other_config(study, current, now)
                if source == "prueba" or current not in recorded:
                    study.test_runs.append(StudyTestRun(at=now, config_hash=current, source=source))
                study.updated_at = now
                self._write(study)
        if result.test_contaminated and not study.contaminated:
            # Defensa: el historial del runner detectó otra configuración que el registro no
            # había marcado (no debería ocurrir; se registra igual para no perderlo).
            study.contaminated = True
            study.contamination_reason = (
                "El historial de la configuración registra ejecuciones de la prueba con otra "
                "configuración."
            )
            self._write(study)
        result.test_contaminated = study.contaminated if Segment.TEST in result.backtests else None
        result.test_contamination_note = contamination_note(study)
        return study, result

    @staticmethod
    def _not_testable(result: RunResult) -> AnalysisBlocked:
        analysis = result.analysis
        if analysis.status == PairStatus.INSUFFICIENT_DATA:
            reasons = [c.message for c in analysis.causes]
        else:
            reasons = [
                d.message for d in analysis.discard_reasons if d.code in STRATEGY_BLOCKING_CODES
            ]
        detail = " ".join(reasons)
        return AnalysisBlocked(
            "No hay prueba que ejecutar: la simulación de la estrategia está bloqueada para este "
            "par, así que el estudio no registra ninguna ejecución de la prueba."
            + (f" Motivo: {detail}" if detail else ""),
            code="PRUEBA_NO_SIMULABLE",
            details={"codigos": [d.code for d in analysis.discard_reasons]},
        )


# --------------------------------------------------------------------------- textos


def _times(study: Study, once: str) -> str:
    """«1 vez», «3 veces» o «2 veces (1 en otro estudio del mismo par)»."""
    n = len(study.test_runs)
    text = once if n == 1 else f"{n} veces"
    foreign = sum(r.source == "otro_estudio" for r in study.test_runs)
    if foreign:
        text += f" ({foreign} en otro estudio del mismo par)" if n > 1 else " (en otro estudio)"
    return text


def contamination_note(study: Study, *, included: bool = True) -> str:
    """Origen y motivo de la marca, para la exportación (línea «Prueba contaminada: …»).

    ``included`` = la exportación incluye el segmento de prueba."""
    where = f"según el registro persistente del estudio «{study.name}» ({study.id})"
    times = _times(study, "1 vez")
    reason = study.contamination_reason or "La configuración cambió después de ver la prueba."
    if not included:
        if not study.test_viewed:
            return f"{where}, la prueba aún no se ha ejecutado."
        state = (
            f" y el estudio está marcado como contaminado: {_lower_first(reason)}"
            if study.contaminated
            else "."
        )
        return f"esta exportación no incluye la prueba; {where}, ya se ejecutó {times}{state}"
    if study.contaminated:
        return f"{where}: {_lower_first(reason)}"
    return (
        f"{where}, la prueba se ha ejecutado {times}, siempre con esta configuración, y la "
        "configuración no se cambió después de verla."
    )


def contamination_text(study: Study) -> str:
    """Texto veraz para la interfaz sobre el estado de la prueba del estudio."""
    if study.contaminated:
        reason = study.contamination_reason or "La configuración cambió después de ver la prueba."
        return f"Prueba contaminada. {reason} La marca es permanente y viaja en las exportaciones."
    if not study.test_viewed:
        return (
            "La prueba aún no se ha ejecutado. Al ejecutarla quedará registrada; si después "
            "cambias cualquier parámetro, el estudio quedará marcado como «prueba contaminada»."
        )
    times = _times(study, "una vez")
    return (
        f"Prueba no contaminada: se ejecutó {times}, siempre con la configuración congelada, y la "
        "configuración no se cambió después de verla. Si cambias cualquier parámetro ahora, el "
        "estudio quedará marcado como «prueba contaminada» de forma permanente."
    )
