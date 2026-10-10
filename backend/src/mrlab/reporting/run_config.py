"""Configuración reproducible de una ejecución (sección 15).

CONTRATO:
- ``canonical_json(obj)``: JSON con claves ordenadas, separadores compactos
  (",", ":"), ``ensure_ascii=False``, fechas ISO, floats con ``repr``; acepta
  modelos pydantic (``model_dump(mode="json")``).
- ``config_hash(cfg)``: sha256 hex de ``canonical_json`` del RunConfig SIN los
  campos ``config_hash``, ``code_version``, ``series`` y ``test_history`` (ver
  ``HASH_EXCLUDED``).
- ``git_version()``: variable de entorno ``GIT_COMMIT`` si existe; si no
  ``git rev-parse HEAD`` (+ "-dirty" si hay cambios sin confirmar); si falla,
  "desconocido". Nunca lanza excepción.
- ``save_run_config(cfg, path)`` escribe JSON legible (indentado, claves
  ordenadas, LF, con ``config_hash`` rellenado); ``load_run_config(path)`` lo
  lee y verifica el hash (ConfigError si no coincide: archivo editado a mano).

Precisiones de la implementación:
- NaN e infinitos no son JSON válido: ``canonical_json`` los rechaza con
  ConfigError en lugar de escribir ``NaN``.
- «Cambios sin confirmar» = archivos versionados modificados (como
  ``git describe --dirty``); los archivos nuevos sin versionar no cuentan.
- Un ``run_config.json`` escrito a mano SIN ``config_hash`` se acepta (no hay
  nada que verificar) y se le calcula el hash al cargarlo; uno con hash que no
  coincide se rechaza. Un ``schema_version`` mayor que el soportado también.
- Identidad (``config_hash``): ``series`` se excluye entero porque solo describe
  los datos (fecha de descarga o lectura, cobertura, notas) y las referencias
  fijadas por ``snapshot_id`` en ``pair`` ya identifican el contenido; así los
  mismos CSV registrados en momentos distintos dan el mismo hash. ``test_history``
  se excluye porque es un registro de ejecuciones, no un parámetro.
- ``include_test`` (forma parte del hash): si la ejecución incluye el segmento
  de prueba. ``test_history``: hashes de las configuraciones con las que se ha
  ejecutado la prueba; el runner lo usa para marcar «prueba contaminada»
  (decisión S10, versión mínima de la fase 1: el registro viaja dentro del
  archivo; el registro persistente de estudios llega en la fase 2).
- Versión de esquema 2 (añade ``include_test`` y ``test_history`` y cambia la
  definición del hash). Un archivo de la versión 1 se verifica con la
  definición de entonces (el JSON tal como se guardó, sin
  ``config_hash`` ni ``code_version``) y se devuelve actualizado a la versión 2
  con su hash nuevo; si no coincide, se rechaza como cualquier archivo editado.
- Errores de formato: mensaje propio en español con la ruta de cada campo
  (p. ej. ``strategy.entry_z``), el problema y el valor recibido; los detalles
  (``details["errores"]``) llevan lo mismo estructurado. Un error de
  coherencia de los parámetros (p. ej. ``UMBRALES_INCONSISTENTES`` o
  ``COSTES_INVALIDOS``) conserva su código y nombra el archivo.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from mrlab.analysis.pair_pipeline import AnalysisConfig
from mrlab.backtest.split import SampleSplit
from mrlab.domain.errors import ConfigError, LabError
from mrlab.domain.models import DataOrigin, SeriesMeta
from mrlab.strategy.params import CostParams, StrategyParams

SCHEMA_VERSION = 2
UNKNOWN_VERSION = "desconocido"
#: Campos que NO forman parte de la identidad de la configuración.
HASH_EXCLUDED = frozenset({"config_hash", "code_version", "series", "test_history"})
#: Definición del hash en la versión 1 del esquema (solo para verificar archivos antiguos).
_HASH_EXCLUDED_V1 = frozenset({"config_hash", "code_version"})
_GIT_TIMEOUT_S = 5
_MAX_REPORTED_ERRORS = 5
_MAX_VALUE_CHARS = 60


class SeriesRef(BaseModel):
    provider: str
    symbol: str
    snapshot_id: str | None = None
    start: str | None = None  # ISO AAAA-MM-DD
    end: str | None = None


class PairSpec(BaseModel):
    a: SeriesRef
    b: SeriesRef
    exec_a: SeriesRef | None = None  # proxy ejecutable (índices/futuros)
    exec_b: SeriesRef | None = None


class RunConfig(BaseModel):
    schema_version: int = SCHEMA_VERSION
    pair: PairSpec
    analysis: AnalysisConfig = AnalysisConfig()
    strategy: StrategyParams = StrategyParams()
    costs: CostParams = CostParams()
    frozen_split: SampleSplit | None = None
    seeds: dict[str, int] = Field(default_factory=dict)
    include_test: bool = False  # ¿la ejecución incluye el segmento de prueba?
    data_origin: DataOrigin | None = None
    series: list[SeriesMeta] = Field(default_factory=list)  # fuera del hash
    test_history: list[str] = Field(default_factory=list)  # fuera del hash (decisión S10)
    code_version: str | None = None  # fuera del hash
    config_hash: str | None = None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    raise TypeError(f"Tipo no serializable a JSON: {type(obj).__name__}")


def _to_jsonable(obj: Any) -> Any:
    return obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj


def _dumps(obj: Any, *, indent: int | None) -> str:
    separators = (",", ":") if indent is None else (",", ": ")
    try:
        return json.dumps(
            _to_jsonable(obj),
            sort_keys=True,
            separators=separators,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            default=_json_default,
        )
    except ValueError as exc:
        raise ConfigError(
            "La configuración contiene valores no finitos (NaN o infinito), que no se pueden "
            "guardar en JSON. Revisa los parámetros.",
            code="RUN_CONFIG_NO_SERIALIZABLE",
            details={"error": str(exc)},
        ) from exc


def canonical_json(obj: Any) -> str:
    """JSON canónico: claves ordenadas, sin espacios, UTF-8 legible, floats con ``repr``."""
    return _dumps(obj, indent=None)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def config_hash(cfg: RunConfig) -> str:
    """Identidad de la configuración (sin los campos de ``HASH_EXCLUDED``)."""
    return _sha256(cfg.model_dump(mode="json", exclude=set(HASH_EXCLUDED)))


def _legacy_v1_hash(cfg: RunConfig) -> str:
    """Hash de la versión 1: el JSON tal como se guardó (campos presentes en el archivo),
    sin ``config_hash`` ni ``code_version``."""
    return _sha256(cfg.model_dump(mode="json", exclude=set(_HASH_EXCLUDED_V1), exclude_unset=True))


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=True,
    )
    return result.stdout.strip()


def git_version() -> str:
    """Hash del commit (``-dirty`` si hay cambios sin confirmar) o ``GIT_COMMIT``."""
    env = os.environ.get("GIT_COMMIT", "").strip()
    if env:
        return env
    try:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError, ValueError):
        return UNKNOWN_VERSION
    if not head:
        return UNKNOWN_VERSION
    return f"{head}-dirty" if dirty else head


def save_run_config(cfg: RunConfig, path: Path | str) -> Path:
    """Escribe ``run_config.json`` legible (indentado, claves ordenadas, LF) con su hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(run_config_bytes(cfg))
    return path


def run_config_bytes(cfg: RunConfig) -> bytes:
    """Mismo contenido que escribe ``save_run_config`` (para exportaciones)."""
    stamped = cfg.model_copy(update={"config_hash": config_hash(cfg)})
    return (_dumps(stamped, indent=2) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- errores de formato

#: Tipo de error de pydantic → problema en español (``{...}`` se rellena con ``ctx``).
_PROBLEMS_ES: dict[str, str] = {
    "missing": "falta este campo",
    "extra_forbidden": "campo no reconocido",
    "float_parsing": "debe ser un número",
    "float_type": "debe ser un número",
    "finite_number": "debe ser un número finito",
    "int_parsing": "debe ser un número entero",
    "int_type": "debe ser un número entero",
    "int_from_float": "debe ser un número entero (sin decimales)",
    "int_parsing_size": "es un número entero demasiado grande",
    "bool_parsing": "debe ser true o false",
    "bool_type": "debe ser true o false",
    "string_type": "debe ser un texto entre comillas",
    "dict_type": "debe ser un objeto JSON ({…})",
    "model_type": "debe ser un objeto JSON ({…})",
    "model_attributes_type": "debe ser un objeto JSON ({…})",
    "list_type": "debe ser una lista JSON ([…])",
    "date_type": "debe ser una fecha AAAA-MM-DD",
    "date_parsing": "debe ser una fecha AAAA-MM-DD",
    "date_from_datetime_parsing": "debe ser una fecha AAAA-MM-DD",
    "date_from_datetime_inexact": "debe ser una fecha AAAA-MM-DD (sin hora)",
    "datetime_type": "debe ser una fecha y hora ISO (AAAA-MM-DDTHH:MM:SS)",
    "datetime_parsing": "debe ser una fecha y hora ISO (AAAA-MM-DDTHH:MM:SS)",
    "datetime_from_date_parsing": "debe ser una fecha y hora ISO (AAAA-MM-DDTHH:MM:SS)",
    "greater_than": "debe ser mayor que {gt}",
    "greater_than_equal": "debe ser mayor o igual que {ge}",
    "less_than": "debe ser menor que {lt}",
    "less_than_equal": "debe ser como máximo {le}",
    "literal_error": "debe ser uno de estos valores: {expected}",
    "enum": "debe ser uno de estos valores: {expected}",
    "string_too_short": "es demasiado corto",
    "string_too_long": "es demasiado largo",
    "too_short": "tiene muy pocos elementos",
    "too_long": "tiene demasiados elementos",
}


def _field_path(loc: tuple[Any, ...]) -> str:
    text = ""
    for part in loc:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text or "configuración"


def _ctx_text(value: Any) -> str:
    text = str(value).replace(" or ", " o ")
    return text.replace(".", ",") if isinstance(value, float) else text


def _problem(err: dict[str, Any]) -> str:
    kind = err.get("type", "")
    ctx = err.get("ctx") or {}
    if kind in ("value_error", "assertion_error") and "error" in ctx:
        return str(ctx["error"])  # mensaje de un validador propio
    template = _PROBLEMS_ES.get(kind)
    if template is None:
        return f"valor no válido (tipo de error «{kind}»)"
    try:
        return template.format(**{k: _ctx_text(v) for k, v in ctx.items()})
    except (KeyError, IndexError):
        return template


def _received(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    return text if len(text) <= _MAX_VALUE_CHARS else text[: _MAX_VALUE_CHARS - 1] + "…"


def _validation_problems(exc: ValidationError) -> list[dict[str, str | None]]:
    problems = []
    for err in exc.errors():
        missing = err.get("type") == "missing"
        problems.append(
            {
                "campo": _field_path(tuple(err.get("loc", ()))),
                "problema": _problem(err),
                "recibido": None if missing else _received(err.get("input")),
            }
        )
    return problems


def _describe(problem: dict[str, str | None]) -> str:
    text = f"«{problem['campo']}» {problem['problema']}"
    if problem["recibido"] is not None:
        text += f" (recibido: {problem['recibido']})"
    return text


# --------------------------------------------------------------------------- cargar


def load_run_config(path: Path | str) -> RunConfig:
    """Lee un ``run_config.json`` y verifica que su hash corresponda a su contenido."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(
            f"No se encontró el archivo de configuración {path}.",
            code="RUN_CONFIG_NO_ENCONTRADO",
            details={"path": str(path)},
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"El archivo {path.name} no es un JSON válido ({exc}). Si lo editaste a mano, revisa "
            "comas y comillas, o vuelve a exportarlo desde el laboratorio.",
            code="RUN_CONFIG_INVALIDO",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"El archivo {path.name} no contiene una configuración (se esperaba un objeto JSON).",
            code="RUN_CONFIG_INVALIDO",
            details={"path": str(path)},
        )
    version = raw.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool) or version > SCHEMA_VERSION:
        raise ConfigError(
            f"El archivo {path.name} usa la versión de esquema {version!r}, más nueva que la que "
            f"entiende esta versión del laboratorio ({SCHEMA_VERSION}). Actualiza el laboratorio.",
            code="RUN_CONFIG_VERSION",
            details={"path": str(path), "schema_version": version},
        )
    try:
        cfg = RunConfig.model_validate(raw)
    except ValidationError as exc:
        problems = _validation_problems(exc)
        listed = "; ".join(_describe(p) for p in problems[:_MAX_REPORTED_ERRORS])
        if len(problems) > _MAX_REPORTED_ERRORS:
            listed += f"; y {len(problems) - _MAX_REPORTED_ERRORS} problema(s) más"
        raise ConfigError(
            f"El archivo {path.name} no tiene el formato de un run_config.json: {listed}. Si lo "
            "editaste a mano, corrige esos campos o vuelve a exportarlo desde el laboratorio.",
            code="RUN_CONFIG_INVALIDO",
            details={"path": str(path), "errores": problems},
        ) from None
    except LabError as exc:  # coherencia de parámetros (umbrales, costes…)
        raise ConfigError(
            f"El archivo {path.name} tiene parámetros no válidos: {exc.message}",
            code=exc.code,
            details={**exc.details, "path": str(path)},
        ) from exc

    legacy = cfg.schema_version < SCHEMA_VERSION
    # Un archivo de la versión 1 se verifica con la definición del hash de entonces.
    stored_def = _legacy_v1_hash(cfg) if legacy else config_hash(cfg)
    if cfg.config_hash is not None and cfg.config_hash != stored_def:
        raise ConfigError(
            f"El hash de {path.name} no coincide con su contenido (guardado {cfg.config_hash[:12]}…, "
            f"calculado {stored_def[:12]}…): el archivo se modificó después de exportarlo. Para "
            "reproducir un análisis usa el archivo original; si quieres cambiar parámetros, hazlo "
            "desde el laboratorio para obtener una configuración nueva con su propio hash.",
            code="RUN_CONFIG_HASH_DISTINTO",
            details={"path": str(path), "guardado": cfg.config_hash, "calculado": stored_def},
        )
    if legacy:
        cfg = cfg.model_copy(update={"schema_version": SCHEMA_VERSION})
    expected = config_hash(cfg)
    if cfg.config_hash != expected:
        cfg = cfg.model_copy(update={"config_hash": expected})
    return cfg
