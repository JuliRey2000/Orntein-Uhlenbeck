"""Configuración reproducible de una ejecución (sección 15).

CONTRATO:
- ``canonical_json(obj)``: JSON con claves ordenadas, separadores compactos
  (",", ":"), ``ensure_ascii=False``, fechas ISO, floats con ``repr``; acepta
  modelos pydantic (``model_dump(mode="json")``).
- ``config_hash(cfg)``: sha256 hex de ``canonical_json`` del RunConfig SIN los
  campos ``config_hash`` y ``code_version`` (la identidad de la configuración no
  depende de la versión del código, que se registra aparte).
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
from mrlab.domain.errors import ConfigError
from mrlab.domain.models import DataOrigin, SeriesMeta
from mrlab.strategy.params import CostParams, StrategyParams

SCHEMA_VERSION = 1
UNKNOWN_VERSION = "desconocido"
_HASH_EXCLUDED = {"config_hash", "code_version"}
_GIT_TIMEOUT_S = 5


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
    data_origin: DataOrigin | None = None
    series: list[SeriesMeta] = Field(default_factory=list)
    code_version: str | None = None
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


def config_hash(cfg: RunConfig) -> str:
    """Identidad de la configuración (sin ``config_hash`` ni ``code_version``)."""
    payload = cfg.model_dump(mode="json", exclude=_HASH_EXCLUDED)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


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
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise ConfigError(
            f"El archivo {path.name} usa la versión de esquema {version!r}, más nueva que la que "
            f"entiende esta versión del laboratorio ({SCHEMA_VERSION}). Actualiza el laboratorio.",
            code="RUN_CONFIG_VERSION",
            details={"path": str(path), "schema_version": version},
        )
    try:
        cfg = RunConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:5]
        )
        raise ConfigError(
            f"El archivo {path.name} no tiene el formato de un run_config.json ({problems}).",
            code="RUN_CONFIG_INVALIDO",
            details={"path": str(path)},
        ) from exc

    expected = config_hash(cfg)
    if cfg.config_hash is None:
        return cfg.model_copy(update={"config_hash": expected})
    if cfg.config_hash != expected:
        raise ConfigError(
            f"El hash de {path.name} no coincide con su contenido (guardado {cfg.config_hash[:12]}…, "
            f"calculado {expected[:12]}…): el archivo se modificó después de exportarlo. Para "
            "reproducir un análisis usa el archivo original; si quieres cambiar parámetros, hazlo "
            "desde el laboratorio para obtener una configuración nueva con su propio hash.",
            code="RUN_CONFIG_HASH_DISTINTO",
            details={"path": str(path), "guardado": cfg.config_hash, "calculado": expected},
        )
    return cfg
