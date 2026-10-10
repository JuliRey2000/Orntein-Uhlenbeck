"""Configuración reproducible de una ejecución (sección 15).

CONTRATO (a implementar):
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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mrlab.analysis.pair_pipeline import AnalysisConfig
from mrlab.backtest.split import SampleSplit
from mrlab.domain.models import DataOrigin, SeriesMeta
from mrlab.strategy.params import CostParams, StrategyParams

SCHEMA_VERSION = 1


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


def canonical_json(obj: Any) -> str:
    raise NotImplementedError


def config_hash(cfg: RunConfig) -> str:
    raise NotImplementedError


def git_version() -> str:
    raise NotImplementedError


def save_run_config(cfg: RunConfig, path: Path | str) -> Path:
    raise NotImplementedError


def load_run_config(path: Path | str) -> RunConfig:
    raise NotImplementedError
