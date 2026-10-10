"""Ejecución completa desde un RunConfig: datos (caché) → análisis → backtests.

CONTRATO (a implementar):
- ``run_pair(config, data, include_test=False)``:
  carga las series con ``data.load_snapshot`` si el RunConfig trae snapshot_id
  (reproducción exacta, sin red) o con ``data.load`` si no; congela el split
  (si config.frozen_split es None se calcula y se guarda en el RunConfig
  devuelto); analiza; si ``strategy_allowed`` ejecuta el backtest de
  FORMACIÓN (marcado como dentro de muestra, solo ilustrativo) y VALIDACIÓN; la
  PRUEBA solo si ``include_test`` (se ejecuta una vez con la configuración
  congelada). Rellena ``config.series`` con los SeriesMeta usados y
  ``config.code_version``. Añade a ``discard_reasons`` "costes superan el PnL
  bruto" si ocurre en validación.
- El z de cada segmento sale de la serie z completa (parámetros de formación):
  no requiere calentamiento y no usa información futura.
- Anualización: ``mrlab.data.calendar.periods_per_year`` del calendario del par.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mrlab.analysis.pair_pipeline import PairAnalysis
from mrlab.backtest.engine import BacktestResult
from mrlab.data.service import DataService
from mrlab.domain.models import Segment
from mrlab.reporting.run_config import RunConfig


@dataclass
class RunResult:
    config: RunConfig
    analysis: PairAnalysis
    backtests: dict[Segment, BacktestResult] = field(default_factory=dict)
    is_synthetic: bool = False


def run_pair(config: RunConfig, data: DataService, include_test: bool = False) -> RunResult:
    raise NotImplementedError
