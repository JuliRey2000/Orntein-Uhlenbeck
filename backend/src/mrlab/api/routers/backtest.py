"""Backtest de formación y validación, y comparación entre pares.

CONTRATO (a implementar):
- POST /backtest  RunConfig (include_test=True → 400 en español: la prueba solo por estudios)
  → BacktestOut {config (fijado), status, status_label, strategy_allowed, illustrative_only,
  illustrative_banner, discard_reasons, segments: {formacion, validacion} → BacktestSegmentOut,
  synthetic, disclaimer, execution_convention}. Si la estrategia no está permitida: segments
  vacío y ``blocked_reason`` en español (no es un error HTTP).
  BacktestSegmentOut {segment, daily: ColumnsOut (columnas DAILY_COLUMNS), trades: list[Trade],
  events: list[Event], metrics: Metrics, metric_help (METRIC_HELP), warnings, max_holding_bars,
  in_sample: bool (True para formación), legs {y: {symbol, asset_class}, x: {...}}, beta,
  spread_type, calendar, periods_per_year, error (si el segmento no se pudo simular)}.
- POST /backtest/compare  CompareRequest {configs: list[RunConfig] (2..10)} → JobAccepted
  {job_id}: trabajo en segundo plano (JobManager) que analiza y simula cada par con su propio
  capital; resultado: list[CompareItem] {label, status, status_label, strategy_allowed,
  illustrative_banner, validation_metrics: Metrics | None, error: ErrorBody | None} — nunca
  ordenado por PnL (orden de entrada). Progreso por par.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["backtest"])
