"""Análisis: correlaciones y análisis completo de un par (cointegración, spread, OU).

CONTRATO (a implementar):
- POST /correlations  CorrelationRequest {series: list[SeriesRef] (2..30), window: int=63,
  pair: [i, j] | None (índices del par para la correlación móvil, los rendimientos y la
  dispersión; por defecto el primero con el segundo)} → CorrelationOut {symbols,
  pearson: list[list[float|None]], spearman, nobs: list[list[int]], normalized: {dates,
  series: {symbol: values}} (base 100 en la intersección), rolling: SeriesOut, returns:
  {dates, a, b} (log-rendimientos del par sobre fechas comunes), concepts (textos de los tres
  conceptos), note (la correlación es un filtro exploratorio, nunca prueba de cointegración),
  synthetic, disclaimer}.
- POST /pairs/analyze  RunConfig (include_test debe ser False; si es True → 400 en español que
  remite a /api/studies) → AnalysisOut {config: RunConfig (fijado: snapshot_id y split),
  analysis: PairAnalysisOut, synthetic, disclaimer}.
  PairAnalysisOut serializa PairAnalysis: symbol_a, symbol_b, exec_symbol_a/b, status,
  status_label, causes, discard_reasons, strategy_allowed, illustrative_only,
  illustrative_banner, illustrative_reasons, validation, calendar, time_unit, split,
  integration (por símbolo), eg (lista con dirección), eg_p_adjusted, eg_decision,
  chosen_direction, direction_was_auto, stability, hedge, ou, forecasts, residuals,
  ou_stability, correlation_train, warnings, y series para gráficos: prices {dates, a, b}
  (alineados), exec_prices (si hay proxy), spread: SeriesOut, zscore: SeriesOut | None,
  bands: BandsOut | None {mu, sigma_eq, upper1, lower1, upper2, lower2} — bands y zscore
  son None si el OU NO es válido (prohibición: no mostrar vida media ni bandas con un OU
  inválido). Nunca lenguaje promocional.
  Implementación: reutiliza el runner para cargar y fijar datos (analysis.runner.run_pair con
  include_test=False devuelve config fijado + analysis; puedes añadir en runner una función
  más ligera que no simule backtests si hace falta, sin cambiar run_pair).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["análisis"])
