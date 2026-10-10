"""Construcción del spread (sección 5).

    Precios:      X_t = P_A,t − α − β·P_B,t
    Logarítmico:  X_t = log(P_A,t) − α − β·log(P_B,t)

"A" es la variable DEPENDIENTE de la dirección elegida en Engle–Granger.

CONTRATO (a implementar):
- ``transform(prices, spread_type)``: log o identidad; con LOG y algún precio
  ≤ 0 → AnalysisBlocked(code="SPREAD_LOG_BLOQUEADO").
- ``fit_hedge_ratio``: MCO de f(y) sobre [1, f(x)] SOLO con los datos que se le
  pasan (el llamador entrega la muestra de formación). α, β, errores estándar
  MCO, R², nobs, fechas de inicio/fin. ``interpretation`` en español:
  log → "β ≈ elasticidad: un 1 % en B se asocia con un β % en A" (con el número);
  precio → "β unidades de B por cada unidad de A". Si β < 0, explicar que ambas
  piernas van en la misma dirección. ``warnings`` SIEMPRE incluye que, con
  series cointegradas, los errores estándar y el R² de MCO no son válidos para
  inferencia convencional.
- ``compute_spread(y, x, fit)``: aplica α, β FIJOS a cualquier periodo
  (formación, validación o prueba). Nombre de la serie: "spread".
- Consistencia α–μ: como α ya está dentro del spread, el residuo de formación
  tiene media ≈ 0; el μ del OU se estima sobre ese residuo y se usa tal cual.
  No restar dos veces el mismo nivel.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel

from mrlab.domain.models import SpreadType

SPREAD_UNITS = {
    SpreadType.LOG: "log-puntos (≈ fracción; 0,01 ≈ 1 %)",
    SpreadType.PRICE: "unidades de moneda",
}


class HedgeFit(BaseModel):
    spread_type: SpreadType
    dependent: str
    independent: str
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    r_squared: float
    nobs: int
    method: str = "MCO (solo formación)"
    estimation_start: date
    estimation_end: date
    units: str
    interpretation: str
    warnings: list[str] = []


def transform(prices: pd.Series, spread_type: SpreadType) -> pd.Series:
    raise NotImplementedError


def fit_hedge_ratio(
    y_prices: pd.Series,
    x_prices: pd.Series,
    spread_type: SpreadType,
    dependent: str,
    independent: str,
) -> HedgeFit:
    raise NotImplementedError


def compute_spread(y_prices: pd.Series, x_prices: pd.Series, fit: HedgeFit) -> pd.Series:
    raise NotImplementedError
