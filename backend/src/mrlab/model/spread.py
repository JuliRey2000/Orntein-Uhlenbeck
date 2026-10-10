"""Construcción del spread (sección 5).

    Precios:      X_t = P_A,t − α − β·P_B,t
    Logarítmico:  X_t = log(P_A,t) − α − β·log(P_B,t)

"A" es la variable DEPENDIENTE de la dirección elegida en Engle–Granger.

CONTRATO:
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

Notas de implementación:
- Las dos series deben llegar alineadas (mismo ``DatetimeIndex``); si no, se
  lanza ``DataError`` en lugar de alinearlas en silencio.
- ``fit_hedge_ratio`` exige datos sin faltantes; ``compute_spread`` propaga los
  NaN que reciba (no inventa observaciones).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.regression.linear_model import OLS

from mrlab.domain.errors import AnalysisBlocked, DataError
from mrlab.domain.models import SpreadType

SPREAD_UNITS = {
    SpreadType.LOG: "log-puntos (≈ fracción; 0,01 ≈ 1 %)",
    SpreadType.PRICE: "unidades de moneda",
}

#: Mínimo matemático para estimar α, β y sus errores estándar (n − 2 > 0).
MIN_OBS_HEDGE = 3

OLS_INFERENCE_WARNING = (
    "Con series cointegradas, los errores estándar y el R² de MCO no son válidos para la "
    "inferencia convencional: el estimador de β converge más rápido de lo habitual "
    "(superconsistencia) y los errores estándar usuales no lo describen bien. Un R² alto no prueba "
    "que la relación sea real ni estable. Se muestran solo como referencia descriptiva; la "
    "evidencia de cointegración la da la prueba de Engle–Granger."
)


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


def _fmt(x: float, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles (convención de la interfaz)."""
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def transform(prices: pd.Series, spread_type: SpreadType) -> pd.Series:
    """log(precio) para el spread logarítmico; copia en float para el spread en precios."""
    values = prices.astype(float)
    if spread_type == SpreadType.PRICE:
        return values
    non_positive = values <= 0
    if non_positive.any():
        first = values.index[non_positive.to_numpy()][0]
        first_txt = first.date().isoformat() if isinstance(first, pd.Timestamp) else str(first)
        name = f" «{prices.name}»" if prices.name is not None else ""
        raise AnalysisBlocked(
            f"El spread logarítmico requiere precios estrictamente positivos, pero la serie{name} "
            f"tiene {int(non_positive.sum())} precio(s) menores o iguales a cero (el primero el "
            f"{first_txt}). Revisa los datos o usa el spread en precios.",
            code="SPREAD_LOG_BLOQUEADO",
            details={
                "serie": None if prices.name is None else str(prices.name),
                "n_no_positivos": int(non_positive.sum()),
                "primera_fecha": first_txt,
            },
        )
    return np.log(values)


def _check_aligned(y_prices: pd.Series, x_prices: pd.Series) -> None:
    if len(y_prices) != len(x_prices) or not y_prices.index.equals(x_prices.index):
        raise DataError(
            "Las dos series del spread deben estar alineadas en las mismas fechas "
            f"({len(y_prices)} y {len(x_prices)} observaciones con índices distintos)."
        )


def _interpretation(spread_type: SpreadType, beta: float, dependent: str, independent: str) -> str:
    a, b, beta_txt = dependent, independent, _fmt(beta)
    if spread_type == SpreadType.LOG:
        text = (
            f"β ≈ elasticidad: un 1 % en {b} se asocia con un {beta_txt} % en {a} "
            "(relación promedio estimada en la muestra de formación). "
            f"El spread es log({a}) − α − β·log({b}), en log-puntos (0,01 ≈ 1 %)."
        )
    else:
        text = (
            f"β = {beta_txt}: {beta_txt} unidades de {b} por cada unidad de {a}. "
            f"El spread es {a} − α − β·{b}, en unidades de moneda."
        )
    if beta < 0:
        text += (
            f" Como β es negativo, en la relación estimada {a} y {b} se mueven en sentidos "
            "opuestos, así que las dos piernas van en la misma dirección: comprar el spread "
            f"significa comprar {a} y también comprar {b}; venderlo significa vender ambos."
        )
    return text


def fit_hedge_ratio(
    y_prices: pd.Series,
    x_prices: pd.Series,
    spread_type: SpreadType,
    dependent: str,
    independent: str,
) -> HedgeFit:
    """MCO de f(y) = α + β·f(x) + residuo con TODOS los datos recibidos (solo formación)."""
    _check_aligned(y_prices, x_prices)
    if not isinstance(y_prices.index, pd.DatetimeIndex):
        raise DataError("Las series del spread deben estar indexadas por fecha.")
    fy = transform(y_prices, spread_type).to_numpy()
    fx = transform(x_prices, spread_type).to_numpy()
    if not (np.isfinite(fy).all() and np.isfinite(fx).all()):
        raise DataError(
            "Las series de formación contienen valores faltantes o no finitos; el hedge ratio "
            "se estima solo con datos limpios."
        )
    n = fy.size
    if n < MIN_OBS_HEDGE:
        raise DataError(
            f"Se necesitan al menos {MIN_OBS_HEDGE} observaciones para estimar α y β; hay {n}."
        )
    if fx.max() == fx.min() or fy.max() == fy.min():
        raise DataError("Una de las series es constante en la muestra: no se puede estimar β.")

    ols = OLS(fy, np.column_stack([np.ones(n), fx])).fit()
    alpha, beta = (float(v) for v in ols.params)
    alpha_se, beta_se = (float(v) for v in ols.bse)

    return HedgeFit(
        spread_type=spread_type,
        dependent=dependent,
        independent=independent,
        alpha=alpha,
        beta=beta,
        alpha_se=alpha_se,
        beta_se=beta_se,
        r_squared=float(ols.rsquared),
        nobs=n,
        estimation_start=y_prices.index[0].date(),
        estimation_end=y_prices.index[-1].date(),
        units=SPREAD_UNITS[spread_type],
        interpretation=_interpretation(spread_type, beta, dependent, independent),
        warnings=[OLS_INFERENCE_WARNING],
    )


def compute_spread(y_prices: pd.Series, x_prices: pd.Series, fit: HedgeFit) -> pd.Series:
    """X_t = f(y_t) − α − β·f(x_t) con α y β FIJOS del ajuste (no se re-estiman)."""
    _check_aligned(y_prices, x_prices)
    fy = transform(y_prices, fit.spread_type)
    fx = transform(x_prices, fit.spread_type)
    spread = fy - fit.alpha - fit.beta * fx
    spread.name = "spread"
    return spread
