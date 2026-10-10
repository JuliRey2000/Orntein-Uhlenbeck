"""Engle–Granger en ambas direcciones y estabilidad por ventanas (4.2 y 4.4).

CONTRATO (a implementar):
- ``engle_granger(y, x, ...)``: p-value y estadístico EXCLUSIVAMENTE de
  ``statsmodels.tsa.stattools.coint(y, x, trend="c", maxlag=12, autolag="aic")``
  (p-values de MacKinnon para residuos estimados, N=2). PROHIBIDO usar el p-value
  de un ADF ordinario sobre los residuos. Los rezagos elegidos se obtienen
  replicando el paso interno de ``coint``: OLS y ~ const + x, luego
  ``adfuller(resid, maxlag, autolag="aic", regression="n")[2]``. ``alpha_ols`` y
  ``beta_ols`` = coeficientes de esa misma regresión. Valores críticos 1/5/10 %
  de ``coint``. Las series de entrada ya están transformadas (log o precio),
  alineadas y limpias.
- ``engle_granger_both``: A sobre B (dependiente A) y B sobre A (dependiente B).
- ``choose_direction(results, override=None)``: menor p-value (empate → A_ON_B)
  salvo que el usuario la fije.
- ``rolling_eg_stability``: ventanas ``mrlab.stats.windows.rolling_windows``
  sobre la muestra de FORMACIÓN, dirección fija. Una ventana falla si su p-value
  (sin ajustar: es diagnóstico) > ``alpha_level``. ``unstable`` si
  frac_fail > max_fail_frac (0,40) o si β cambia de signo entre ventanas.
  ``reasons`` en español con los números. Sin ventanas → unstable=False y una
  razón explicándolo.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel


class Direction(StrEnum):
    A_ON_B = "a_sobre_b"  # A dependiente: A = α + β·B + residuo
    B_ON_A = "b_sobre_a"  # B dependiente: B = α + β·A + residuo


class EGResult(BaseModel):
    direction: Direction
    dependent: str
    independent: str
    statistic: float
    pvalue: float
    critical_values: dict[str, float]  # {"1%": ..., "5%": ..., "10%": ...}
    nobs: int
    trend: str = "c"
    lags: int
    maxlag: int
    autolag: str = "aic"
    alpha_ols: float
    beta_ols: float
    method: str = "Engle–Granger (statsmodels.coint, p-values de MacKinnon para residuos estimados)"


class EGWindow(BaseModel):
    start: date
    end: date
    nobs: int
    statistic: float
    pvalue: float
    alpha: float
    beta: float
    passed: bool


class EGStability(BaseModel):
    direction: Direction
    window: int
    step: int
    alpha_level: float
    max_fail_frac: float
    windows: list[EGWindow]
    frac_fail: float | None
    beta_sign_change: bool
    unstable: bool
    reasons: list[str]


def engle_granger(
    y: pd.Series,
    x: pd.Series,
    *,
    direction: Direction,
    dependent: str,
    independent: str,
    maxlag: int = 12,
    trend: str = "c",
) -> EGResult:
    raise NotImplementedError


def engle_granger_both(
    fa: pd.Series, fb: pd.Series, symbol_a: str, symbol_b: str, maxlag: int = 12
) -> dict[Direction, EGResult]:
    raise NotImplementedError


def choose_direction(results: dict[Direction, EGResult], override: Direction | None = None) -> Direction:
    raise NotImplementedError


def rolling_eg_stability(
    y: pd.Series,
    x: pd.Series,
    *,
    direction: Direction,
    dependent: str,
    independent: str,
    window: int = 252,
    step: int = 63,
    alpha_level: float = 0.05,
    max_fail_frac: float = 0.40,
    maxlag: int = 12,
) -> EGStability:
    raise NotImplementedError
