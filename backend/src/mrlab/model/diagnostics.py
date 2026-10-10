"""Diagnóstico del ajuste OU (6.2): residuos AR(1), ACF, Ljung–Box y estabilidad por ventanas.

CONTRATO (a implementar):
- ``ar1_residuals(spread, fit)``: ε_t = X_{t+1} − a − b·X_t con a, b del ajuste
  (también si el OU no es válido, mientras a y b existan).
- ``residual_diagnostics``: ACF de 1..nlags (statsmodels ``acf``, fft=False),
  banda ±1,96/√n, Ljung–Box en ``lb_lags`` (``acorr_ljungbox``). Si el p-value
  de Ljung–Box en el rezago mayor < 0,05 → ``autocorrelation_detected`` y
  ``message`` en español explicando que el AR(1) no captura toda la dinámica.
- ``rolling_ou_stability(spread, window, step)``: ``fit_ou`` sobre cada
  ventana del spread de FORMACIÓN (α, β fijos: es el spread que se opera).
  ``cv_lambda`` = desviación/media de λ sobre las ventanas válidas (None si
  hay < 2 válidas). ``message`` en español.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel

from mrlab.model.ou import OUFit


class LjungBoxResult(BaseModel):
    lag: int
    statistic: float
    pvalue: float


class ResidualDiagnostics(BaseModel):
    nlags: int
    acf: list[float]  # rezagos 1..nlags
    conf_band: float
    ljung_box: list[LjungBoxResult]
    residual_mean: float
    residual_std: float
    autocorrelation_detected: bool
    message: str


class OUWindow(BaseModel):
    start: date
    end: date
    nobs: int
    valid: bool
    lam: float | None = None
    mu: float | None = None
    sigma: float | None = None
    sigma_eq: float | None = None
    half_life: float | None = None


class OUStability(BaseModel):
    window: int
    step: int
    windows: list[OUWindow]
    n_invalid: int
    cv_lambda: float | None
    message: str


def ar1_residuals(spread: pd.Series, fit: OUFit) -> pd.Series:
    raise NotImplementedError


def residual_diagnostics(
    spread: pd.Series, fit: OUFit, nlags: int = 20, lb_lags: tuple[int, ...] = (10, 20)
) -> ResidualDiagnostics:
    raise NotImplementedError


def rolling_ou_stability(spread: pd.Series, window: int = 252, step: int = 63, dt: float = 1.0) -> OUStability:
    raise NotImplementedError
