"""Proceso Ornstein–Uhlenbeck: estimación exacta discreta, pronóstico y simulación.

    dX_t = λ(μ − X_t) dt + σ dW_t        (λ ≡ θ en otras notaciones)

Estimación exacta con observaciones equiespaciadas (Δt = 1 observación):

    X_{t+Δt} = a + b·X_t + ε_t,  ε_t ~ N(0, s²)
    si 0 < b < 1:  λ = −ln(b)/Δt,  μ = a/(1 − b),  σ = sqrt(2λ·s²/(1 − b²)),
                   σ_eq = σ/sqrt(2λ),  vida media = ln(2)/λ

CONTRATO (a implementar, salvo ``simulate_ou_exact`` que ya está hecho):
- ``fit_ou``: AR(1) por MCO con constante sobre el spread de formación;
  ``s2 = SRC/(n−2)``. Si b ≥ 1 → status NON_STATIONARY; si b ≤ 0 →
  NON_POSITIVE_B; menos de 30 puntos → INSUFFICIENT. En todos esos casos
  ``valid=False`` y lam/mu/sigma/sigma_eq/half_life/mu_se = None, con
  ``message`` en español ("el spread no muestra reversión estacionaria
  compatible con un OU"). Nunca producir números engañosos.
  Advertencias (si válido): vida media > 126 → demasiado lenta para operar con
  este horizonte; vida media < 1 → no operable ni fiable con datos diarios.
  ``mu_se = σ_eq·sqrt((1+b)/((1−b)·n))`` (error estándar asintótico de μ̂).
- ``conditional_mean/variance``: E[X_{t+h}|X_t] = μ + (X_t − μ)e^{−λh};
  Var = σ_eq²(1 − e^{−2λh}). ``forecast`` → intervalos simétricos normales al
  ``level`` dado. Si el ajuste no es válido → ``ModelError``.
- ``zscore``: (X_t − μ)/σ_eq con μ, σ_eq FIJOS (de formación). ModelError si inválido.
- ``rolling_zscore``: comparación pedagógica, ventana móvil hacia atrás que
  incluye t (sin información futura); NaN hasta tener ``window`` observaciones.
- ``simulate_paths_summary``: para la pestaña de simulación (máx. 500
  trayectorias, ConfigError si se excede). Tiempo hasta cruzar μ = primer paso
  k ≥ 1 en que (X_k − μ) cambia de signo o se anula respecto a (X_0 − μ)
  (detección discreta: sesgo hacia arriba, explicarlo en ``notes``); exige
  X₀ ≠ μ (ConfigError). Vida media empírica = primer h en que la media
  muestral de (X_h − μ) cae a la mitad de (X_0 − μ), interpolando linealmente;
  None si no ocurre dentro del horizonte.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mrlab.domain.errors import ConfigError

TIME_UNIT_DEFAULT = "día de negociación"
MAX_SIM_PATHS = 500
HALF_LIFE_MAX_WARN = 126.0
HALF_LIFE_MIN_WARN = 1.0
MIN_OBS_OU = 30


class OUStatus(str, Enum):
    VALID = "valido"
    NON_STATIONARY = "no_estacionario"  # b ≥ 1
    NON_POSITIVE_B = "b_no_positivo"  # b ≤ 0
    INSUFFICIENT = "datos_insuficientes"


class OUFit(BaseModel):
    status: OUStatus
    valid: bool
    # Regresión AR(1)
    a: float | None = None
    b: float | None = None
    a_se: float | None = None
    b_se: float | None = None
    s2: float | None = None
    nobs: int = 0
    dt: float = 1.0
    # Parámetros OU (None si no es válido)
    lam: float | None = None
    mu: float | None = None
    sigma: float | None = None
    sigma_eq: float | None = None
    half_life: float | None = None
    mu_se: float | None = None
    time_unit: str = TIME_UNIT_DEFAULT
    message: str = ""
    warnings: list[str] = []


class OUForecast(BaseModel):
    horizon: int
    mean: float
    std: float
    lower: float
    upper: float
    level: float


class OUSimulationSummary(BaseModel):
    lam: float
    mu: float
    sigma: float
    x0: float
    horizon: int
    n_paths: int
    seed: int
    sigma_eq: float
    half_life_theoretical: float
    half_life_empirical: float | None
    time_grid: list[float]
    paths: list[list[float]]  # n_paths × (horizon+1)
    mean_empirical: list[float]
    mean_theoretical: list[float]
    band_lower_theoretical: list[float]  # media condicional ± 2·sd condicional
    band_upper_theoretical: list[float]
    stationary_lower: list[float]  # μ ± 2σ_eq (constantes)
    stationary_upper: list[float]
    crossing_times: list[float | None]  # por trayectoria; None si no cruzó
    n_not_crossed: int
    notes: list[str]
    label: str = "SINTÉTICO"


def simulate_ou_exact(
    lam: float,
    mu: float,
    sigma: float,
    x0: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Simula trayectorias con la transición EXACTA del OU (no Euler).

    X_{t+Δt} = μ + (X_t − μ)·e^{−λΔt} + σ_eq·sqrt(1 − e^{−2λΔt})·Z,  Z ~ N(0,1)

    Devuelve un arreglo (n_paths, n_steps + 1) cuya primera columna es x0.
    Las normales se extraen de ``np.random.default_rng(seed)`` con forma
    (n_paths, n_steps), en ese orden: misma semilla → mismas trayectorias.
    """
    if lam <= 0 or sigma <= 0:
        raise ConfigError(
            "λ y σ deben ser positivos para simular un OU.",
            code="PARAMETROS_OU_INVALIDOS",
            details={"lam": lam, "sigma": sigma},
        )
    if n_steps < 1 or n_paths < 1:
        raise ConfigError("Se requiere al menos 1 paso y 1 trayectoria.", code="PARAMETROS_OU_INVALIDOS")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    decay = math.exp(-lam * dt)
    sigma_eq = sigma / math.sqrt(2.0 * lam)
    step_sd = sigma_eq * math.sqrt(1.0 - decay * decay)
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = x0
    for t in range(n_steps):
        out[:, t + 1] = mu + (out[:, t] - mu) * decay + step_sd * z[:, t]
    return out


def fit_ou(spread: pd.Series | np.ndarray, dt: float = 1.0, time_unit: str = TIME_UNIT_DEFAULT) -> OUFit:
    raise NotImplementedError


def conditional_mean(x_t: float, fit: OUFit, h: float) -> float:
    raise NotImplementedError


def conditional_variance(fit: OUFit, h: float) -> float:
    raise NotImplementedError


def forecast(fit: OUFit, x_t: float, horizons: tuple[int, ...] = (1, 5, 20), level: float = 0.95) -> list[OUForecast]:
    raise NotImplementedError


def zscore(spread: pd.Series, fit: OUFit) -> pd.Series:
    raise NotImplementedError


def rolling_zscore(spread: pd.Series, window: int = 63) -> pd.Series:
    raise NotImplementedError


def simulate_paths_summary(
    lam: float,
    mu: float,
    sigma: float,
    x0: float,
    horizon: int,
    n_paths: int,
    seed: int,
    dt: float = 1.0,
) -> OUSimulationSummary:
    raise NotImplementedError
