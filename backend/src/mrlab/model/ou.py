"""Proceso Ornstein–Uhlenbeck: estimación exacta discreta, pronóstico y simulación.

    dX_t = λ(μ − X_t) dt + σ dW_t        (λ ≡ θ en otras notaciones)

Estimación exacta con observaciones equiespaciadas (Δt = 1 observación):

    X_{t+Δt} = a + b·X_t + ε_t,  ε_t ~ N(0, s²)
    si 0 < b < 1:  λ = −ln(b)/Δt,  μ = a/(1 − b),  σ = sqrt(2λ·s²/(1 − b²)),
                   σ_eq = σ/sqrt(2λ),  vida media = ln(2)/λ

CONTRATO:
- ``fit_ou``: AR(1) por MCO con constante sobre el spread de formación;
  ``s2 = SRC/(n−2)``. El AR(1) SOLO estima a, b, λ, μ, σ, σ_eq y la vida media.
  El ajuste solo es VÁLIDO si 0 < b < 1 Y una prueba de Dickey–Fuller
  AUMENTADA (ADF, rezagos de las diferencias elegidos por AIC hasta ``maxlag``,
  como en Engle–Granger) RECHAZA la raíz unitaria al nivel ``unit_root_alpha``
  (decisión O7). Si b ≥ 1, o si 0 < b < 1 pero la prueba no rechaza
  (``near_unit_root``) → status NON_STATIONARY; si b ≤ 0 → NON_POSITIVE_B;
  menos de 30 puntos, spread constante, determinista o nulo frente a ``scale``
  → INSUFFICIENT. En todos esos casos ``valid=False`` y
  lam/mu/sigma/sigma_eq/half_life/mu_se = None, con ``message`` en español ("el
  spread no muestra reversión estacionaria compatible con un OU"). Nunca
  producir números engañosos. Con ``require_unit_root_rejection=False`` (lo usan
  las ventanas de estabilidad) la prueba se calcula e informa igual, pero la
  validez solo exige 0 < b < 1: sirve para obtener λ, μ, σ y la vida media.
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

Notas de implementación:
- ``fit_ou`` usa solo pares consecutivos (X_t, X_{t+1}) con ambos valores
  finitos: un NaN nunca se rellena ni crea una transición ficticia. ``nobs`` es
  el número de pares de la regresión.
- Un spread constante o sin componente aleatorio → INSUFFICIENT (no hay
  variación que modelar).
- Paseo aleatorio y sesgo de Dickey–Fuller (decisión O7): con un paseo
  aleatorio, el b̂ de MCO queda por debajo de 1 en ≈ 95 % de las muestras, así
  que el criterio b ≥ 1 casi nunca lo detecta. Por eso la validez exige además
  que la prueba ADF rechace la raíz unitaria. Se usa la versión AUMENTADA (no la
  de cero rezagos): si los incrementos del spread tienen autocorrelación (p. ej.
  ruido de microestructura sobre un paseo aleatorio), la prueba sin rezagos
  rechaza de más (30 % de OU «válidos» con ruido de 0,5 %).
  * Serie observada (``estimated_residual=False``, por defecto):
    ``adfuller(x, regression="c", autolag="AIC", maxlag)`` con los p-values de
    MacKinnon para N = 1 (los mismos que devuelve ``adfuller``).
  * Residuo de una regresión de cointegración estimada con los mismos datos
    (``estimated_residual=True``): EXACTAMENTE el paso interno de
    ``statsmodels.coint``: ``adfuller(resid, regression="n", autolag="aic",
    maxlag)`` con los p-values de MacKinnon para N = 2 con constante (la MCO
    hace que el residuo parezca más estacionario). Sobre el spread de formación
    la validez del OU coincide así con la prueba de Engle–Granger SIN AJUSTAR de
    esa dirección (mismo estadístico, rezagos y p-value).
  La decisión es p-value ≤ α, y ``df_crit`` es el valor crítico de MacKinnon que
  corresponde exactamente a esos p-values (``mackinnonp`` invertido), de modo
  que «estadístico ≤ valor crítico» y «p-value ≤ α» nunca se contradicen. La
  tabla de muestra finita (``mackinnoncrit``) difiere en menos de 0,01 con
  n ≥ 252, pero haría discrepar la validez del OU y Engle–Granger en una franja
  estrecha. Si no se rechaza: NON_STATIONARY, ``near_unit_root=True`` y se
  conservan a, b y los resultados de la prueba (df_*).
- La prueba ADF necesita observaciones consecutivas: con NaN en el spread se
  aplica al tramo continuo sin NaN más largo (nunca se unen tramos) y, si ese
  tramo tiene menos de ``MIN_OBS_OU`` puntos, el ajuste es INSUFFICIENT. Si la
  serie es corta, ``maxlag`` se reduce al máximo que admite ``adfuller``.
- ``scale`` (opcional): escala absoluta de las series de origen (p. ej. la
  desviación de f(y) en formación). Si la desviación del spread es menor que
  1e-10·scale, el spread es ruido de redondeo (dos piernas idénticas o
  proporcionales) → INSUFFICIENT.
- λ, la vida media y los horizontes de pronóstico están en ``time_unit`` (con
  Δt = 1, una observación = un día de negociación, decisión O4).
"""

from __future__ import annotations

import math
import warnings
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import SingularMatrixWarning
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller

from mrlab.domain.errors import ConfigError, ModelError

TIME_UNIT_DEFAULT = "día de negociación"
MAX_SIM_PATHS = 500
MAX_SIM_HORIZON = 2520  # 10 años de días de negociación: acota el tamaño de la respuesta
HALF_LIFE_MAX_WARN = 126.0
HALF_LIFE_MIN_WARN = 1.0
MIN_OBS_OU = 30
SIM_BAND_K = 2.0  # bandas de la simulación: ± 2 desviaciones estándar
UNIT_ROOT_ALPHA_DEFAULT = 0.05
#: Rezagos máximos de la prueba ADF (los mismos que Engle–Granger: ``AnalysisConfig.maxlag``).
UNIT_ROOT_MAXLAG_DEFAULT = 12
#: Un spread con desviación < SCALE_REL_TOL·escala de las series es ruido de redondeo.
SCALE_REL_TOL = 1e-10

NOT_OU_MESSAGE = "El spread no muestra reversión estacionaria compatible con un OU"

#: Avisos para la interfaz (sección 6.2): qué representan bandas e intervalos.
PARAMETER_UNCERTAINTY_NOTE = (
    "Las bandas y los intervalos usan λ, μ y σ estimados como si fueran exactos: no incluyen la "
    "incertidumbre de los parámetros (no incluida en esta versión), así que son más estrechos de "
    "lo que deberían."
)
BANDS_VS_INTERVALS_NOTE = (
    "Las bandas μ ± k·σ_eq describen dónde se mueve el spread a largo plazo (distribución "
    "estacionaria). Los intervalos de pronóstico describen dónde podría estar dentro de h días "
    "partiendo del valor actual (distribución condicional): son más estrechos a horizontes cortos "
    "y se acercan a las bandas estacionarias cuando h crece."
)
MU_REFERENCE_NOTE = (
    "μ es la referencia estadística del modelo estimada en formación, no un valor justo ni un "
    "objetivo de precio."
)


class OUStatus(StrEnum):
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
    # Prueba de raíz unitaria ADF (decisión O7, ver notas del módulo); None si no se calculó
    df_stat: float | None = None  # estadístico de Dickey–Fuller aumentado (ADF)
    df_crit: float | None = None  # valor crítico de MacKinnon (con constante) al nivel df_alpha
    df_pvalue: float | None = None  # p-value de MacKinnon (rechaza si ≤ df_alpha)
    df_lags: int | None = None  # rezagos de las diferencias elegidos por AIC
    df_alpha: float = UNIT_ROOT_ALPHA_DEFAULT
    df_estimated_residual: bool = False  # True → residuo estimado: N = 2, como Engle–Granger
    near_unit_root: bool = False  # 0 < b̂ < 1 pero b̂ no se distingue de 1 (implica no válido)


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
    crossing_time_median: float | None = None  # None si la mitad o más no cruzó


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
        raise ConfigError(
            "Se requiere al menos 1 paso y 1 trayectoria.", code="PARAMETROS_OU_INVALIDOS"
        )
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


# --------------------------------------------------------------------------- formato


def _fmt(x: float, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles (convención de la interfaz)."""
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _fmt_g(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _fmt_sci(x: float) -> str:
    """Número muy pequeño o muy grande en notación científica con coma decimal: «4,77e-16»."""
    return f"{x:.2e}".replace(".", ",")


def _fmt_pct(fraction: float) -> str:
    """0.05 → «5 %», 0.025 → «2,5 %»."""
    return f"{_fmt_g(round(fraction * 100, 6))} %"


def _plural(time_unit: str) -> str:
    """'día de negociación' → 'días de negociación' (otras unidades se dejan igual)."""
    return "días" + time_unit[3:] if time_unit.startswith("día") else time_unit


# --------------------------------------------------------------------------- raíz unitaria


def _check_unit_root_alpha(alpha: float, name: str = "alpha") -> None:
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ConfigError(
            "El nivel de la prueba de raíz unitaria debe estar entre 0 y 1 (por ejemplo 0,05).",
            code="NIVEL_INVALIDO",
            details={name: alpha},
        )


def unit_root_pvalue(stat: float, *, estimated_residual: bool = False) -> float:
    """p-value de MacKinnon (con constante) del estadístico ADF.

    ``estimated_residual=False`` → N = 1 (el que devuelve ``adfuller`` con ``regression="c"``);
    ``True`` → N = 2 (el que devuelve ``coint`` para el residuo de la regresión estimada).
    """
    return float(mackinnonp(stat, regression="c", N=2 if estimated_residual else 1))


def unit_root_critical_value(alpha: float, *, estimated_residual: bool = False) -> float:
    """Valor crítico (cola izquierda) que corresponde EXACTAMENTE a ``unit_root_pvalue``.

    Se obtiene invirtiendo ``mackinnonp``: «estadístico ≤ valor crítico» equivale a
    «p-value ≤ α», así que la comparación que se muestra nunca contradice la decisión.
    """
    _check_unit_root_alpha(alpha)
    n_vars = 2 if estimated_residual else 1
    return float(brentq(lambda s: mackinnonp(s, regression="c", N=n_vars) - alpha, -50.0, 50.0))


def _check_maxlag(maxlag: int) -> None:
    if isinstance(maxlag, bool) or not isinstance(maxlag, int | np.integer) or maxlag < 0:
        raise ConfigError(
            "Los rezagos máximos de la prueba de raíz unitaria deben ser un entero ≥ 0.",
            code="REZAGOS_INVALIDOS",
            details={"maxlag": maxlag},
        )


class UnitRootResult(BaseModel):
    """Prueba ADF de una serie (decisión O7)."""

    stat: float
    pvalue: float
    crit: float
    lags: int
    maxlag: int
    nobs: int  # observaciones de la regresión ADF con los rezagos elegidos
    alpha: float
    estimated_residual: bool
    rejected: bool  # p-value ≤ α: se rechaza la raíz unitaria


def _longest_finite_run(x: np.ndarray) -> np.ndarray:
    """Tramo continuo más largo sin NaN ni infinitos (la prueba ADF nunca une tramos)."""
    finite = np.isfinite(x)
    if finite.all():
        return x
    best_start = best_len = run_start = 0
    run_len = 0
    for i, ok in enumerate(finite):
        if ok:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    return x[best_start : best_start + best_len]


def adf_unit_root_test(
    series: pd.Series | np.ndarray,
    *,
    alpha: float = UNIT_ROOT_ALPHA_DEFAULT,
    estimated_residual: bool = False,
    maxlag: int = UNIT_ROOT_MAXLAG_DEFAULT,
) -> UnitRootResult | None:
    """Dickey–Fuller aumentado con rezagos elegidos por AIC (decisión O7).

    - Serie observada: ``adfuller(x, regression="c", autolag="AIC", maxlag)``, p-value de
      MacKinnon N = 1 (el mismo de ``adfuller``).
    - Residuo estimado: ``adfuller(resid, regression="n", autolag="aic", maxlag)`` y p-value de
      MacKinnon N = 2 con constante: exactamente el cálculo de ``statsmodels.coint``.

    Con NaN se usa el tramo continuo sin NaN más largo. Devuelve None si la prueba no se puede
    calcular (tramo de menos de ``MIN_OBS_OU`` puntos, serie constante, regresión de rango
    incompleto o estadístico no finito).
    """
    _check_unit_root_alpha(alpha)
    _check_maxlag(maxlag)
    x = _longest_finite_run(np.asarray(series, dtype=float))
    if x.size < MIN_OBS_OU or x.max() == x.min():
        return None
    regression = "n" if estimated_residual else "c"
    n_trend = 0 if estimated_residual else 1
    usable_maxlag = min(int(maxlag), x.size // 2 - n_trend - 1)  # condición de adfuller
    with warnings.catch_warnings():
        # Una regresión ADF de rango incompleto (p. ej. ruido de redondeo de ~1e-16) no da un
        # estadístico fiable: la prueba se considera no calculable.
        warnings.simplefilter("error", SingularMatrixWarning)
        try:
            stat, _, lags, nobs, *_ = adfuller(
                x,
                maxlag=usable_maxlag,
                autolag="aic",
                regression=regression,
                result_object=False,
            )
        except SingularMatrixWarning:
            return None
    stat = float(stat)
    if not math.isfinite(stat):
        return None
    pvalue = unit_root_pvalue(stat, estimated_residual=estimated_residual)
    return UnitRootResult(
        stat=stat,
        pvalue=pvalue,
        crit=unit_root_critical_value(alpha, estimated_residual=estimated_residual),
        lags=int(lags),
        maxlag=usable_maxlag,
        nobs=int(nobs),
        alpha=alpha,
        estimated_residual=estimated_residual,
        rejected=pvalue <= alpha,
    )


def _fmt_p(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else _fmt(p)


def _df_comparison(test: UnitRootResult) -> str:
    """«estadístico −3,04 frente al valor crítico −3,34 al 5 %, p-value 0,0731; …»."""
    lags = (
        f"{test.lags} rezago{'s' if test.lags != 1 else ''} de las diferencias "
        f"elegido{'s' if test.lags != 1 else ''} por AIC, máximo {test.maxlag}"
    )
    which = (
        "valores críticos de Engle–Granger, porque el spread es el residuo de una regresión "
        "estimada con estos mismos datos: es el mismo cálculo que la prueba de Engle–Granger "
        "sin corrección por comparaciones múltiples"
        if test.estimated_residual
        else "valores críticos de MacKinnon para una serie observada, con constante"
    )
    return (
        f"estadístico {_fmt(test.stat, 2)} frente al valor crítico {_fmt(test.crit, 2)} al "
        f"{_fmt_pct(test.alpha)}, p-value {_fmt_p(test.pvalue)}; {lags}; {which}"
    )


# --------------------------------------------------------------------------- estimación


def fit_ou(
    spread: pd.Series | np.ndarray,
    dt: float = 1.0,
    time_unit: str = TIME_UNIT_DEFAULT,
    *,
    half_life_max: float = HALF_LIFE_MAX_WARN,
    half_life_min: float = HALF_LIFE_MIN_WARN,
    unit_root_alpha: float = UNIT_ROOT_ALPHA_DEFAULT,
    estimated_residual: bool = False,
    maxlag: int = UNIT_ROOT_MAXLAG_DEFAULT,
    scale: float | None = None,
    require_unit_root_rejection: bool = True,
) -> OUFit:
    """Ajusta el OU por su discretización exacta: AR(1) por MCO con constante.

    El ajuste es válido solo si 0 < b < 1 y la prueba ADF (rezagos por AIC hasta ``maxlag``)
    rechaza la raíz unitaria al nivel ``unit_root_alpha`` (decisión O7).
    ``estimated_residual=True``: el spread es el residuo de una regresión de cointegración
    estimada con estos datos → el mismo cálculo que Engle–Granger (N = 2).
    ``scale``: escala absoluta de las series de origen; un spread con desviación menor que
    ``1e-10·scale`` es ruido de redondeo → INSUFFICIENT.
    ``require_unit_root_rejection=False``: la prueba se calcula e informa, pero la validez solo
    exige 0 < b < 1 (para estimar λ, μ, σ y la vida media en ventanas cortas, donde la prueba
    tiene poca potencia).
    """
    if not (math.isfinite(dt) and dt > 0):
        raise ConfigError("Δt debe ser un número positivo.", code="DT_INVALIDO", details={"dt": dt})
    _check_unit_root_alpha(unit_root_alpha, "unit_root_alpha")
    _check_maxlag(maxlag)
    if scale is not None and not (math.isfinite(scale) and scale > 0):
        raise ConfigError(
            "La escala de referencia del spread debe ser un número positivo.",
            code="ESCALA_INVALIDA",
            details={"scale": scale},
        )
    x = np.asarray(spread, dtype=float)
    if x.ndim != 1:
        raise ConfigError("El spread debe ser una serie unidimensional.", code="SPREAD_INVALIDO")

    prev, nxt = x[:-1], x[1:]
    usable = np.isfinite(prev) & np.isfinite(nxt)
    prev, nxt = prev[usable], nxt[usable]
    n = int(prev.size)
    base = {
        "nobs": n,
        "dt": dt,
        "time_unit": time_unit,
        "df_alpha": unit_root_alpha,
        "df_estimated_residual": estimated_residual,
    }

    if n + 1 < MIN_OBS_OU:
        n_points = n + 1 if n else int(np.isfinite(x).sum())
        return OUFit(
            status=OUStatus.INSUFFICIENT,
            valid=False,
            message=(
                f"Para estimar el OU se necesitan al menos {MIN_OBS_OU} observaciones consecutivas "
                f"del spread; hay {n_points}."
            ),
            **base,
        )

    m0, m1 = prev.mean(), nxt.mean()
    d0 = prev - m0
    sxx = float(d0 @ d0)
    if sxx <= 0.0:
        return OUFit(
            status=OUStatus.INSUFFICIENT,
            valid=False,
            message="El spread es constante: no hay variación que modelar con un OU.",
            **base,
        )
    if scale is not None:
        spread_sd = float(np.std(x[np.isfinite(x)]))
        if spread_sd < SCALE_REL_TOL * scale:
            return OUFit(
                status=OUStatus.INSUFFICIENT,
                valid=False,
                message=(
                    "La variación del spread es prácticamente nula frente al nivel de las series "
                    f"(desviación estándar {_fmt_sci(spread_sd)} frente a una escala de "
                    f"{_fmt_sci(scale)}): lo que queda es ruido de redondeo numérico, típico "
                    "cuando las dos piernas son la misma serie o una es un múltiplo exacto de la "
                    "otra. Un OU no es un modelo adecuado para estos datos."
                ),
                **base,
            )
    b = float(d0 @ (nxt - m1)) / sxx
    a = float(m1 - b * m0)
    resid = nxt - a - b * prev
    s2 = float(resid @ resid) / (n - 2)
    if s2 <= 1e-12 * sxx / n:
        return OUFit(
            status=OUStatus.INSUFFICIENT,
            valid=False,
            a=a,
            b=b,
            s2=s2,
            message=(
                "El spread evoluciona de forma prácticamente determinista (sin componente "
                "aleatorio): un OU no es un modelo adecuado para estos datos."
            ),
            **base,
        )
    b_se = math.sqrt(s2 / sxx)
    a_se = math.sqrt(s2 * (1.0 / n + m0 * m0 / sxx))
    test = adf_unit_root_test(
        x, alpha=unit_root_alpha, estimated_residual=estimated_residual, maxlag=maxlag
    )
    regression: dict = {"a": a, "b": b, "a_se": a_se, "b_se": b_se, "s2": s2}
    if test is not None:
        regression.update(
            df_stat=test.stat, df_crit=test.crit, df_pvalue=test.pvalue, df_lags=test.lags
        )
    elif require_unit_root_rejection:
        return OUFit(
            status=OUStatus.INSUFFICIENT,
            valid=False,
            message=(
                "No se pudo calcular la prueba de raíz unitaria (Dickey–Fuller aumentada): "
                f"necesita un tramo de al menos {MIN_OBS_OU} observaciones consecutivas sin "
                "faltantes y con variación suficiente para estimar su regresión. Sin esa prueba "
                "no se puede afirmar que el spread revierta."
            ),
            **regression,
            **base,
        )

    if b >= 1.0:
        return OUFit(
            status=OUStatus.NON_STATIONARY,
            valid=False,
            message=(
                f"{NOT_OU_MESSAGE}: el coeficiente AR(1) estimado es b = {_fmt(b)} ≥ 1, así que "
                "las desviaciones no tienden a corregirse (el spread se comporta como un paseo "
                "aleatorio o se aleja de su nivel). No se calculan λ, μ, σ, vida media ni bandas."
            ),
            **regression,
            **base,
        )
    if b <= 0.0:
        return OUFit(
            status=OUStatus.NON_POSITIVE_B,
            valid=False,
            message=(
                f"{NOT_OU_MESSAGE}: el coeficiente AR(1) estimado es b = {_fmt(b)} ≤ 0, es decir, "
                "el spread tiende a cambiar de signo de una observación a la siguiente (oscila). "
                "Un OU observado a intervalos regulares siempre tiene 0 < b < 1 (b = e^(−λΔt)), "
                "así que este comportamiento no corresponde a un OU. No se calculan λ, μ, σ, "
                "vida media ni bandas."
            ),
            **regression,
            **base,
        )
    if require_unit_root_rejection and test is not None and not test.rejected:
        return OUFit(
            status=OUStatus.NON_STATIONARY,
            valid=False,
            near_unit_root=True,
            message=(
                f"{NOT_OU_MESSAGE}: el coeficiente AR(1) estimado es b = {_fmt(b)}, menor que 1, "
                "pero no se distingue estadísticamente de 1: la prueba de Dickey–Fuller aumentada "
                f"(ADF) no rechaza la raíz unitaria ({_df_comparison(test)}). Con estos datos no "
                "se puede distinguir el spread de un paseo aleatorio: un b algo menor que 1 es lo "
                "esperable incluso en un paseo aleatorio (sesgo de Dickey–Fuller), así que no hay "
                "evidencia de que las desviaciones se corrijan. No se calculan λ, μ, σ, vida "
                "media ni bandas."
            ),
            **regression,
            **base,
        )

    lam = -math.log(b) / dt
    mu = a / (1.0 - b)
    sigma = math.sqrt(2.0 * lam * s2 / (1.0 - b * b))
    sigma_eq = sigma / math.sqrt(2.0 * lam)
    half_life = math.log(2.0) / lam
    mu_se = sigma_eq * math.sqrt((1.0 + b) / ((1.0 - b) * n))
    units = _plural(time_unit)

    warnings: list[str] = []
    if half_life > half_life_max:
        warnings.append(
            f"Vida media de {_fmt(half_life, 1)} {units} (> {_fmt_g(half_life_max)}): demasiado "
            "lenta para operar con este horizonte."
        )
    if half_life < half_life_min:
        warnings.append(
            f"Vida media de {_fmt(half_life, 2)} {units} (< {_fmt_g(half_life_min)}): con datos "
            "diarios no es operable ni fiable."
        )
    if require_unit_root_rejection:
        assert test is not None
        message = (
            f"El AR(1) del spread tiene 0 < b < 1 (b = {_fmt(b)}) y la prueba de Dickey–Fuller "
            f"aumentada (ADF) rechaza la raíz unitaria ({_df_comparison(test)}): en esta "
            "muestra las desviaciones respecto de μ tienden a corregirse, con una vida media "
            f"estimada de {_fmt(half_life, 1)} {units}. Que el ajuste sea válido no garantiza "
            "que el spread "
            "siga revirtiendo fuera de la muestra: la evidencia de cointegración la da la prueba "
            "de Engle–Granger."
        )
    else:
        if test is None:
            verdict = "la prueba ADF no se pudo calcular"
        else:
            verb = "rechaza" if test.rejected else "no rechaza"
            verdict = f"la prueba ADF {verb} la raíz unitaria ({_df_comparison(test)})"
        message = (
            f"El AR(1) del spread tiene 0 < b < 1 (b = {_fmt(b)}): λ, μ, σ y la vida media "
            f"({_fmt(half_life, 1)} {units}) se calculan sin exigir que la prueba de raíz unitaria "
            f"la rechace; {verdict}."
        )

    return OUFit(
        status=OUStatus.VALID,
        valid=True,
        lam=lam,
        mu=mu,
        sigma=sigma,
        sigma_eq=sigma_eq,
        half_life=half_life,
        mu_se=mu_se,
        message=message,
        warnings=warnings,
        **regression,
        **base,
    )


# --------------------------------------------------------------------------- pronóstico y z


def _require_valid(fit: OUFit, what: str) -> tuple[float, float, float]:
    """(λ, μ, σ_eq) de un ajuste válido; ModelError en otro caso."""
    if not fit.valid or fit.lam is None or fit.mu is None or fit.sigma_eq is None:
        raise ModelError(
            f"No se puede calcular {what}: el ajuste OU no es válido. {fit.message}".strip(),
            details={"status": fit.status.value},
        )
    return fit.lam, fit.mu, fit.sigma_eq


def _check_horizon(h: float) -> None:
    if not (math.isfinite(h) and h >= 0):
        raise ConfigError(
            "El horizonte debe ser un número no negativo.",
            code="HORIZONTE_INVALIDO",
            details={"h": h},
        )


def conditional_mean(x_t: float, fit: OUFit, h: float) -> float:
    """E[X_{t+h} | X_t] = μ + (X_t − μ)·e^{−λh}."""
    lam, mu, _ = _require_valid(fit, "la media condicional")
    _check_horizon(h)
    return mu + (x_t - mu) * math.exp(-lam * h)


def conditional_variance(fit: OUFit, h: float) -> float:
    """Var[X_{t+h} | X_t] = σ_eq²·(1 − e^{−2λh})."""
    lam, _, sigma_eq = _require_valid(fit, "la varianza condicional")
    _check_horizon(h)
    return sigma_eq * sigma_eq * -math.expm1(-2.0 * lam * h)


def forecast(
    fit: OUFit, x_t: float, horizons: tuple[int, ...] = (1, 5, 20), level: float = 0.95
) -> list[OUForecast]:
    """Pronóstico condicional con intervalos normales simétricos (sin incertidumbre de parámetros)."""
    _require_valid(fit, "el pronóstico")
    if not 0.0 < level < 1.0:
        raise ConfigError(
            "El nivel del intervalo debe estar entre 0 y 1 (por ejemplo 0,95).",
            code="NIVEL_INVALIDO",
            details={"level": level},
        )
    if any(int(h) != h or h < 1 for h in horizons):
        raise ConfigError(
            "Los horizontes de pronóstico deben ser enteros mayores o iguales a 1.",
            code="HORIZONTE_INVALIDO",
            details={"horizons": list(horizons)},
        )
    q = float(norm.ppf(0.5 + level / 2.0))
    out: list[OUForecast] = []
    for h in horizons:
        mean = conditional_mean(x_t, fit, h)
        std = math.sqrt(conditional_variance(fit, h))
        out.append(
            OUForecast(
                horizon=int(h),
                mean=mean,
                std=std,
                lower=mean - q * std,
                upper=mean + q * std,
                level=level,
            )
        )
    return out


def zscore(spread: pd.Series, fit: OUFit) -> pd.Series:
    """z_t = (X_t − μ)/σ_eq con μ y σ_eq FIJOS del ajuste de formación (μ se usa tal cual)."""
    _, mu, sigma_eq = _require_valid(fit, "el z-score")
    z = (spread.astype(float) - mu) / sigma_eq
    z.name = "zscore"
    return z


def rolling_zscore(spread: pd.Series, window: int = 63) -> pd.Series:
    """z móvil (solo comparación pedagógica): media y desviación de las ``window`` últimas
    observaciones hasta t inclusive. NaN mientras la ventana no esté completa."""
    if window < 2:
        raise ConfigError(
            "La ventana del z móvil debe tener al menos 2 observaciones.",
            code="VENTANA_INVALIDA",
            details={"window": window},
        )
    values = spread.astype(float)
    rolling = values.rolling(window=window, min_periods=window)
    z = (values - rolling.mean()) / rolling.std(ddof=1)
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = "zscore_movil"
    return z


# --------------------------------------------------------------------------- simulación


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
    """Simulación SINTÉTICA para la pestaña educativa (transición exacta, no Euler)."""
    for name, value in (("λ", lam), ("μ", mu), ("σ", sigma), ("X₀", x0), ("Δt", dt)):
        if not math.isfinite(value):
            raise ConfigError(f"{name} debe ser un número finito.", code="PARAMETROS_OU_INVALIDOS")
    if dt <= 0:
        raise ConfigError("Δt debe ser positivo.", code="DT_INVALIDO", details={"dt": dt})
    if n_paths > MAX_SIM_PATHS:
        raise ConfigError(
            f"Se pueden simular como máximo {MAX_SIM_PATHS} trayectorias; se pidieron {n_paths}.",
            code="DEMASIADAS_TRAYECTORIAS",
            details={"n_paths": n_paths, "max": MAX_SIM_PATHS},
        )
    if n_paths < 1:
        raise ConfigError("Se requiere al menos 1 trayectoria.", code="PARAMETROS_OU_INVALIDOS")
    if not 1 <= horizon <= MAX_SIM_HORIZON:
        raise ConfigError(
            f"El horizonte debe estar entre 1 y {MAX_SIM_HORIZON} pasos; se pidieron {horizon}.",
            code="HORIZONTE_INVALIDO",
            details={"horizon": horizon, "max": MAX_SIM_HORIZON},
        )
    if x0 == mu:
        raise ConfigError(
            "X₀ debe ser distinto de μ: el tiempo hasta cruzar μ y la vida media empírica se miden "
            "a partir de una desviación inicial distinta de cero.",
            code="X0_IGUAL_A_MU",
            details={"x0": x0, "mu": mu},
        )

    paths = simulate_ou_exact(lam, mu, sigma, x0, horizon, n_paths, seed, dt)
    t = np.arange(horizon + 1) * dt
    sigma_eq = sigma / math.sqrt(2.0 * lam)
    half_life = math.log(2.0) / lam
    dev0 = x0 - mu

    mean_emp = paths.mean(axis=0)
    mean_th = mu + dev0 * np.exp(-lam * t)
    sd_cond = sigma_eq * np.sqrt(-np.expm1(-2.0 * lam * t))

    # Cruce de μ: primer k ≥ 1 en que la desviación cambia de signo o se anula.
    crossed = (paths[:, 1:] - mu) * math.copysign(1.0, dev0) <= 0.0
    has_crossed = crossed.any(axis=1)
    first_k = crossed.argmax(axis=1) + 1
    crossing = np.where(has_crossed, first_k * dt, np.inf)
    crossing_times = [float(c) if np.isfinite(c) else None for c in crossing]
    n_not_crossed = int((~has_crossed).sum())
    median_crossing = float(np.median(crossing))

    # Vida media empírica: la media de las trayectorias llega a la mitad de la desviación inicial.
    ratio = (mean_emp - mu) / dev0
    ratio[0] = 1.0
    below = np.flatnonzero(ratio <= 0.5)
    half_life_emp: float | None = None
    if below.size:
        k = int(below[0])
        r_prev, r_k = ratio[k - 1], ratio[k]
        half_life_emp = float(t[k - 1] + (r_prev - 0.5) / (r_prev - r_k) * dt)

    units = _plural(TIME_UNIT_DEFAULT)
    notes = [
        "Datos SINTÉTICOS: trayectorias simuladas con la transición exacta del OU (no Euler); "
        "no son precios históricos.",
        f"Vida media teórica ln(2)/λ = {_fmt(half_life, 1)} {units}: es el tiempo en que la "
        "desviación ESPERADA respecto de μ se reduce a la mitad, no el tiempo típico hasta cruzar μ.",
        "El cruce de μ solo se detecta en los instantes simulados (tiempo discreto): si una "
        "trayectoria cruza μ y vuelve entre dos observaciones no se ve, así que los tiempos de cruce "
        "están sesgados hacia arriba.",
        f"Bandas teóricas: media condicional ± {_fmt_g(SIM_BAND_K)} desviaciones estándar "
        f"condicionales (≈ 95 %); bandas estacionarias: μ ± {_fmt_g(SIM_BAND_K)}·σ_eq.",
    ]
    if np.isfinite(median_crossing):
        notes.append(
            f"Mediana del tiempo hasta cruzar μ: {_fmt(median_crossing, 1)} {units}, frente a una "
            f"vida media teórica de {_fmt(half_life, 1)}."
        )
    if n_not_crossed:
        notes.append(
            f"{n_not_crossed} de {n_paths} trayectorias no cruzaron μ dentro del horizonte."
        )
    if half_life_emp is None:
        notes.append(
            "La media de las trayectorias no llegó a la mitad de la desviación inicial dentro del "
            "horizonte: amplía el horizonte para estimar la vida media empírica."
        )

    return OUSimulationSummary(
        lam=lam,
        mu=mu,
        sigma=sigma,
        x0=x0,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
        sigma_eq=sigma_eq,
        half_life_theoretical=half_life,
        half_life_empirical=half_life_emp,
        time_grid=t.tolist(),
        paths=paths.tolist(),
        mean_empirical=mean_emp.tolist(),
        mean_theoretical=mean_th.tolist(),
        band_lower_theoretical=(mean_th - SIM_BAND_K * sd_cond).tolist(),
        band_upper_theoretical=(mean_th + SIM_BAND_K * sd_cond).tolist(),
        stationary_lower=[mu - SIM_BAND_K * sigma_eq] * (horizon + 1),
        stationary_upper=[mu + SIM_BAND_K * sigma_eq] * (horizon + 1),
        crossing_times=crossing_times,
        n_not_crossed=n_not_crossed,
        notes=notes,
        crossing_time_median=median_crossing if np.isfinite(median_crossing) else None,
    )
