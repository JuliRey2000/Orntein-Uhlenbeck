"""Diagnóstico del ajuste OU (6.2): residuos AR(1), ACF, Ljung–Box y estabilidad por ventanas.

CONTRATO:
- ``ar1_residuals(spread, fit)``: ε_t = X_{t+1} − a − b·X_t con a, b del ajuste
  (también si el OU no es válido, mientras a y b existan).
- ``residual_diagnostics``: ACF de 1..nlags (statsmodels ``acf``, fft=False),
  banda ±1,96/√n, Ljung–Box en ``lb_lags`` (``acorr_ljungbox``). Si el p-value
  de Ljung–Box en el rezago mayor < 0,05 → ``autocorrelation_detected`` y
  ``message`` en español explicando que el AR(1) no captura toda la dinámica.
- ``rolling_ou_stability(spread, window, step)``: AR(1)/OU sobre cada ventana
  del spread de FORMACIÓN (α, β fijos: es el spread que se opera). En las
  ventanas NO se exige el rechazo de la raíz unitaria para obtener parámetros
  (``fit_ou(..., require_unit_root_rejection=False)``): λ, μ, σ y la vida media
  se calculan en toda ventana con 0 < b < 1 (``estimable``). ``cv_lambda`` =
  desviación/media de λ sobre las ventanas estimables (None si hay < 2). La
  prueba ADF de cada ventana se informa APARTE (``unit_root_rejected``,
  ``df_*``) con un resumen (``n_unit_root_rejected``): valores críticos de una
  serie observada (N = 1), porque α y β se estimaron con toda la formación y no
  se re-estiman en la ventana. ``message`` en español; explica que con ~252
  observaciones la prueba tiene poca potencia y que no rechazar en una ventana
  no prueba inestabilidad (decisión O7: la validez del OU la decide el ajuste de
  toda la formación).

Notas de implementación:
- Los residuos se indexan con la fecha de X_{t+1} (el valor que se pronosticó)
  y se descartan los pares con algún NaN, igual que en ``fit_ou``.
- Ljung–Box se calcula con ``model_df=1`` (un coeficiente AR estimado): el
  estadístico sobre residuos de un AR(1) sigue una χ² con m − 1 grados de
  libertad. Por eso los rezagos de ``lb_lags`` deben ser ≥ 2.
- ``cv_lambda`` usa la desviación estándar muestral (ddof = 1).
- Las ventanas son las de ``mrlab.stats.windows.rolling_windows`` (las mismas
  de la estabilidad de cointegración, sección 4.4).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from mrlab.domain.errors import ConfigError, DataError, ModelError
from mrlab.model.ou import (
    TIME_UNIT_DEFAULT,
    UNIT_ROOT_ALPHA_DEFAULT,
    UNIT_ROOT_MAXLAG_DEFAULT,
    OUFit,
    fit_ou,
)
from mrlab.stats.windows import rolling_windows

LB_ALPHA = 0.05
AR1_MODEL_DF = 1
ACF_Z = 1.96


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
    estimable: bool  # 0 < b < 1: λ, μ, σ y vida media calculados (sin exigir la prueba ADF)
    b: float | None = None
    lam: float | None = None
    mu: float | None = None
    sigma: float | None = None
    sigma_eq: float | None = None
    half_life: float | None = None
    # Prueba ADF de la ventana (N = 1; solo informativa). None si no se pudo calcular.
    unit_root_rejected: bool | None = None
    df_stat: float | None = None
    df_pvalue: float | None = None
    df_lags: int | None = None


class OUStability(BaseModel):
    window: int
    step: int
    windows: list[OUWindow]
    n_invalid: int  # ventanas sin parámetros: b fuera de (0, 1) o datos insuficientes
    cv_lambda: float | None
    message: str
    unit_root_alpha: float = UNIT_ROOT_ALPHA_DEFAULT
    n_unit_root_tested: int = 0  # ventanas en las que se pudo calcular la prueba ADF
    n_unit_root_rejected: int = 0  # ventanas en las que la prueba ADF rechaza la raíz unitaria


def _fmt(x: float, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles (convención de la interfaz)."""
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _fmt_p(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else _fmt(p)


def _fmt_level(alpha: float) -> str:
    """0.05 → «5 %», 0.025 → «2,5 %»."""
    return f"{round(alpha * 100, 6):g}".replace(".", ",") + " %"


def _plural(time_unit: str) -> str:
    return "días" + time_unit[3:] if time_unit.startswith("día") else time_unit


def ar1_residuals(spread: pd.Series, fit: OUFit) -> pd.Series:
    """ε_t = X_{t+1} − a − b·X_t con los coeficientes del ajuste (válido o no)."""
    if fit.a is None or fit.b is None:
        raise ModelError(
            "No hay coeficientes AR(1) para calcular residuos: el ajuste no llegó a estimarse "
            f"({fit.message})."
        )
    values = spread.astype(float)
    resid = (values - fit.a - fit.b * values.shift(1)).dropna()
    resid.name = "residuo_ar1"
    return resid


def residual_diagnostics(
    spread: pd.Series, fit: OUFit, nlags: int = 20, lb_lags: tuple[int, ...] = (10, 20)
) -> ResidualDiagnostics:
    """ACF y Ljung–Box de los residuos del AR(1): ¿queda dinámica sin explicar?"""
    if nlags < 1 or not lb_lags or min(lb_lags) <= AR1_MODEL_DF:
        raise ConfigError(
            f"Se requiere al menos 1 rezago de ACF y rezagos de Ljung–Box mayores que "
            f"{AR1_MODEL_DF}.",
            code="REZAGOS_INVALIDOS",
            details={"nlags": nlags, "lb_lags": list(lb_lags)},
        )
    resid = ar1_residuals(spread, fit)
    values = resid.to_numpy()
    n = values.size
    max_lag = max(nlags, *lb_lags)
    if n <= max_lag + 1:
        raise DataError(
            f"Hay {n} residuos: se necesitan más de {max_lag + 1} para calcular la ACF y "
            f"Ljung–Box hasta el rezago {max_lag}."
        )

    acf_values = acf(values, nlags=nlags, fft=False)[1:]
    lb = acorr_ljungbox(values, lags=sorted(set(lb_lags)), model_df=AR1_MODEL_DF, return_df=True)
    ljung_box = [
        LjungBoxResult(
            lag=int(lag), statistic=float(row["lb_stat"]), pvalue=float(row["lb_pvalue"])
        )
        for lag, row in lb.iterrows()
    ]
    last = ljung_box[-1]  # rezago mayor
    detected = last.pvalue < LB_ALPHA
    if detected:
        message = (
            f"Los residuos del AR(1) muestran autocorrelación (Ljung–Box con {last.lag} rezagos: "
            f"p = {_fmt_p(last.pvalue)} < {_fmt(LB_ALPHA, 2)}): el modelo AR(1)/OU no captura "
            "toda la dinámica del spread. Las bandas, los intervalos y la vida media pueden ser "
            "poco fiables."
        )
    else:
        message = (
            f"No se detecta autocorrelación en los residuos del AR(1) (Ljung–Box con {last.lag} "
            f"rezagos: p = {_fmt_p(last.pvalue)} ≥ {_fmt(LB_ALPHA, 2)}). Es compatible con que el "
            "AR(1)/OU capture la dinámica principal del spread, pero no prueba que el modelo sea "
            "correcto."
        )

    return ResidualDiagnostics(
        nlags=nlags,
        acf=[float(v) for v in acf_values],
        conf_band=ACF_Z / float(np.sqrt(n)),
        ljung_box=ljung_box,
        residual_mean=float(values.mean()),
        residual_std=float(values.std(ddof=1)),
        autocorrelation_detected=detected,
        message=message,
    )


def rolling_ou_stability(
    spread: pd.Series,
    window: int = 252,
    step: int = 63,
    dt: float = 1.0,
    *,
    time_unit: str = TIME_UNIT_DEFAULT,
    unit_root_alpha: float = UNIT_ROOT_ALPHA_DEFAULT,
    maxlag: int = UNIT_ROOT_MAXLAG_DEFAULT,
    scale: float | None = None,
) -> OUStability:
    """OU por ventanas del spread de formación con α y β fijos (decisiones E8 y O7).

    Los parámetros se estiman en toda ventana con 0 < b < 1; la prueba ADF (N = 1) se
    informa aparte y no condiciona λ, CV(λ) ni el rango de la vida media (si lo hiciera,
    solo quedarían las ventanas con λ alto y la vida media saldría sesgada a la baja).
    """
    if not isinstance(spread.index, pd.DatetimeIndex):
        raise DataError("El spread debe estar indexado por fecha para evaluar su estabilidad.")
    if window < 2 or step < 1:
        raise ConfigError(
            "La ventana debe tener al menos 2 observaciones y el paso al menos 1.",
            code="VENTANA_INVALIDA",
            details={"window": window, "step": step},
        )

    windows: list[OUWindow] = []
    for start, end in rolling_windows(len(spread), window, step):
        fit = fit_ou(
            spread.iloc[start:end],
            dt=dt,
            time_unit=time_unit,
            unit_root_alpha=unit_root_alpha,
            estimated_residual=False,  # α y β no se re-estiman en la ventana
            maxlag=maxlag,
            scale=scale,
            require_unit_root_rejection=False,
        )
        windows.append(
            OUWindow(
                start=spread.index[start].date(),
                end=spread.index[end - 1].date(),
                nobs=end - start,
                estimable=fit.valid,
                b=fit.b,
                lam=fit.lam,
                mu=fit.mu,
                sigma=fit.sigma,
                sigma_eq=fit.sigma_eq,
                half_life=fit.half_life,
                unit_root_rejected=(
                    None if fit.df_pvalue is None else fit.df_pvalue <= unit_root_alpha
                ),
                df_stat=fit.df_stat,
                df_pvalue=fit.df_pvalue,
                df_lags=fit.df_lags,
            )
        )

    if not windows:
        return OUStability(
            window=window,
            step=step,
            windows=[],
            n_invalid=0,
            cv_lambda=None,
            unit_root_alpha=unit_root_alpha,
            message=(
                f"El spread de formación tiene {len(spread)} observaciones, menos que una ventana "
                f"de {window}: no se puede evaluar la estabilidad del OU entre ventanas."
            ),
        )

    estimable = [w for w in windows if w.estimable]
    n_invalid = len(windows) - len(estimable)
    tested = [w for w in windows if w.unit_root_rejected is not None]
    n_rejected = sum(bool(w.unit_root_rejected) for w in tested)
    parts = [
        f"Se ajustó el AR(1)/OU en {len(windows)} ventanas de {window} observaciones (paso "
        f"{step}) sobre el spread de formación, con α y β fijos. En cada ventana λ, μ, σ y la "
        "vida media se calculan si 0 < b < 1, sin exigir la prueba de raíz unitaria."
    ]
    if n_invalid:
        parts.append(
            f"En {n_invalid} de ellas no se pudieron estimar (b fuera de (0, 1) o datos "
            "insuficientes)."
        )

    cv_lambda: float | None = None
    if len(estimable) >= 2:
        lams = np.array([w.lam for w in estimable], dtype=float)
        cv_lambda = float(lams.std(ddof=1) / lams.mean())
        half_lives = np.array([w.half_life for w in estimable], dtype=float)
        parts.append(
            f"Coeficiente de variación de λ entre las {len(estimable)} ventanas con parámetros: "
            f"{_fmt(cv_lambda, 2)} (desviación estándar / media; cuanto mayor, menos estable es la "
            f"velocidad de reversión). La vida media osciló entre {_fmt(half_lives.min(), 1)} y "
            f"{_fmt(half_lives.max(), 1)} {_plural(time_unit)} (mediana "
            f"{_fmt(float(np.median(half_lives)), 1)})."
        )
    else:
        parts.append("Hay menos de 2 ventanas con 0 < b < 1: no se puede medir la variación de λ.")
    if tested:
        parts.append(
            f"Prueba de Dickey–Fuller aumentada en cada ventana (valores críticos de una serie "
            f"observada, porque α y β no se re-estiman en la ventana): rechaza la raíz unitaria "
            f"al {_fmt_level(unit_root_alpha)} en {n_rejected} de {len(tested)} ventanas. Con "
            f"{window} observaciones esta prueba tiene poca potencia: es habitual que no rechace "
            "en una ventana aunque el spread revierta, así que no rechazar en una ventana no "
            "prueba que la dinámica sea inestable. La validez del OU la decide el ajuste de toda "
            "la formación."
        )
    parts.append(
        "Con pocas observaciones por ventana λ se estima con mucho ruido (y con sesgo hacia "
        "arriba, así que la vida media tiende a salir algo más corta), por lo que es esperable "
        "algo de variación aunque la dinámica no cambie."
    )

    return OUStability(
        window=window,
        step=step,
        windows=windows,
        n_invalid=n_invalid,
        cv_lambda=cv_lambda,
        message=" ".join(parts),
        unit_root_alpha=unit_root_alpha,
        n_unit_root_tested=len(tested),
        n_unit_root_rejected=n_rejected,
    )
