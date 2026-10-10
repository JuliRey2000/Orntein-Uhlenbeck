"""Diagnóstico del ajuste OU (6.2): residuos AR(1), ACF, Ljung–Box y estabilidad por ventanas.

CONTRATO:
- ``ar1_residuals(spread, fit)``: ε_t = X_{t+1} − a − b·X_t con a, b del ajuste
  (también si el OU no es válido, mientras a y b existan).
- ``residual_diagnostics``: ACF de 1..nlags (statsmodels ``acf``, fft=False),
  banda ±1,96/√n, Ljung–Box en ``lb_lags`` (``acorr_ljungbox``). Si el p-value
  de Ljung–Box en el rezago mayor < 0,05 → ``autocorrelation_detected`` y
  ``message`` en español explicando que el AR(1) no captura toda la dinámica.
- ``rolling_ou_stability(spread, window, step)``: ``fit_ou`` sobre cada
  ventana del spread de FORMACIÓN (α, β fijos: es el spread que se opera).
  ``cv_lambda`` = desviación/media de λ sobre las ventanas válidas (None si
  hay < 2 válidas). ``message`` en español. ``unit_root_alpha``,
  ``estimated_residual`` y ``scale`` se pasan tal cual a ``fit_ou`` (decisión
  O7): el pipeline usa ``estimated_residual=True`` porque el spread es el
  residuo de la regresión de cointegración de formación.

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
from mrlab.model.ou import TIME_UNIT_DEFAULT, UNIT_ROOT_ALPHA_DEFAULT, OUFit, fit_ou
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


def _fmt(x: float, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles (convención de la interfaz)."""
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _fmt_p(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else _fmt(p)


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
    estimated_residual: bool = False,
    scale: float | None = None,
) -> OUStability:
    """``fit_ou`` por ventanas del spread de formación con α y β fijos (decisión E8).

    Cada ventana usa la misma regla de validez que el ajuste completo (decisión O7).
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
            estimated_residual=estimated_residual,
            scale=scale,
        )
        windows.append(
            OUWindow(
                start=spread.index[start].date(),
                end=spread.index[end - 1].date(),
                nobs=end - start,
                valid=fit.valid,
                lam=fit.lam,
                mu=fit.mu,
                sigma=fit.sigma,
                sigma_eq=fit.sigma_eq,
                half_life=fit.half_life,
            )
        )

    if not windows:
        return OUStability(
            window=window,
            step=step,
            windows=[],
            n_invalid=0,
            cv_lambda=None,
            message=(
                f"El spread de formación tiene {len(spread)} observaciones, menos que una ventana "
                f"de {window}: no se puede evaluar la estabilidad del OU entre ventanas."
            ),
        )

    valid = [w for w in windows if w.valid]
    n_invalid = len(windows) - len(valid)
    parts = [
        f"Se ajustó el OU en {len(windows)} ventanas de {window} observaciones (paso {step}) "
        "sobre el spread de formación, con α y β fijos."
    ]
    if n_invalid:
        parts.append(
            f"En {n_invalid} de ellas el OU no fue válido (b fuera de (0, 1), b que no se "
            "distingue de 1 según Dickey–Fuller o datos insuficientes)."
        )

    cv_lambda: float | None = None
    if len(valid) >= 2:
        lams = np.array([w.lam for w in valid], dtype=float)
        cv_lambda = float(lams.std(ddof=1) / lams.mean())
        half_lives = [w.half_life for w in valid]
        parts.append(
            f"Coeficiente de variación de λ entre las {len(valid)} ventanas válidas: "
            f"{_fmt(cv_lambda, 2)} (desviación estándar / media; cuanto mayor, menos estable es la "
            f"velocidad de reversión). La vida media osciló entre {_fmt(min(half_lives), 1)} y "
            f"{_fmt(max(half_lives), 1)} {_plural(time_unit)}."
        )
    else:
        parts.append("Hay menos de 2 ventanas con OU válido: no se puede medir la variación de λ.")
    parts.append(
        "Con pocas observaciones por ventana λ se estima con mucho ruido, así que es esperable "
        "algo de variación aunque la dinámica no cambie."
    )

    return OUStability(
        window=window,
        step=step,
        windows=windows,
        n_invalid=n_invalid,
        cv_lambda=cv_lambda,
        message=" ".join(parts),
    )
