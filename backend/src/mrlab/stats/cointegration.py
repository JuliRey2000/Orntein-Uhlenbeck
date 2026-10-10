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
- ``eg_decision(results, ...)`` (función pura para el pipeline): ajusta los
  p-values por FDR, elige la dirección y decide si hay evidencia.
  * Familia por defecto: las 2 direcciones del par (m = 2).
  * Universo: ``family_pvalues`` = [p(A_ON_B), p(B_ON_A), *p de las demás
    pruebas]. Las DOS PRIMERAS posiciones deben ser las del par, en ese orden
    (se verifica contra ``results``; si no coinciden → ``ConfigError``).
  * ``evidence`` = p ajustado de la dirección elegida ≤ ``alpha_level``. Los
    p ajustados no dependen de ``fdr_q`` (q solo fija qué rechazaría BH); la
    decisión usa ``alpha_level``, que es el umbral que ve el usuario.
  * ``message``: dirección usada, por qué (automática o fijada) y la decisión.

Notas de implementación:
- Si las dos series son casi colineales, ``coint`` devuelve estadístico −∞ y
  p = 0 sin evaluar los rezagos: se informa ``lags = 0`` y un ``warning`` en
  español (p = 0 en ese caso NO es evidencia fiable).
- Las ventanas usan fechas del índice (``DatetimeIndex``); ``start`` y ``end``
  son la primera y la última fecha incluidas en la ventana.
"""

from __future__ import annotations

import warnings
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.tsatools import add_trend

from mrlab.domain.errors import ConfigError, DataError
from mrlab.stats.multitest import FDR_METHOD_LABELS, adjust_pvalues
from mrlab.stats.windows import rolling_windows


class Direction(StrEnum):
    A_ON_B = "a_sobre_b"  # A dependiente: A = α + β·B + residuo
    B_ON_A = "b_sobre_a"  # B dependiente: B = α + β·A + residuo


#: Orden canónico de las direcciones (también el de ``family_pvalues``).
DIRECTION_ORDER = (Direction.A_ON_B, Direction.B_ON_A)

COLLINEAR_WARNING = (
    "Las dos series son prácticamente idénticas salvo escala (R² ≈ 1): la prueba de Engle–Granger "
    "no es fiable en este caso y statsmodels asigna p = 0. Revisa si se eligió dos veces el mismo "
    "activo o series duplicadas; ese p-value no debe leerse como evidencia."
)

STABILITY_POWER_NOTE = (
    "Con pocas observaciones por ventana la prueba de Engle–Granger tiene poca potencia, así que "
    "incluso una relación real puede no detectarse en algunas ventanas."
)


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
    warning: str | None = None  # p. ej. series casi colineales (prueba no fiable)


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


class EGDecision(BaseModel):
    direction: Direction
    direction_was_auto: bool
    p_raw: dict[Direction, float]
    p_adjusted: dict[Direction, float]
    evidence: bool
    message: str
    alpha_level: float = 0.05
    fdr_method: str = "bh"
    family_size: int = 2  # m: número de pruebas de la familia corregida


def _fmt(x: float, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles (convención de la interfaz)."""
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _fmt_p(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else _fmt(p)


def _fmt_pct(frac: float) -> str:
    return f"{100 * frac:.0f} %"


def _pair_arrays(y: pd.Series, x: pd.Series, maxlag: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) != len(x):
        raise DataError(
            f"Las dos series deben tener el mismo número de observaciones ({len(y)} frente a {len(x)})."
        )
    if isinstance(y, pd.Series) and isinstance(x, pd.Series) and not y.index.equals(x.index):
        raise DataError("Las dos series no están alineadas en las mismas fechas.")
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    if not (np.isfinite(yv).all() and np.isfinite(xv).all()):
        raise DataError(
            "Las series contienen valores faltantes o no finitos; la prueba requiere datos limpios."
        )
    # Condición de statsmodels para el ADF sin constante sobre los residuos: maxlag ≤ nobs//2 − 1.
    min_obs = 2 * (maxlag + 1)
    if yv.size < min_obs:
        raise DataError(
            f"Engle–Granger con hasta {maxlag} rezagos necesita al menos {min_obs} observaciones; "
            f"hay {yv.size}."
        )
    if yv.max() == yv.min() or xv.max() == xv.min():
        raise DataError("Una de las series es constante: la prueba de cointegración no aplica.")
    return yv, xv


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
    """Engle–Granger aumentado de ``y`` (dependiente) sobre ``x``, tal como lo calcula ``coint``."""
    yv, xv = _pair_arrays(y, x, maxlag)

    with warnings.catch_warnings():
        # El caso colineal se detecta por el estadístico −∞ y se explica en español.
        warnings.simplefilter("ignore", CollinearityWarning)
        statistic, pvalue, crit = coint(yv, xv, trend=trend, maxlag=maxlag, autolag="aic")

    # Mismo diseño que coint: [x, constante] (prepend=False), para obtener residuos idénticos.
    exog = xv[:, None] if trend == "n" else add_trend(xv[:, None], trend=trend, prepend=False)
    ols = OLS(yv, exog).fit()
    beta = float(ols.params[0])
    alpha = 0.0 if trend == "n" else float(ols.params[1])

    warning = None
    if np.isfinite(statistic):
        _, _, lags, *_ = adfuller(
            ols.resid, maxlag=maxlag, autolag="aic", regression="n", result_object=False
        )
    else:
        lags, warning = 0, COLLINEAR_WARNING

    return EGResult(
        direction=direction,
        dependent=dependent,
        independent=independent,
        statistic=float(statistic),
        pvalue=float(pvalue),
        critical_values={k: float(v) for k, v in zip(("1%", "5%", "10%"), crit, strict=True)},
        nobs=int(yv.size),
        trend=trend,
        lags=int(lags),
        maxlag=maxlag,
        alpha_ols=alpha,
        beta_ols=beta,
        warning=warning,
    )


def engle_granger_both(
    fa: pd.Series, fb: pd.Series, symbol_a: str, symbol_b: str, maxlag: int = 12
) -> dict[Direction, EGResult]:
    """Engle–Granger en las dos direcciones: A sobre B (A dependiente) y B sobre A."""
    return {
        Direction.A_ON_B: engle_granger(
            fa,
            fb,
            direction=Direction.A_ON_B,
            dependent=symbol_a,
            independent=symbol_b,
            maxlag=maxlag,
        ),
        Direction.B_ON_A: engle_granger(
            fb,
            fa,
            direction=Direction.B_ON_A,
            dependent=symbol_b,
            independent=symbol_a,
            maxlag=maxlag,
        ),
    }


def choose_direction(
    results: dict[Direction, EGResult], override: Direction | None = None
) -> Direction:
    """Dirección fijada por el usuario o, si no, la de menor p-value (empate → A sobre B)."""
    if not results:
        raise ConfigError("No hay resultados de Engle–Granger para elegir una dirección.")
    if override is not None:
        if override not in results:
            raise ConfigError("La dirección fijada no tiene resultado de Engle–Granger.")
        return override
    return min(results, key=lambda d: (results[d].pvalue, DIRECTION_ORDER.index(d)))


def eg_decision(
    results: dict[Direction, EGResult],
    *,
    alpha_level: float = 0.05,
    fdr_q: float = 0.05,
    fdr_method: str = "bh",
    override: Direction | None = None,
    family_pvalues: list[float] | None = None,
) -> EGDecision:
    """Ajuste FDR + elección de dirección + decisión de evidencia (decisiones E3 y E4).

    ``family_pvalues`` (opcional, para universos): todos los p-values sin ajustar
    de la familia, con las dos direcciones de ESTE par en las posiciones 0
    (A sobre B) y 1 (B sobre A). Sin él, la familia son las 2 direcciones del par.
    """
    if set(results) != set(DIRECTION_ORDER):
        raise ConfigError(
            "Se necesitan los resultados de Engle–Granger en las dos direcciones del par."
        )
    if not 0.0 < alpha_level < 1.0:
        raise ConfigError(
            f"El nivel de significancia debe estar entre 0 y 1 (se recibió {alpha_level})."
        )
    if not 0.0 < fdr_q < 1.0:
        raise ConfigError(
            f"La tasa de falsos descubrimientos q debe estar entre 0 y 1 (se recibió {fdr_q})."
        )
    if fdr_method not in FDR_METHOD_LABELS:
        raise ConfigError(f"Método de corrección desconocido: {fdr_method!r}. Usa 'bh' o 'by'.")

    p_raw = {d: results[d].pvalue for d in DIRECTION_ORDER}
    own = [p_raw[d] for d in DIRECTION_ORDER]
    if family_pvalues is None:
        family = own
    else:
        family = [float(p) for p in family_pvalues]
        if len(family) < 2 or not np.allclose(family[:2], own, rtol=1e-9, atol=1e-12):
            raise ConfigError(
                "family_pvalues debe empezar por los p-values de este par en el orden "
                "[A sobre B, B sobre A] (se recibió "
                f"{family[:2]}, se esperaba {own})."
            )
    _, adjusted = adjust_pvalues(family, method=fdr_method, q=fdr_q)
    p_adjusted = {d: float(adjusted[i]) for i, d in enumerate(DIRECTION_ORDER)}

    direction = choose_direction(results, override)
    other = DIRECTION_ORDER[1 - DIRECTION_ORDER.index(direction)]
    chosen = results[direction]
    # Con series casi colineales coint da estadístico −∞ y p = 0: no es evidencia fiable.
    reliable = bool(np.isfinite(chosen.statistic))
    below = p_adjusted[direction] <= alpha_level
    evidence = reliable and below

    label = (
        f"{chosen.dependent} sobre {chosen.independent} "
        f"({chosen.dependent} como variable dependiente)"
    )
    if override is not None:
        why = (
            f"fijada por el usuario (p-value {_fmt_p(p_raw[direction])}; la dirección contraria "
            f"tiene {_fmt_p(p_raw[other])})"
        )
    elif p_raw[direction] == p_raw[other]:
        why = (
            "elegida automáticamente: ambas direcciones tienen el mismo p-value "
            f"({_fmt_p(p_raw[direction])})"
        )
    else:
        why = (
            f"elegida automáticamente por tener el menor p-value de Engle–Granger "
            f"({_fmt_p(p_raw[direction])} frente a {_fmt_p(p_raw[other])} en la dirección contraria)"
        )
    family_desc = (
        "las 2 direcciones del par"
        if family_pvalues is None
        else f"{len(family)} pruebas del universo (pares × direcciones)"
    )
    if not reliable:
        verdict = f", pero no se considera evidencia. {chosen.warning or COLLINEAR_WARNING}"
    elif evidence:
        verdict = ": hay evidencia de cointegración en la ventana evaluada."
    else:
        verdict = ": la evidencia de cointegración es insuficiente al nivel elegido."
    message = (
        f"Dirección usada: {label}, {why}. Con la corrección de {FDR_METHOD_LABELS[fdr_method]} "
        f"sobre {family_desc}, el p-value ajustado ({_fmt_p(p_adjusted[direction])}) "
        f"{'no supera' if below else 'supera'} el nivel {_fmt(alpha_level, 2)}{verdict}"
    )

    return EGDecision(
        direction=direction,
        direction_was_auto=override is None,
        p_raw=p_raw,
        p_adjusted=p_adjusted,
        evidence=evidence,
        message=message,
        alpha_level=alpha_level,
        fdr_method=fdr_method,
        family_size=len(family),
    )


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
    """Engle–Granger por ventanas móviles dentro de formación, con la dirección fija (decisión E5)."""
    if not isinstance(y.index, pd.DatetimeIndex):
        raise DataError("Las series de estabilidad deben estar indexadas por fecha.")
    if not y.index.equals(x.index):
        raise DataError("Las dos series no están alineadas en las mismas fechas.")

    windows: list[EGWindow] = []
    for start, end in rolling_windows(len(y), window, step):
        eg = engle_granger(
            y.iloc[start:end],
            x.iloc[start:end],
            direction=direction,
            dependent=dependent,
            independent=independent,
            maxlag=maxlag,
        )
        windows.append(
            EGWindow(
                start=y.index[start].date(),
                end=y.index[end - 1].date(),
                nobs=end - start,
                statistic=eg.statistic,
                pvalue=eg.pvalue,
                alpha=eg.alpha_ols,
                beta=eg.beta_ols,
                passed=eg.pvalue <= alpha_level,
            )
        )

    if not windows:
        return EGStability(
            direction=direction,
            window=window,
            step=step,
            alpha_level=alpha_level,
            max_fail_frac=max_fail_frac,
            windows=[],
            frac_fail=None,
            beta_sign_change=False,
            unstable=False,
            reasons=[
                f"La muestra de formación tiene {len(y)} observaciones, menos que una ventana de "
                f"{window}: no se puede evaluar la estabilidad entre ventanas."
            ],
        )

    n_windows = len(windows)
    n_fail = sum(not w.passed for w in windows)
    frac_fail = n_fail / n_windows
    betas = np.array([w.beta for w in windows])
    beta_sign_change = bool((betas > 0).any() and (betas < 0).any())
    too_many_failures = frac_fail > max_fail_frac

    failures = (
        f"En {n_fail} de {n_windows} ventanas de {window} observaciones ({_fmt_pct(frac_fail)}) el "
        f"p-value de Engle–Granger superó el nivel {_fmt(alpha_level, 2)}; el máximo tolerado es "
        f"{_fmt_pct(max_fail_frac)}."
    )
    if too_many_failures:
        failures += f" {STABILITY_POWER_NOTE}"
    beta_range = f"mínimo {_fmt(betas.min(), 3)}, máximo {_fmt(betas.max(), 3)}"
    if beta_sign_change:
        betas_msg = (
            f"β cambió de signo entre ventanas ({beta_range}): la relación entre los activos se "
            "invirtió dentro de la muestra de formación."
        )
    else:
        betas_msg = f"β mantuvo el mismo signo en todas las ventanas ({beta_range})."

    return EGStability(
        direction=direction,
        window=window,
        step=step,
        alpha_level=alpha_level,
        max_fail_frac=max_fail_frac,
        windows=windows,
        frac_fail=frac_fail,
        beta_sign_change=beta_sign_change,
        unstable=too_many_failures or beta_sign_change,
        reasons=[failures, betas_msg],
    )
