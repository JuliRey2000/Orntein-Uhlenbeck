"""Pruebas de integración de cada serie: ADF + KPSS en niveles y primeras diferencias (4.1).

CONTRATO (a implementar):
- ADF: ``statsmodels.tsa.stattools.adfuller(x, maxlag=12, regression="c",
  autolag="AIC")`` (pasar ``result_object=False`` si la versión lo admite, para
  evitar FutureWarning y mantener la tupla). H0: raíz unitaria.
- KPSS: ``kpss(x, regression="c", nlags="auto")``. H0: estacionariedad. Su
  p-value solo se interpola entre 0,01 y 0,10: si queda en el borde, reportar
  ``pvalue_bound`` = "menor_que" (0,01) o "mayor_que" (0,10) y suprimir el
  InterpolationWarning.
- Interpretación combinada (alpha = nivel elegido, default 0,05):
  * niveles: ADF NO rechaza y KPSS rechaza, Y diferencias: ADF rechaza y KPSS
    NO rechaza → "compatible_I1";
  * niveles: ADF rechaza y KPSS NO rechaza → "posiblemente_estacionaria";
  * cualquier otro caso → "ambiguo".
  ``explanation``: frase en español que diga qué dijo cada prueba.
- La serie de entrada ya viene transformada (log-precios o precios) y limpia.

Notas de implementación:
- Se rechaza H0 cuando p ≤ alpha. En KPSS con p acotado solo se rechaza si el
  límite permite confirmarlo: "menor_que" (p < 0,01) rechaza si alpha ≥ 0,01;
  "mayor_que" (p > 0,10) nunca rechaza (con alpha > 0,10 no se puede confirmar).
- Entradas con NaN/infinitos, constantes o demasiado cortas para ``maxlag`` →
  ``DataError`` con mensaje en español (en vez del error genérico de statsmodels).
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from mrlab.domain.errors import DataError

IntegrationClass = Literal["compatible_I1", "posiblemente_estacionaria", "ambiguo"]

INTEGRATION_LABELS = {
    "compatible_I1": "Compatible con I(1)",
    "posiblemente_estacionaria": "Posiblemente estacionaria",
    "ambiguo": "Resultado ambiguo",
}

#: Rango de la tabla de KPSS (Kwiatkowski et al., 1992): fuera de él el p-value es un límite.
KPSS_P_MIN = 0.01
KPSS_P_MAX = 0.10

_INTUITION = {
    "compatible_I1": (
        "La serie deambula sin volver a un nivel fijo (como un paseo aleatorio), pero sus cambios "
        "diarios sí son estables: es el tipo de serie para el que tiene sentido buscar cointegración."
    ),
    "posiblemente_estacionaria": (
        "La serie parece volver por sí sola a un nivel estable. Engle–Granger supone series I(1), así "
        "que los resultados de cointegración con esta serie deben tomarse con cautela."
    ),
    "ambiguo": (
        "Las pruebas no coinciden, así que no se puede afirmar que la serie sea I(1); los resultados "
        "de cointegración con esta serie deben tomarse con cautela."
    ),
}


class UnitRootTest(BaseModel):
    test: Literal["ADF", "KPSS"]
    series: Literal["niveles", "diferencias"]
    statistic: float
    pvalue: float
    pvalue_bound: Literal["exacto", "menor_que", "mayor_que"] = "exacto"
    lags: int
    nobs: int
    regression: str = "c"
    critical_values: dict[str, float]
    rejects_null: bool


class IntegrationResult(BaseModel):
    symbol: str
    alpha: float
    level_adf: UnitRootTest
    level_kpss: UnitRootTest
    diff_adf: UnitRootTest
    diff_kpss: UnitRootTest
    classification: IntegrationClass
    explanation: str


def _fmt(x: float, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles (convención de la interfaz)."""
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def format_pvalue(test: UnitRootTest) -> str:
    """p-value tal como se muestra: «p = 0,0312», «p < 0,01», «p > 0,10» o «p < 0,0001»."""
    if test.pvalue_bound == "menor_que":
        return f"p < {_fmt(KPSS_P_MIN, 2)}"
    if test.pvalue_bound == "mayor_que":
        return f"p > {_fmt(KPSS_P_MAX, 2)}"
    if test.pvalue < 1e-4:
        return "p < 0,0001"
    return f"p = {_fmt(test.pvalue)}"


def _as_clean_array(x: pd.Series | np.ndarray, what: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise DataError(f"La serie ({what}) debe ser unidimensional.")
    if not np.isfinite(arr).all():
        raise DataError(
            f"La serie ({what}) contiene valores faltantes o no finitos; "
            "las pruebas de raíz unitaria requieren una serie limpia."
        )
    if arr.size and arr.max() == arr.min():
        raise DataError(f"La serie ({what}) es constante: las pruebas de raíz unitaria no aplican.")
    return arr


def adf_test(
    x: pd.Series | np.ndarray, series_kind: str = "niveles", maxlag: int = 12, alpha: float = 0.05
) -> UnitRootTest:
    """ADF con constante y rezagos por AIC (máximo ``maxlag``). H0: raíz unitaria."""
    arr = _as_clean_array(x, series_kind)
    # Condición de statsmodels con constante: maxlag ≤ nobs//2 − 2.
    min_obs = 2 * (maxlag + 2)
    if arr.size < min_obs:
        raise DataError(
            f"La prueba ADF con hasta {maxlag} rezagos necesita al menos {min_obs} observaciones "
            f"({series_kind}); hay {arr.size}."
        )
    stat, pvalue, lags, nobs, crit, _ = adfuller(
        arr, maxlag=maxlag, regression="c", autolag="AIC", result_object=False
    )
    return UnitRootTest(
        test="ADF",
        series=series_kind,
        statistic=float(stat),
        pvalue=float(pvalue),
        lags=int(lags),
        nobs=int(nobs),
        critical_values={k: float(v) for k, v in crit.items()},
        rejects_null=bool(pvalue <= alpha),
    )


def kpss_test(
    x: pd.Series | np.ndarray, series_kind: str = "niveles", alpha: float = 0.05
) -> UnitRootTest:
    """KPSS con constante y rezagos automáticos (Hobijn et al.). H0: estacionariedad."""
    arr = _as_clean_array(x, series_kind)
    with warnings.catch_warnings():
        # El borde de la tabla se informa en ``pvalue_bound``; el aviso en inglés no llega al usuario.
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, pvalue, lags, crit = kpss(arr, regression="c", nlags="auto", result_object=False)

    pvalue = float(pvalue)
    if pvalue <= KPSS_P_MIN:
        bound, rejects = "menor_que", alpha >= KPSS_P_MIN
    elif pvalue >= KPSS_P_MAX:
        bound, rejects = "mayor_que", False
    else:
        bound, rejects = "exacto", pvalue <= alpha
    return UnitRootTest(
        test="KPSS",
        series=series_kind,
        statistic=float(stat),
        pvalue=pvalue,
        pvalue_bound=bound,
        lags=int(lags),
        nobs=int(arr.size),
        critical_values={k: float(v) for k, v in crit.items()},
        rejects_null=bool(rejects),
    )


def _classify(
    level_adf: UnitRootTest,
    level_kpss: UnitRootTest,
    diff_adf: UnitRootTest,
    diff_kpss: UnitRootTest,
) -> IntegrationClass:
    levels_nonstationary = not level_adf.rejects_null and level_kpss.rejects_null
    diffs_stationary = diff_adf.rejects_null and not diff_kpss.rejects_null
    if levels_nonstationary and diffs_stationary:
        return "compatible_I1"
    if level_adf.rejects_null and not level_kpss.rejects_null:
        return "posiblemente_estacionaria"
    return "ambiguo"


def _describe(test: UnitRootTest) -> str:
    if test.test == "ADF":
        verdict = "rechaza la raíz unitaria" if test.rejects_null else "no rechaza la raíz unitaria"
    else:
        verdict = (
            "rechaza la estacionariedad" if test.rejects_null else "no rechaza la estacionariedad"
        )
    return f"{test.test} {verdict} ({format_pvalue(test)})"


def integration_tests(
    x: pd.Series, symbol: str, alpha: float = 0.05, maxlag: int = 12
) -> IntegrationResult:
    """ADF + KPSS en niveles y primeras diferencias, con interpretación combinada (decisión E2)."""
    levels = _as_clean_array(x, "niveles")
    diffs = np.diff(levels)

    level_adf = adf_test(levels, "niveles", maxlag=maxlag, alpha=alpha)
    level_kpss = kpss_test(levels, "niveles", alpha=alpha)
    diff_adf = adf_test(diffs, "diferencias", maxlag=maxlag, alpha=alpha)
    diff_kpss = kpss_test(diffs, "diferencias", alpha=alpha)
    classification = _classify(level_adf, level_kpss, diff_adf, diff_kpss)

    explanation = (
        f"{symbol}: {INTEGRATION_LABELS[classification]} (nivel de significancia {_fmt(alpha, 2)}). "
        f"En niveles, {_describe(level_adf)} y {_describe(level_kpss)}. "
        f"En primeras diferencias, {_describe(diff_adf)} y {_describe(diff_kpss)}. "
        f"{_INTUITION[classification]}"
    )
    return IntegrationResult(
        symbol=symbol,
        alpha=alpha,
        level_adf=level_adf,
        level_kpss=level_kpss,
        diff_adf=diff_adf,
        diff_kpss=diff_kpss,
        classification=classification,
        explanation=explanation,
    )
