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
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel

IntegrationClass = Literal["compatible_I1", "posiblemente_estacionaria", "ambiguo"]

INTEGRATION_LABELS = {
    "compatible_I1": "Compatible con I(1)",
    "posiblemente_estacionaria": "Posiblemente estacionaria",
    "ambiguo": "Resultado ambiguo",
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


def adf_test(x: pd.Series, series_kind: str = "niveles", maxlag: int = 12, alpha: float = 0.05) -> UnitRootTest:
    raise NotImplementedError


def kpss_test(x: pd.Series, series_kind: str = "niveles", alpha: float = 0.05) -> UnitRootTest:
    raise NotImplementedError


def integration_tests(x: pd.Series, symbol: str, alpha: float = 0.05, maxlag: int = 12) -> IntegrationResult:
    raise NotImplementedError
