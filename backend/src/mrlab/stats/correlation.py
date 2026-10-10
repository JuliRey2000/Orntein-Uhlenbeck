"""Explorador de relaciones: correlaciones de rendimientos logarítmicos (sección 3).

CONTRATO (a implementar):
- Los rendimientos de cada PAR se calculan sobre la intersección de fechas de
  sus precios (no sobre el calendario propio de cada serie): así, en un par
  cripto + bolsa el rendimiento cripto de viernes a lunes incluye el fin de
  semana, coherente con la política de calendario.
- ``correlation_matrix(prices)``: para cada par (i, j), Pearson y Spearman de
  los log-rendimientos sobre sus fechas comunes y número de observaciones
  (rendimientos) usadas. Diagonal = 1 y n = rendimientos de la serie.
- ``rolling_correlation(a, b, window=63)``: Pearson móvil hacia atrás de los
  log-rendimientos alineados; NaN hasta completar la ventana.
- ``normalized_prices(prices)``: base 100 en la primera fecha común (las
  columnas se alinean por intersección).
- Precios ≤ 0 → DataError (los logaritmos no están definidos).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

CONCEPTS = {
    "correlacion": "Correlación: los activos se mueven juntos día a día (rendimientos).",
    "cointegracion": (
        "Cointegración: existe una relación de largo plazo entre sus (log)precios; "
        "el spread tiende a volver."
    ),
    "aptitud_ou": (
        "Aptitud para OU: el spread se comporta como un proceso con reversión medible "
        "y una vida media útil para operar."
    ),
}


@dataclass(frozen=True)
class CorrelationResult:
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    nobs: pd.DataFrame  # enteros


def log_returns(prices: pd.Series) -> pd.Series:
    raise NotImplementedError


def correlation_matrix(prices: Mapping[str, pd.Series]) -> CorrelationResult:
    raise NotImplementedError


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    raise NotImplementedError


def normalized_prices(prices: Mapping[str, pd.Series] | pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    raise NotImplementedError
