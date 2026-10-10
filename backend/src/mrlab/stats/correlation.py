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
- Fechas duplicadas: la misma política que la validación de datos (2.3): orden
  estable por fecha y se conserva la ÚLTIMA aparición de cada fecha (el
  informe de validación es quien avisa de los duplicados). Entradas sin índice
  de fechas → DataError; ventana móvil < 2 → ConfigError. Nunca un ValueError
  crudo de pandas.

Notas de implementación:
- Las fechas con precio faltante (NaN) no son observaciones: se descartan antes
  de calcular rendimientos, de modo que el rendimiento siguiente abarca el hueco
  (igual que el de un fin de semana). Nunca se rellena.
- La positividad se exige sobre toda la serie recibida, no solo en las fechas
  comunes: un precio ≤ 0 indica un problema de datos aunque quede fuera del cruce.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mrlab.domain.errors import ConfigError, DataError

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


def _check_positive(prices: pd.Series, name: object) -> None:
    bad = prices[prices <= 0]
    if not bad.empty:
        when = bad.index[0]
        when_txt = when.date().isoformat() if isinstance(when, pd.Timestamp) else str(when)
        raise DataError(
            f"La serie «{name}» tiene {len(bad)} precio(s) menor(es) o igual(es) a cero "
            f"(el primero en {when_txt}: {bad.iloc[0]:g}). Los rendimientos logarítmicos requieren "
            "precios estrictamente positivos; revisa los datos de origen.",
            details={"serie": str(name), "fecha": when_txt, "n_no_positivos": int(len(bad))},
        )


def _clean_prices(prices: pd.Series, name: object = None) -> pd.Series:
    """Precios sin NaN ni fechas repetidas, en orden cronológico, como float y positivos.

    Misma política que ``mrlab.data.validation.clean_series``: orden ESTABLE (entre
    fechas repetidas se conserva el orden original), se queda la última aparición de
    cada fecha y después se descartan los NaN (nunca se rellenan).
    """
    label = prices.name if name is None else name
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataError(
            f"La serie «{label}» debe estar indexada por fecha para calcular rendimientos.",
            details={"serie": str(label)},
        )
    ordered = prices.sort_index(kind="stable")
    unique = ordered[~ordered.index.duplicated(keep="last")]
    try:
        clean = unique.dropna().astype(float)
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"La serie «{label}» contiene valores que no son números: revisa los datos de origen.",
            details={"serie": str(label)},
        ) from exc
    _check_positive(clean, label)
    return clean


def log_returns(prices: pd.Series) -> pd.Series:
    """Rendimientos logarítmicos ln(P_t / P_{t−1}) entre observaciones consecutivas válidas."""
    clean = _clean_prices(prices)
    returns = np.log(clean).diff().iloc[1:]
    returns.name = prices.name
    return returns


def _common_log_returns(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Log-rendimientos de dos series ya limpias, calculados sobre sus fechas comunes."""
    common = pd.concat([a, b], axis=1, keys=["a", "b"], join="inner")
    returns = np.log(common).diff().iloc[1:]
    return returns["a"].rename(a.name), returns["b"].rename(b.name)


def correlation_matrix(prices: Mapping[str, pd.Series]) -> CorrelationResult:
    """Pearson y Spearman de log-rendimientos por par sobre fechas comunes, con n por celda."""
    clean = {name: _clean_prices(series, name) for name, series in prices.items()}
    names = list(clean)
    pearson = pd.DataFrame(np.nan, index=names, columns=names)
    spearman = pd.DataFrame(np.nan, index=names, columns=names)
    nobs = pd.DataFrame(0, index=names, columns=names, dtype=int)

    for i, ni in enumerate(names):
        pearson.loc[ni, ni] = spearman.loc[ni, ni] = 1.0
        nobs.loc[ni, ni] = max(len(clean[ni]) - 1, 0)
        for nj in names[i + 1 :]:
            ra, rb = _common_log_returns(clean[ni], clean[nj])
            p = ra.corr(rb)
            s = ra.corr(rb, method="spearman")
            pearson.loc[ni, nj] = pearson.loc[nj, ni] = p
            spearman.loc[ni, nj] = spearman.loc[nj, ni] = s
            nobs.loc[ni, nj] = nobs.loc[nj, ni] = len(ra)
    return CorrelationResult(pearson=pearson, spearman=spearman, nobs=nobs)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    """Pearson móvil de los últimos ``window`` log-rendimientos comunes (solo pasado; NaN al inicio)."""
    if window < 2:
        raise ConfigError(
            f"La ventana de correlación móvil debe ser de al menos 2 observaciones (se recibió {window}).",
            code="VENTANA_INVALIDA",
            details={"window": window},
        )
    ra, rb = _common_log_returns(_clean_prices(a), _clean_prices(b))
    rolling = ra.rolling(window, min_periods=window).corr(rb)
    rolling.name = f"corr_{window}"
    return rolling


def normalized_prices(
    prices: Mapping[str, pd.Series] | pd.DataFrame, base: float = 100.0
) -> pd.DataFrame:
    """Precios en base ``base`` a partir de la primera fecha común a todas las series."""
    if isinstance(prices, pd.DataFrame):
        columns = {name: prices[name] for name in prices.columns}
    else:
        columns = dict(prices)
    clean = {name: _clean_prices(series, name) for name, series in columns.items()}
    frame = pd.concat(clean, axis=1, join="inner").sort_index()
    if frame.empty:
        raise DataError("Las series no tienen fechas en común: no se pueden comparar en base 100.")
    return frame / frame.iloc[0] * base
