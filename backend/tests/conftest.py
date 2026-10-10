"""Utilidades compartidas por las pruebas (no modificar sin coordinar: la usan todos los módulos)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mrlab.domain.models import (
    AssetClass,
    AssetInfo,
    DataOrigin,
    PriceSeries,
    PriceType,
    SeriesMeta,
    coverage_of,
)


def make_series(
    close: Sequence[float] | np.ndarray | pd.Series,
    *,
    symbol: str = "TEST",
    dates: Sequence | pd.DatetimeIndex | None = None,
    start: str = "2015-01-01",
    currency: str | None = "USD",
    asset_class: AssetClass = AssetClass.EQUITY,
    origin: DataOrigin = DataOrigin.HISTORICAL,
    provider: str = "test",
    freq: str = "B",
) -> PriceSeries:
    """Construye una PriceSeries de prueba. Por defecto días hábiles desde ``start``."""
    values = np.asarray(close, dtype=float)
    if dates is None:
        idx = pd.date_range(start=start, periods=len(values), freq=freq)
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    idx = idx.normalize()
    idx.name = "date"
    df = pd.DataFrame({"close": values}, index=idx)
    cs, ce, n = coverage_of(df)
    asset = AssetInfo(
        symbol=symbol,
        name=f"Prueba {symbol}",
        asset_class=asset_class,
        currency=currency,
        provider=provider,
        origin=origin,
    )
    meta = SeriesMeta(
        asset=asset,
        price_type=PriceType.SYNTHETIC if origin == DataOrigin.SYNTHETIC else PriceType.ADJUSTED_CLOSE,
        coverage_start=cs,
        coverage_end=ce,
        n_obs=n,
        requested_start=cs,
        requested_end=ce,
    )
    return PriceSeries(meta=meta, data=df)


def cointegrated_log_prices(
    n: int = 1000,
    seed: int = 0,
    beta: float = 1.0,
    alpha: float = 0.0,
    lam: float = 0.1,
    sigma: float = 0.01,
    b_vol: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """(log A, log B) con log A = α + β·log B + OU(λ, 0, σ) (transición exacta)."""
    rng = np.random.default_rng(seed)
    log_b = np.log(100.0) + np.cumsum(rng.normal(0.0, b_vol, n))
    decay = np.exp(-lam)
    sd = sigma / np.sqrt(2 * lam) * np.sqrt(1 - decay**2)
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = x[t - 1] * decay + sd * eps[t]
    return alpha + beta * log_b + x, log_b


@pytest.fixture
def series_factory():
    return make_series


@pytest.fixture
def d():
    """Atajo para fechas: d("2020-01-02")."""
    return lambda s: date.fromisoformat(s)
