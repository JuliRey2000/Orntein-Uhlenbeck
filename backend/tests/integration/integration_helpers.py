"""Utilidades de las pruebas de integración (no es un conftest: no oculta el compartido)."""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import make_series

from mrlab.data.cache import SnapshotCache
from mrlab.data.providers.csv_provider import CSVMetadata, CSVProvider
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.data.registry import ProviderRegistry
from mrlab.data.service import DataService
from mrlab.domain.models import AssetClass, DataOrigin, PriceSeries, PriceType
from mrlab.reporting.run_config import PairSpec, RunConfig, SeriesRef

START = "2012-01-02"  # lunes
WIDE = (date(1900, 1, 1), date(2100, 12, 31))
FIXED_MTIME = datetime(2020, 1, 2, 12, 0).timestamp()


def bdates(n: int, start: str = START) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n, name="date")


def series_pair(
    log_a: np.ndarray,
    log_b: np.ndarray,
    *,
    symbols: tuple[str, str] = ("AAA", "BBB"),
    currency_b: str | None = "USD",
    start: str = START,
) -> tuple[PriceSeries, PriceSeries]:
    a = make_series(np.exp(log_a), symbol=symbols[0], start=start)
    b = make_series(np.exp(log_b), symbol=symbols[1], start=start, currency=currency_b)
    return a, b


def synthetic_preset(prefix: str) -> tuple[PriceSeries, PriceSeries]:
    """Par sintético preset (p. ej. «SINT-COI») completo, sin pasar por la caché."""
    provider = SyntheticProvider()
    return (
        provider.get_history(f"{prefix}-A", *WIDE),
        provider.get_history(f"{prefix}-B", *WIDE),
    )


def csv_text(index: pd.DatetimeIndex, prices: np.ndarray) -> str:
    lines = ["date,close"]
    lines += [
        f"{d.date().isoformat()},{format(float(p), '.12g')}"
        for d, p in zip(index, prices, strict=True)
    ]
    return "\n".join(lines) + "\n"


def csv_provider(
    directory: Path,
    log_a: np.ndarray,
    log_b: np.ndarray,
    *,
    symbols: tuple[str, str] = ("AAA", "BBB"),
    origin: DataOrigin = DataOrigin.HISTORICAL,
    start: str = START,
) -> CSVProvider:
    """Registra dos CSV (con fecha de modificación fija, para que ``retrieved_at`` no varíe)."""
    provider = CSVProvider(directory)
    index = bdates(len(log_a), start)
    price_type = PriceType.SYNTHETIC if origin == DataOrigin.SYNTHETIC else PriceType.ADJUSTED_CLOSE
    for symbol, logs in zip(symbols, (log_a, log_b), strict=True):
        meta = CSVMetadata(
            symbol=symbol,
            name=f"Prueba {symbol}",
            currency="USD",
            asset_class=AssetClass.EQUITY,
            price_type=price_type,
            origin=origin,
        )
        provider.register(csv_text(index, np.exp(logs)), meta)
    for path in directory.glob("*.csv"):
        os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return provider


def service(cache_dir: Path, **providers) -> DataService:
    return DataService(ProviderRegistry(providers), SnapshotCache(cache_dir))


def pair_config(
    provider: str,
    symbols: tuple[str, str] = ("AAA", "BBB"),
    start: str = "1990-01-01",
    end: str = "2030-12-31",
    **updates,
) -> RunConfig:
    pair = PairSpec(
        a=SeriesRef(provider=provider, symbol=symbols[0], start=start, end=end),
        b=SeriesRef(provider=provider, symbol=symbols[1], start=start, end=end),
    )
    return RunConfig(pair=pair, **updates)


# --------------------------------------------------------------------------- escenarios


def explosive_spread(
    n: int = 900, seed: int = 1, rate: float = 0.008
) -> tuple[np.ndarray, np.ndarray]:
    """log A = log B + X con X explosivo (X_t = (1 + rate)·X_{t−1} + ε): spread NO estacionario."""
    rng = np.random.default_rng(seed)
    log_b = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    eps = rng.normal(0.0, 0.01, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = (1.0 + rate) * x[t - 1] + eps[t]
    return log_b + x, log_b


def broken_relationship(
    n: int = 1500,
    seed: int = 10,
    *,
    lam: float = 0.15,
    sigma: float = 0.01,
    break_start: int = 300,
    break_end: int = 600,
    rw_sigma: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Par cointegrado con una RUPTURA temporal: en [break_start, break_end) el spread deja de
    revertir (paseo aleatorio) y después vuelve a ser un OU. ``break_end=break_start`` = sin ruptura."""
    rng = np.random.default_rng(seed)
    log_b = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    eps = rng.standard_normal(n)
    decay = np.exp(-lam)
    step_sd = sigma / np.sqrt(2 * lam) * np.sqrt(1 - decay**2)
    x = np.zeros(n)
    for t in range(1, n):
        if break_start <= t < break_end:
            x[t] = x[t - 1] + rw_sigma * eps[t]
        else:
            x[t] = x[t - 1] * decay + step_sd * eps[t]
    return log_b + x, log_b


def random_walk_and_stationary(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A: paseo aleatorio (I(1)); B: AR(1) estacionario alrededor de log(50) (NO es I(1))."""
    rng = np.random.default_rng(seed)
    log_a = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    eps = rng.normal(0.0, 0.01, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + eps[t]
    return log_a, np.log(50.0) + x
