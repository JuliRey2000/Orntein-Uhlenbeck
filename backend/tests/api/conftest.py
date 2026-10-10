"""Fixtures de la API: aplicación con directorios temporales y un proveedor «yahoo» falso, sin red.

Símbolos del proveedor falso (precios deterministas, días hábiles desde 2016-01-04, 1.500 obs.):
- KO / PEP: par cointegrado (spread OU con vida media ≈ 7 días).
- XOM / CVX: paseos aleatorios independientes (sin cointegración).
- ^GSPC: índice (no ejecutable, proxy sugerido SPY); SPY e IVV: ETF que siguen al índice.
- BTC-USD: cripto (calendario de 7 días).
- SAP.DE: acción en EUR (para probar monedas distintas).
- FALLA_RED → ProviderError ERROR_RED; LIMITE → ProviderError LIMITE_PETICIONES;
  cualquier otro símbolo → ProviderError SIMBOLO_SIN_DATOS.
``FakeYahoo.calls`` registra cada descarga (para comprobar la caché).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mrlab.api.main import create_app
from mrlab.config import BACKEND_ROOT, Settings
from mrlab.data.calendar import to_date_index
from mrlab.data.providers.csv_provider import CSVProvider
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.data.proxies import build_asset_info
from mrlab.data.registry import ProviderRegistry
from mrlab.domain.errors import ProviderError
from mrlab.domain.models import (
    AssetClass,
    AssetInfo,
    PriceSeries,
    PriceType,
    SeriesMeta,
    coverage_of,
)

N_OBS = 1500
START = "2016-01-04"


def _bdays() -> pd.DatetimeIndex:
    return pd.bdate_range(START, periods=N_OBS)


def _ou(rng: np.random.Generator, n: int, lam: float = 0.1, sigma: float = 0.01) -> np.ndarray:
    decay = np.exp(-lam)
    sd = sigma / np.sqrt(2 * lam) * np.sqrt(1 - decay**2)
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = x[t - 1] * decay + sd * eps[t]
    return x


def _universe() -> dict[str, tuple[AssetClass, str, str, pd.Series]]:
    """símbolo → (clase, moneda, nombre, precios)."""
    rng = np.random.default_rng(2024)
    idx = _bdays()
    n = len(idx)
    out: dict[str, tuple[AssetClass, str, str, pd.Series]] = {}

    log_pep = np.log(60.0) + np.cumsum(rng.normal(0.0002, 0.01, n))
    log_ko = 0.2 + 1.0 * log_pep + _ou(rng, n)
    out["KO"] = (AssetClass.EQUITY, "USD", "Coca-Cola (prueba)", pd.Series(np.exp(log_ko), idx))
    out["PEP"] = (AssetClass.EQUITY, "USD", "PepsiCo (prueba)", pd.Series(np.exp(log_pep), idx))

    out["XOM"] = (
        AssetClass.EQUITY,
        "USD",
        "ExxonMobil (prueba)",
        pd.Series(np.exp(np.log(80) + np.cumsum(rng.normal(0, 0.012, n))), idx),
    )
    out["CVX"] = (
        AssetClass.EQUITY,
        "USD",
        "Chevron (prueba)",
        pd.Series(np.exp(np.log(110) + np.cumsum(rng.normal(0, 0.012, n))), idx),
    )

    log_idx = np.log(2000.0) + np.cumsum(rng.normal(0.0003, 0.009, n))
    out["^GSPC"] = (AssetClass.INDEX, "USD", "S&P 500 (prueba)", pd.Series(np.exp(log_idx), idx))
    out["SPY"] = (
        AssetClass.ETF,
        "USD",
        "SPDR S&P 500 (prueba)",
        pd.Series(np.exp(log_idx - np.log(10) + _ou(rng, n, lam=0.3, sigma=0.002)), idx),
    )
    out["IVV"] = (
        AssetClass.ETF,
        "USD",
        "iShares Core S&P 500 (prueba)",
        pd.Series(np.exp(log_idx - np.log(9.9) + _ou(rng, n, lam=0.3, sigma=0.002)), idx),
    )

    cidx = pd.date_range(START, periods=int(N_OBS * 7 / 5), freq="D")
    out["BTC-USD"] = (
        AssetClass.CRYPTO,
        "USD",
        "Bitcoin (prueba)",
        pd.Series(np.exp(np.log(400) + np.cumsum(rng.normal(0.001, 0.03, len(cidx)))), cidx),
    )
    out["SAP.DE"] = (
        AssetClass.EQUITY,
        "EUR",
        "SAP (prueba, EUR)",
        pd.Series(np.exp(np.log(70) + np.cumsum(rng.normal(0, 0.01, n))), idx),
    )
    return out


class FakeYahoo:
    """Doble de prueba del proveedor Yahoo (misma interfaz DataProvider)."""

    name = "yahoo"
    description = "Proveedor falso de pruebas (sustituye a Yahoo Finance; sin red)."

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date]] = []
        self._data = _universe()

    def _info(self, symbol: str) -> AssetInfo:
        sym = symbol.strip().upper()
        if sym == "FALLA_RED":
            raise ProviderError(
                "No se pudo conectar con Yahoo Finance (prueba).",
                code="ERROR_RED",
                details={"symbol": sym},
            )
        if sym == "LIMITE":
            raise ProviderError(
                "Yahoo Finance limitó las peticiones (prueba).",
                code="LIMITE_PETICIONES",
                details={"symbol": sym},
            )
        if sym not in self._data:
            raise ProviderError(
                f"Yahoo Finance no devolvió datos para «{sym}» (prueba).",
                code="SIMBOLO_SIN_DATOS",
                details={"symbol": sym},
            )
        cls, cur, name, _ = self._data[sym]
        return build_asset_info(sym, cls, name=name, currency=cur, provider=self.name)

    def search(self, query: str) -> list[AssetInfo]:
        q = query.strip().upper()
        return [self._info(s) for s in self._data if q in s or q in self._data[s][2].upper()]

    def get_metadata(self, symbol: str) -> AssetInfo:
        return self._info(symbol)

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d") -> PriceSeries:
        info = self._info(symbol)
        self.calls.append((info.symbol, start, end))
        s = self._data[info.symbol][3]
        s = s.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if s.empty:
            raise ProviderError(
                f"Sin datos para «{info.symbol}» en el rango pedido (prueba).",
                code="SIMBOLO_SIN_DATOS",
                details={"symbol": info.symbol},
            )
        df = pd.DataFrame({"close": s.to_numpy(dtype=float)}, index=to_date_index(s.index))
        cs, ce, n = coverage_of(df)
        meta = SeriesMeta(
            asset=info,
            price_type=PriceType.ADJUSTED_CLOSE,
            retrieved_at=pd.Timestamp("2026-10-10T11:30:00Z").to_pydatetime(),
            requested_start=start,
            requested_end=end,
            coverage_start=cs,
            coverage_end=ce,
            n_obs=n,
            source_note="Proveedor falso de pruebas.",
        )
        return PriceSeries(meta=meta, data=df)


@pytest.fixture
def lab_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        cache_dir=data_dir / "cache",
        studies_dir=data_dir / "studies",
        uploads_dir=data_dir / "uploads",
        providers_file=BACKEND_ROOT / "providers.toml",
        cors_origins="http://localhost:5173",
        job_workers=2,
        max_upload_mb=1.0,
    )


@pytest.fixture
def fake_yahoo() -> FakeYahoo:
    return FakeYahoo()


@pytest.fixture
def registry(lab_settings: Settings, fake_yahoo: FakeYahoo) -> ProviderRegistry:
    return ProviderRegistry.from_toml(
        lab_settings.providers_file,
        overrides={
            "yahoo": fake_yahoo,
            "csv": CSVProvider(lab_settings.uploads_dir),
            "synthetic": SyntheticProvider(),
        },
    )


@pytest.fixture
def app(lab_settings: Settings, registry: ProviderRegistry):
    return create_app(lab_settings, registry)


@pytest.fixture
def client(app) -> TestClient:
    # raise_server_exceptions=False: los 500 se prueban como respuestas, no como excepciones.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def pair_config(a: str = "KO", b: str = "PEP", provider: str = "yahoo", **extra) -> dict:
    """Cuerpo JSON mínimo de un RunConfig para el proveedor falso."""
    ref = {"provider": provider, "start": "2016-01-04", "end": "2021-12-31"}
    cfg = {"pair": {"a": {**ref, "symbol": a}, "b": {**ref, "symbol": b}}}
    cfg.update(extra)
    return cfg
