"""YahooProvider con factorías falsas: sin red."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, date, datetime

import pandas as pd
import pytest
import requests
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError, YFTzMissingError

from mrlab.data.providers.yahoo import YahooProvider, classify_error
from mrlab.domain.errors import ProviderError
from mrlab.domain.models import AssetClass, PriceType

FIXED_NOW = datetime(2026, 10, 10, 11, 30, tzinfo=UTC)

KO_META = {
    "currency": "USD",
    "symbol": "KO",
    "exchangeName": "NYQ",
    "fullExchangeName": "NYSE",
    "instrumentType": "EQUITY",
    "timezone": "EDT",
    "exchangeTimezoneName": "America/New_York",
    "longName": "The Coca-Cola Company",
    "shortName": "Coca-Cola Company (The)",
}


def yf_frame(dates, closes, tz: str | None = "America/New_York") -> pd.DataFrame:
    """Imita ``Ticker.history``: columnas capitalizadas e índice con zona horaria."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)), name="Date")
    if tz:
        idx = idx.tz_localize(tz)
    n = len(closes)
    return pd.DataFrame(
        {
            "Open": [c - 0.5 for c in closes],
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": list(range(1000, 1000 + n)),
            "Dividends": [0.0] * n,
            "Stock Splits": [0.0] * n,
        },
        index=idx,
    )


class FakeTicker:
    def __init__(self, frame=None, metadata=None, errors=()):
        self.frame = frame
        self.history_metadata = metadata if metadata is not None else {}
        self.errors = list(errors)
        self.calls: list[dict] = []

    def history(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return self.frame


class FakeSearch:
    def __init__(self, quotes):
        self.quotes = quotes


def provider(ticker: FakeTicker, **kwargs) -> tuple[YahooProvider, list[float], list[str]]:
    sleeps: list[float] = []
    symbols: list[str] = []

    def factory(symbol: str) -> FakeTicker:
        symbols.append(symbol)
        return ticker

    kwargs.setdefault("jitter", 0.0)
    p = YahooProvider(
        sleep=sleeps.append, ticker_factory=factory, clock=lambda: FIXED_NOW, **kwargs
    )
    return p, sleeps, symbols


KO_FRAME = yf_frame(["2020-01-02", "2020-01-03", "2020-01-06"], [54.69, 54.0, 54.05])


def test_history_call_uses_auto_adjust_and_exclusive_end_plus_one_day():
    ticker = FakeTicker(KO_FRAME, KO_META)
    p, _, symbols = provider(ticker)
    p.get_history("ko", date(2020, 1, 2), date(2020, 1, 6))

    assert symbols == ["KO"]
    (call,) = ticker.calls
    assert call["auto_adjust"] is True
    assert call["start"] == "2020-01-02"
    assert call["end"] == "2020-01-07"  # yfinance trata end como exclusivo
    assert call["interval"] == "1d"


def test_history_maps_metadata_and_price_type():
    p, _, _ = provider(FakeTicker(KO_FRAME, KO_META))
    s = p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31))

    asset = s.meta.asset
    assert asset.symbol == "KO" and asset.name == "The Coca-Cola Company"
    assert asset.asset_class == AssetClass.EQUITY
    assert asset.currency == "USD"
    assert asset.exchange == "NYSE" and asset.timezone == "America/New_York"
    assert asset.provider == "yahoo" and asset.executable and asset.suggested_proxy is None
    assert s.meta.price_type == PriceType.ADJUSTED_CLOSE
    assert s.meta.retrieved_at == FIXED_NOW
    assert s.meta.requested_start == date(2020, 1, 1) and s.meta.requested_end == date(2020, 1, 31)
    assert s.meta.coverage_start == date(2020, 1, 2) and s.meta.coverage_end == date(2020, 1, 6)
    assert s.meta.n_obs == 3
    assert "no oficial" in s.meta.source_note and "dividendos" in s.meta.source_note
    assert not s.is_synthetic


def test_columns_lowercase_and_tz_aware_index_normalized():
    p, _, _ = provider(FakeTicker(KO_FRAME, KO_META))
    s = p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31))

    assert list(s.data.columns) == ["close", "open", "high", "low", "volume"]
    assert (s.data.dtypes == "float64").all()
    idx = s.data.index
    assert isinstance(idx, pd.DatetimeIndex) and idx.tz is None and idx.name == "date"
    assert (idx == idx.normalize()).all()
    assert [ts.date() for ts in idx] == [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    assert s.close.tolist() == [54.69, 54.0, 54.05]


def test_tz_aware_index_keeps_local_exchange_date():
    # Medianoche en Tokio = 15:00 UTC del día anterior: debe conservarse la fecha local.
    frame = yf_frame(["2020-01-06", "2020-01-07"], [100.0, 101.0], tz="Asia/Tokyo")
    meta = {**KO_META, "symbol": "7203.T", "currency": "JPY", "exchangeTimezoneName": "Asia/Tokyo"}
    p, _, _ = provider(FakeTicker(frame, meta))
    s = p.get_history("7203.T", date(2020, 1, 6), date(2020, 1, 7))
    assert [ts.date() for ts in s.data.index] == [date(2020, 1, 6), date(2020, 1, 7)]
    assert s.meta.asset.currency == "JPY"


def test_naive_index_and_rows_outside_range_are_trimmed():
    frame = yf_frame(["2020-01-02", "2020-01-03", "2020-01-04"], [1.0, 2.0, 3.0], tz=None)
    p, _, _ = provider(FakeTicker(frame, {**KO_META, "instrumentType": "CRYPTOCURRENCY"}))
    s = p.get_history("BTC-USD", date(2020, 1, 2), date(2020, 1, 3))
    assert s.close.tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("instrument_type", "expected"),
    [
        ("EQUITY", AssetClass.EQUITY),
        ("ETF", AssetClass.ETF),
        ("INDEX", AssetClass.INDEX),
        ("FUTURE", AssetClass.FUTURE),
        ("CRYPTOCURRENCY", AssetClass.CRYPTO),
        ("CURRENCY", AssetClass.FX),
        ("MUTUALFUND", AssetClass.OTHER),
    ],
)
def test_instrument_type_mapping(instrument_type, expected):
    p, _, _ = provider(FakeTicker(KO_FRAME, {**KO_META, "instrumentType": instrument_type}))
    s = p.get_history("X", date(2020, 1, 1), date(2020, 1, 31))
    assert s.meta.asset.asset_class == expected


@pytest.mark.parametrize(
    ("symbol", "instrument_type", "proxy"),
    [("^GSPC", "INDEX", "SPY"), ("^NDX", "INDEX", "QQQ"), ("GC=F", "FUTURE", "GLD")],
)
def test_index_and_future_are_not_executable_and_suggest_proxy(symbol, instrument_type, proxy):
    meta = {"currency": "USD", "instrumentType": instrument_type, "symbol": symbol}
    p, _, _ = provider(FakeTicker(KO_FRAME, meta))
    asset = p.get_history(symbol, date(2020, 1, 1), date(2020, 1, 31)).meta.asset

    assert asset.executable is False
    assert asset.suggested_proxy == proxy
    assert asset.symbol == symbol  # nunca se sustituye en silencio
    assert any(proxy in note and "silencio" in note for note in asset.notes)


def test_index_without_known_proxy_asks_user_to_choose():
    p, _, _ = provider(FakeTicker(KO_FRAME, {"currency": "USD", "instrumentType": "INDEX"}))
    asset = p.get_history("^XYZ", date(2020, 1, 1), date(2020, 1, 31)).meta.asset
    assert not asset.executable and asset.suggested_proxy is None
    assert "no hay un proxy" in asset.notes[0]


def test_missing_instrument_type_falls_back_to_symbol_heuristic():
    p, _, _ = provider(FakeTicker(KO_FRAME, {"currency": "USD"}))
    asset = p.get_history("^GSPC", date(2020, 1, 1), date(2020, 1, 31)).meta.asset
    assert asset.asset_class == AssetClass.INDEX and not asset.executable
    assert asset.name == "S&P 500"


def test_bitcoin_spot_is_executable_with_short_selling_note():
    p, _, _ = provider(
        FakeTicker(KO_FRAME, {"currency": "USD", "instrumentType": "CRYPTOCURRENCY"})
    )
    asset = p.get_history("BTC-USD", date(2020, 1, 1), date(2020, 1, 31)).meta.asset
    assert asset.executable and asset.suggested_proxy is None
    assert any("corto" in note for note in asset.notes)


def test_retries_with_exponential_backoff_then_succeeds():
    ticker = FakeTicker(
        KO_FRAME, KO_META, errors=[YFRateLimitError(), ConnectionError("x"), TimeoutError()]
    )
    p, sleeps, _ = provider(ticker)
    s = p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31))
    assert s.meta.n_obs == 3
    assert sleeps == [1.0, 2.0, 4.0]
    assert len(ticker.calls) == 4


def test_retries_exhausted_raise_spanish_rate_limit_error():
    ticker = FakeTicker(KO_FRAME, KO_META, errors=[YFRateLimitError() for _ in range(10)])
    p, sleeps, _ = provider(ticker, max_retries=4, backoff_base=1.0)
    with pytest.raises(ProviderError) as exc_info:
        p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31))

    err = exc_info.value
    assert err.code == "LIMITE_PETICIONES"
    assert sleeps == [1.0, 2.0, 4.0, 8.0]  # 4 reintentos → 5 intentos
    assert len(ticker.calls) == 5
    assert err.details["intentos"] == 5
    assert "limitó las peticiones" in err.message and "5 intento" in err.message
    assert isinstance(err.__cause__, YFRateLimitError)


def test_backoff_jitter_is_small_and_deterministic():
    def run() -> list[float]:
        ticker = FakeTicker(KO_FRAME, KO_META, errors=[ConnectionError() for _ in range(10)])
        p, sleeps, _ = provider(ticker, jitter=0.1, backoff_base=0.5)
        with pytest.raises(ProviderError):
            p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31))
        return sleeps

    first, second = run(), run()
    assert first == second
    for k, delay in enumerate(first):
        base = 0.5 * 2**k
        assert base <= delay < base * 1.1


@pytest.mark.parametrize(
    ("exc", "code", "retried", "words"),
    [
        (YFRateLimitError(), "LIMITE_PETICIONES", True, "limitó"),
        (YFPricesMissingError("KO", ""), "SIMBOLO_SIN_DATOS", False, "no devolvió datos"),
        (YFTzMissingError("KO"), "SIMBOLO_SIN_DATOS", False, "no devolvió datos"),
        (ConnectionError("reset"), "ERROR_RED", True, "conexión"),
        (TimeoutError(), "ERROR_RED", True, "conexión"),
        (requests.exceptions.ConnectionError("dns"), "ERROR_RED", True, "conexión"),
        (
            requests.exceptions.HTTPError("429 Client Error: Too Many Requests"),
            "LIMITE_PETICIONES",
            True,
            "limitó",
        ),
        (
            requests.exceptions.HTTPError("500 Server Error"),
            "RESPUESTA_INESPERADA",
            True,
            "inesperada",
        ),
        (
            ValueError("Expecting value: line 1 column 1"),
            "RESPUESTA_INESPERADA",
            True,
            "inesperada",
        ),
        (KeyError("chart"), "RESPUESTA_INESPERADA", True, "inesperada"),
    ],
)
def test_errors_are_mapped_to_provider_error(exc, code, retried, words):
    ticker = FakeTicker(KO_FRAME, KO_META, errors=[exc, exc])
    p, sleeps, _ = provider(ticker, max_retries=1)
    with pytest.raises(ProviderError) as exc_info:
        p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31))
    err = exc_info.value
    assert err.code == code
    assert words in err.message
    assert "KO" in err.message
    assert len(ticker.calls) == (2 if retried else 1)
    assert sleeps == ([1.0] if retried else [])


def test_classify_error_by_class_name():
    assert classify_error(YFRateLimitError()) == ("LIMITE_PETICIONES", True)
    assert classify_error(YFPricesMissingError("X", "")) == ("SIMBOLO_SIN_DATOS", False)
    assert classify_error(RuntimeError("??")) == ("RESPUESTA_INESPERADA", True)


@pytest.mark.parametrize("empty", [pd.DataFrame(), None])
def test_empty_response_means_symbol_without_data(empty):
    ticker = FakeTicker(empty, {})
    p, sleeps, _ = provider(ticker)
    with pytest.raises(ProviderError) as exc_info:
        p.get_history("NOEXISTE", date(2020, 1, 1), date(2020, 1, 31))
    err = exc_info.value
    assert err.code == "SIMBOLO_SIN_DATOS"
    assert "NOEXISTE" in err.message and "2020-01-01" in err.message
    assert sleeps == [] and len(ticker.calls) == 1


def test_only_rows_outside_range_means_no_data():
    frame = yf_frame(["2020-02-03"], [1.0])
    p, _, _ = provider(FakeTicker(frame, KO_META))
    with pytest.raises(ProviderError, match="no devolvió datos") as exc_info:
        p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31))
    assert exc_info.value.code == "SIMBOLO_SIN_DATOS"


@pytest.mark.parametrize(
    "bad",
    ["no es una tabla", pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2020-01-02"]))],
)
def test_unexpected_response_shape(bad):
    p, _, _ = provider(FakeTicker(bad, KO_META))
    with pytest.raises(ProviderError) as exc_info:
        p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31))
    assert exc_info.value.code == "RESPUESTA_INESPERADA"


def test_only_daily_interval_is_supported():
    p, _, _ = provider(FakeTicker(KO_FRAME, KO_META))
    with pytest.raises(ProviderError) as exc_info:
        p.get_history("KO", date(2020, 1, 1), date(2020, 1, 31), interval="1wk")
    assert exc_info.value.code == "INTERVALO_NO_SOPORTADO"


def test_get_metadata_from_history_metadata():
    ticker = FakeTicker(
        KO_FRAME, {**KO_META, "symbol": "^GSPC", "instrumentType": "INDEX", "longName": "S&P 500"}
    )
    p, _, _ = provider(ticker)
    asset = p.get_metadata("^gspc")
    assert asset.symbol == "^GSPC" and asset.name == "S&P 500"
    assert not asset.executable and asset.suggested_proxy == "SPY"
    assert ticker.calls[0]["auto_adjust"] is True and ticker.calls[0]["period"] == "5d"


def test_get_metadata_without_metadata_is_symbol_without_data():
    p, _, _ = provider(FakeTicker(pd.DataFrame(), {}))
    with pytest.raises(ProviderError) as exc_info:
        p.get_metadata("NOEXISTE")
    assert exc_info.value.code == "SIMBOLO_SIN_DATOS"


def test_search_maps_quotes():
    quotes = [
        {
            "symbol": "KO",
            "shortname": "Coca-Cola",
            "longname": "The Coca-Cola Company",
            "quoteType": "EQUITY",
            "exchDisp": "NYSE",
        },
        {"symbol": "^GSPC", "shortname": "S&P 500", "quoteType": "INDEX", "exchange": "SNP"},
        {"shortname": "sin símbolo", "quoteType": "EQUITY"},
    ]
    queries: list[str] = []

    def search_factory(q: str) -> FakeSearch:
        queries.append(q)
        return FakeSearch(quotes)

    p = YahooProvider(search_factory=search_factory, sleep=lambda s: None)
    results = p.search("  coca ")
    assert queries == ["coca"]
    assert [r.symbol for r in results] == ["KO", "^GSPC"]
    assert results[0].name == "The Coca-Cola Company" and results[0].exchange == "NYSE"
    assert results[1].asset_class == AssetClass.INDEX and results[1].suggested_proxy == "SPY"
    assert p.search("   ") == [] and queries == ["coca"]


def test_search_errors_are_mapped():
    def failing(q: str):
        raise ConnectionError("sin red")

    sleeps: list[float] = []
    p = YahooProvider(search_factory=failing, sleep=sleeps.append, max_retries=2, jitter=0.0)
    with pytest.raises(ProviderError) as exc_info:
        p.search("KO")
    assert exc_info.value.code == "ERROR_RED" and "buscar" in exc_info.value.message
    assert sleeps == [1.0, 2.0]


def test_yfinance_is_imported_lazily():
    code = (
        "import sys\n"
        "from mrlab.data.providers.yahoo import YahooProvider\n"
        "YahooProvider()\n"
        "print('yfinance' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.strip() == "False"


def test_default_factory_makes_yfinance_raise_errors():
    # En un subproceso: no se altera la configuración global de yfinance de esta sesión.
    code = (
        "from mrlab.data.providers.yahoo import _yfinance\n"
        "yf = _yfinance()\n"
        "print(yf.config.debug.hide_exceptions)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.strip() == "False"
