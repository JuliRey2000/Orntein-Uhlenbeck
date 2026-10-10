"""Proveedor Yahoo Finance vía ``yfinance`` (por defecto).

CONTRATO:
- Precios con ``auto_adjust=True`` (ajustados por splits y dividendos). Por eso
  el PnL NO suma dividendos aparte (evita doble conteo). PriceType.ADJUSTED_CLOSE.
- ``yfinance`` se importa de forma perezosa. Para pruebas sin red, el
  constructor acepta ``ticker_factory`` (símbolo → objeto con ``.history(...)`` y
  ``.history_metadata``) y ``search_factory`` (consulta → objeto con ``.quotes``).
- Metadatos (moneda, tipo de instrumento, nombre, bolsa, zona horaria) desde
  ``history_metadata`` (no ``.info``: es lento y sufre más bloqueos).
  instrumentType: EQUITY→accion, ETF→etf, INDEX→indice, FUTURE→futuro,
  CRYPTOCURRENCY→cripto, CURRENCY→divisa, resto→otro (si falta, ``^…`` → índice
  y ``…=F`` → futuro). Índices y futuros: ``executable=False`` y
  ``suggested_proxy`` desde ``mrlab.data.proxies`` (con la explicación en ``notes``).
- Los símbolos se normalizan a mayúsculas (Yahoo no distingue).
- ``yfinance`` trata ``end`` como exclusivo: pedir ``end + 1 día`` (fechas ISO).
  Las filas fuera de [start, end] que devuelva Yahoo se descartan.
- Fechas: índice normalizado, sin zona horaria (fecha local de la bolsa; cripto
  = fecha UTC). Columnas en minúscula: close (obligatoria), open, high, low,
  volume, en float64; el resto (dividendos, splits) se descarta.
- Reintentos con espera exponencial: ``max_retries`` REINTENTOS tras el primer
  intento (por defecto 4 → hasta 5 llamadas, esperas 1, 2, 4 y 8 s), espera
  ``backoff_base * 2**k`` más un jitter determinista de hasta ``jitter`` × la
  espera (derivado del símbolo y del intento); ``sleep`` inyectable. Solo se
  reintentan errores transitorios (límite de peticiones, red, respuesta
  inesperada); «sin datos» no se reintenta.
- Errores → ``ProviderError`` con mensaje claro en español:
  símbolo inexistente / sin datos (code="SIMBOLO_SIN_DATOS", también con
  respuesta vacía), límite de peticiones (code="LIMITE_PETICIONES"), red
  (code="ERROR_RED"), respuesta inesperada (code="RESPUESTA_INESPERADA").
  Nunca dejar escapar excepciones crudas.
- ``retrieved_at`` = ahora en UTC (inyectable vía ``clock`` para pruebas).
- ``source_note`` recuerda que Yahoo es una fuente gratuita no oficial.
"""

from __future__ import annotations

import re
import time
import zlib
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from mrlab.data.calendar import to_date_index
from mrlab.data.providers.base import check_interval
from mrlab.data.proxies import build_asset_info, looks_non_executable
from mrlab.domain.errors import DataError, ProviderError
from mrlab.domain.models import (
    AssetClass,
    AssetInfo,
    PriceSeries,
    PriceType,
    SeriesMeta,
    coverage_of,
)

YAHOO_NOTICE = (
    "Yahoo Finance es una fuente gratuita y no oficial. Es adecuada para uso educativo, "
    "pero puede fallar, limitar peticiones o cambiar sin aviso."
)

ADJUSTMENT_NOTE = (
    "Precios de cierre ajustados por splits y dividendos (auto_adjust=True): los dividendos ya "
    "están incluidos en el precio y no se suman aparte en el PnL, para no contarlos dos veces."
)

INSTRUMENT_TYPES: dict[str, AssetClass] = {
    "EQUITY": AssetClass.EQUITY,
    "ETF": AssetClass.ETF,
    "INDEX": AssetClass.INDEX,
    "FUTURE": AssetClass.FUTURE,
    "CRYPTOCURRENCY": AssetClass.CRYPTO,
    "CURRENCY": AssetClass.FX,
}

PRICE_COLUMNS = ("close", "open", "high", "low", "volume")

# Códigos de error estables.
NO_DATA = "SIMBOLO_SIN_DATOS"
RATE_LIMIT = "LIMITE_PETICIONES"
NETWORK = "ERROR_RED"
UNEXPECTED = "RESPUESTA_INESPERADA"

_NO_DATA_EXC = {
    "YFTickerMissingError",
    "YFPricesMissingError",
    "YFTzMissingError",
    "YFInvalidPeriodError",
}
_NETWORK_EXC = {
    "RequestException",
    "CurlError",
    "ConnectionError",
    "Timeout",
    "TimeoutError",
    "ProxyError",
    "SSLError",
}
_RATE_LIMIT_TEXT = re.compile(r"too many requests|rate.?limit|\b429\b", re.IGNORECASE)

T = TypeVar("T")


def classify_error(exc: BaseException) -> tuple[str, bool]:
    """(código, ¿reintentar?) para una excepción de yfinance o de la capa HTTP.

    Se compara por nombre de clase para no importar yfinance ni sus clientes HTTP.
    """
    names = {k.__name__ for k in type(exc).__mro__}
    if "YFRateLimitError" in names or _RATE_LIMIT_TEXT.search(str(exc)):
        return RATE_LIMIT, True
    if names & _NO_DATA_EXC:
        return NO_DATA, False
    if "HTTPError" in names:  # 5xx, 404 de la API…: respuesta inesperada, quizá transitoria
        return UNEXPECTED, True
    if isinstance(exc, ConnectionError | TimeoutError) or names & _NETWORK_EXC:
        return NETWORK, True
    return UNEXPECTED, True


def asset_class_from_yahoo(instrument_type: str | None, symbol: str) -> AssetClass:
    if instrument_type:
        return INSTRUMENT_TYPES.get(instrument_type.strip().upper(), AssetClass.OTHER)
    if looks_non_executable(symbol):
        return AssetClass.INDEX if symbol.startswith("^") else AssetClass.FUTURE
    return AssetClass.OTHER


def _yfinance() -> Any:
    """Importa ``yfinance`` (perezoso) y le pide que LANCE sus errores.

    Por defecto yfinance ≥ 1.0 solo registra los fallos de red y devuelve una
    tabla vacía, que aquí se confundiría con «símbolo sin datos». Con
    ``hide_exceptions = False`` (su configuración documentada) cada fallo llega a
    ``classify_error`` y recibe el mensaje y los reintentos que le corresponden.
    """
    import yfinance as yf

    debug = getattr(getattr(yf, "config", None), "debug", None)
    if debug is not None and getattr(debug, "hide_exceptions", False):
        debug.hide_exceptions = False
    return yf


def _default_ticker(symbol: str) -> Any:
    return _yfinance().Ticker(symbol)


def _default_search(query: str) -> Any:
    return _yfinance().Search(query, max_results=10, news_count=0)


def _read_metadata(ticker: Any) -> dict[str, Any]:
    # Solo se tolera que falte el atributo; cualquier otro fallo (p. ej. límite de
    # peticiones al cargarlo) sube a la lógica de reintentos.
    md = getattr(ticker, "history_metadata", None)
    return md if isinstance(md, dict) else {}


class YahooProvider:
    name = "yahoo"
    description = YAHOO_NOTICE

    def __init__(
        self,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        ticker_factory: Callable[[str], Any] | None = None,
        search_factory: Callable[[str], Any] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        jitter: float = 0.1,
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.jitter = jitter
        self._sleep = sleep
        self._ticker_factory = ticker_factory or _default_ticker
        self._search_factory = search_factory or _default_search
        self._clock = clock

    # ------------------------------------------------------------------ API

    def search(self, query: str) -> list[AssetInfo]:
        q = query.strip()
        if not q:
            return []
        result = self._with_retries(lambda: self._search_factory(q), f"buscar «{q}»", q)
        quotes = getattr(result, "quotes", None) or []
        if not isinstance(quotes, list):
            raise self._unexpected(
                q, f"buscar «{q}»", "la lista de resultados no tiene el formato esperado"
            )
        found: list[AssetInfo] = []
        for quote in quotes:
            if not isinstance(quote, dict) or not quote.get("symbol"):
                continue
            symbol = str(quote["symbol"]).strip().upper()
            found.append(
                build_asset_info(
                    symbol,
                    asset_class_from_yahoo(quote.get("quoteType"), symbol),
                    name=quote.get("longname") or quote.get("shortname"),
                    currency=None,  # la búsqueda no la informa; se confirma al descargar
                    exchange=quote.get("exchDisp") or quote.get("exchange"),
                    provider=self.name,
                )
            )
        return found

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d") -> PriceSeries:
        check_interval(interval)
        sym = self._clean_symbol(symbol)
        if start > end:
            raise DataError(
                f"Rango de fechas inválido: el inicio ({start.isoformat()}) es posterior al fin "
                f"({end.isoformat()}).",
                code="RANGO_FECHAS_INVALIDO",
            )
        action = f"descargar «{sym}»"

        def fetch() -> tuple[Any, dict[str, Any]]:
            ticker = self._ticker_factory(sym)
            raw = ticker.history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # yfinance: fin exclusivo
                interval=interval,
                auto_adjust=True,
                actions=False,
            )
            return raw, _read_metadata(ticker)

        raw, md = self._with_retries(fetch, action, sym)
        data = self._normalize_history(raw, sym, start, end)
        cov_start, cov_end, n_obs = coverage_of(data)
        meta = SeriesMeta(
            asset=self._asset_from_metadata(sym, md),
            price_type=PriceType.ADJUSTED_CLOSE,
            interval=interval,
            retrieved_at=self._clock(),
            requested_start=start,
            requested_end=end,
            coverage_start=cov_start,
            coverage_end=cov_end,
            n_obs=n_obs,
            source_note=f"Yahoo Finance (yfinance). {ADJUSTMENT_NOTE} {YAHOO_NOTICE}",
        )
        return PriceSeries(meta=meta, data=data)

    def get_metadata(self, symbol: str) -> AssetInfo:
        sym = self._clean_symbol(symbol)

        def fetch() -> tuple[Any, dict[str, Any]]:
            ticker = self._ticker_factory(sym)
            raw = ticker.history(period="5d", interval="1d", auto_adjust=True, actions=False)
            return raw, _read_metadata(ticker)

        _, md = self._with_retries(fetch, f"consultar «{sym}»", sym)
        if not md:
            raise self._no_data(sym)
        return self._asset_from_metadata(sym, md)

    # ------------------------------------------------------------------ internos

    def _clean_symbol(self, symbol: str) -> str:
        sym = (symbol or "").strip().upper()
        if not sym:
            raise ProviderError(
                "Indica un símbolo de Yahoo Finance (p. ej. «KO»).", code="SIMBOLO_VACIO"
            )
        return sym

    def _asset_from_metadata(self, symbol: str, md: dict[str, Any]) -> AssetInfo:
        return build_asset_info(
            symbol,
            asset_class_from_yahoo(md.get("instrumentType"), symbol),
            name=md.get("longName") or md.get("shortName"),
            currency=md.get("currency"),
            exchange=md.get("fullExchangeName") or md.get("exchangeName"),
            timezone=md.get("exchangeTimezoneName") or md.get("timezone"),
            provider=self.name,
        )

    def _normalize_history(self, raw: Any, symbol: str, start: date, end: date) -> pd.DataFrame:
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            raise self._no_data(symbol, start, end)
        if not isinstance(raw, pd.DataFrame):
            raise self._unexpected(
                symbol, f"descargar «{symbol}»", f"se recibió {type(raw).__name__}"
            )
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "close" not in df.columns:
            raise self._unexpected(
                symbol,
                f"descargar «{symbol}»",
                f"faltan los precios de cierre (columnas: {list(df.columns)})",
            )
        cols = [c for c in PRICE_COLUMNS if c in df.columns]
        try:
            out = pd.DataFrame(
                {c: pd.to_numeric(df[c]).to_numpy(dtype="float64") for c in cols},
                index=to_date_index(df.index),
            )
        except (TypeError, ValueError) as exc:
            raise self._unexpected(symbol, f"descargar «{symbol}»", str(exc)) from exc
        in_range = (out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))
        out = out[np.asarray(in_range)]
        if out.empty:
            raise self._no_data(symbol, start, end)
        return out

    def _delay(self, attempt: int, key: str) -> float:
        base = self.backoff_base * 2**attempt
        unit = zlib.crc32(f"{key}:{attempt}".encode()) / 2**32  # determinista en [0, 1)
        return base + base * self.jitter * unit

    def _with_retries(self, fn: Callable[[], T], action: str, key: str) -> T:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                return fn()
            except (ProviderError, DataError):
                raise
            except Exception as exc:  # cualquier fallo se traduce a ProviderError en español
                code, retryable = classify_error(exc)
                if not retryable or attempt == attempts - 1:
                    raise self._error(code, key, action, attempt + 1, exc) from exc
                self._sleep(self._delay(attempt, key))
        raise AssertionError("inalcanzable")  # pragma: no cover

    def _no_data(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> ProviderError:
        span = f" entre {start.isoformat()} y {end.isoformat()}" if start and end else ""
        return ProviderError(
            f"Yahoo Finance no devolvió datos para «{symbol}»{span}. Revisa que el símbolo esté "
            "escrito como en Yahoo (p. ej. «KO», «^GSPC», «BTC-USD», «GC=F») y que el rango de "
            "fechas incluya días de negociación. Si el símbolo es correcto, puede ser un fallo "
            "temporal de Yahoo: inténtalo de nuevo en unos minutos.",
            code=NO_DATA,
            details={"symbol": symbol},
        )

    def _unexpected(self, symbol: str, action: str, why: str) -> ProviderError:
        return ProviderError(
            f"Yahoo Finance devolvió una respuesta inesperada al {action} ({why}). Es una API no "
            "oficial y puede haber cambiado: inténtalo más tarde o carga los datos desde un CSV.",
            code=UNEXPECTED,
            details={"symbol": symbol},
        )

    def _error(
        self, code: str, key: str, action: str, attempts: int, exc: BaseException
    ) -> ProviderError:
        tries = f"tras {attempts} intento(s)"
        if code == NO_DATA:
            err = self._no_data(key)
            err.details.update(error=str(exc)[:300])
            return err
        messages = {
            RATE_LIMIT: (
                f"Yahoo Finance limitó las peticiones (demasiadas solicitudes) al {action}, {tries}. "
                "Espera unos minutos antes de volver a intentarlo; los datos ya descargados "
                "siguen disponibles en la caché."
            ),
            NETWORK: (
                f"No se pudo conectar con Yahoo Finance al {action} ({tries}). Revisa tu conexión "
                "a internet o el proxy. Mientras tanto puedes usar datos en caché, un CSV o el "
                "generador sintético."
            ),
            UNEXPECTED: (
                f"Yahoo Finance devolvió una respuesta inesperada al {action} ({tries}). Es una "
                "API no oficial y puede haber cambiado: inténtalo más tarde o carga los datos "
                "desde un CSV."
            ),
        }
        return ProviderError(
            messages[code],
            code=code,
            details={
                "symbol": key,
                "intentos": attempts,
                "error_tipo": type(exc).__name__,
                "error": str(exc)[:300],
            },
        )
