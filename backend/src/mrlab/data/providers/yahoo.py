"""Proveedor Yahoo Finance vía ``yfinance`` (por defecto).

CONTRATO (a implementar):
- Precios con ``auto_adjust=True`` (ajustados por splits y dividendos). Por eso
  el PnL NO suma dividendos aparte (evita doble conteo). PriceType.ADJUSTED_CLOSE.
- ``yfinance`` se importa de forma perezosa. Para pruebas sin red, el
  constructor acepta ``ticker_factory`` (símbolo → objeto con ``.history(...)`` y
  ``.history_metadata``) y ``search_factory`` (consulta → objeto con ``.quotes``).
- Metadatos (moneda, tipo de instrumento, nombre, bolsa, zona horaria) desde
  ``history_metadata`` (no ``.info``: es lento y sufre más bloqueos).
  instrumentType: EQUITY→accion, ETF→etf, INDEX→indice, FUTURE→futuro,
  CRYPTOCURRENCY→cripto, CURRENCY→divisa, resto→otro. Índices y futuros:
  ``executable=False`` y ``suggested_proxy`` desde ``mrlab.data.proxies``.
- ``yfinance`` trata ``end`` como exclusivo: pedir ``end + 1 día``.
- Fechas: índice normalizado, sin zona horaria (fecha local de la bolsa; cripto
  = fecha UTC). Columnas en minúscula: close (obligatoria), open, high, low, volume.
- Reintentos con espera exponencial: ``max_retries`` intentos, espera
  ``backoff_base * 2**k`` (+ jitter determinista pequeño); ``sleep`` inyectable.
- Errores → ``ProviderError`` con mensaje claro en español, p. ej.:
  símbolo inexistente / sin datos (code="SIMBOLO_SIN_DATOS"), límite de
  peticiones (code="LIMITE_PETICIONES"), red (code="ERROR_RED"), respuesta
  inesperada (code="RESPUESTA_INESPERADA"). Nunca dejar escapar excepciones crudas.
- ``retrieved_at`` = ahora en UTC (inyectable vía ``clock`` para pruebas).
- ``source_note`` recuerda que Yahoo es una fuente gratuita no oficial.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from mrlab.domain.models import AssetInfo, PriceSeries

YAHOO_NOTICE = (
    "Yahoo Finance es una fuente gratuita y no oficial. Es adecuada para uso educativo, "
    "pero puede fallar, limitar peticiones o cambiar sin aviso."
)


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
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._ticker_factory = ticker_factory
        self._search_factory = search_factory
        self._clock = clock

    def search(self, query: str) -> list[AssetInfo]:
        raise NotImplementedError

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d") -> PriceSeries:
        raise NotImplementedError

    def get_metadata(self, symbol: str) -> AssetInfo:
        raise NotImplementedError
