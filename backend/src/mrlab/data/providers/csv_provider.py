"""Proveedor CSV: archivos con columnas ``date, close`` (opcional open, high, low, volume).

CONTRATO (a implementar):
- ``parse_csv(content, meta)``: acepta bytes o str (UTF-8, con o sin BOM).
  Detecta separador ``,`` ``;`` o tabulador. Con ``;`` acepta coma decimal.
  Encabezados sin distinguir mayúsculas y con espacios recortados; se aceptan
  alias en español: fecha→date, cierre→close, apertura→open, maximo/máximo→high,
  minimo/mínimo→low, volumen→volume. Fechas ISO ``AAAA-MM-DD`` (también con hora,
  que se descarta). NO ordena ni elimina duplicados ni rellena: entrega la serie
  cruda para que ``validation`` lo reporte. Celdas vacías de close → NaN.
- Errores (``DataError``, mensajes en español que digan qué hacer):
  archivo vacío (CSV_VACIO), falta columna obligatoria (CSV_FALTA_COLUMNA, lista
  las columnas encontradas), fechas no interpretables (CSV_FECHA_INVALIDA, con
  ejemplos de filas), close no numérico (CSV_VALOR_NO_NUMERICO, con ejemplos).
- Metadatos obligatorios declarados por el usuario (``CSVMetadata``): moneda,
  clase de activo, tipo de precio. Sin ellos no se puede validar la moneda ni
  elegir 252/365.
- ``CSVProvider(directory)``: cada serie es ``<archivo>.csv`` + ``<archivo>.meta.json``
  (CSVMetadata serializado). El símbolo es el del sidecar. Si ``origin`` es
  SINTETICO, el archivo se guarda con prefijo ``SINTETICO_``.
  ``register(content, meta)`` valida (parse_csv) y guarda ambos archivos.
  ``get_history`` filtra por [start, end] inclusive. ``search`` busca por
  subcadena (sin distinguir mayúsculas) en símbolo o nombre.
  ``retrieved_at`` = fecha de modificación del archivo (UTC) o None si sintético.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel

from mrlab.domain.models import AssetClass, AssetInfo, DataOrigin, PriceSeries, PriceType


class CSVMetadata(BaseModel):
    symbol: str
    name: str
    currency: str
    asset_class: AssetClass
    price_type: PriceType = PriceType.CLOSE
    origin: DataOrigin = DataOrigin.HISTORICAL
    notes: tuple[str, ...] = ()


def parse_csv(content: bytes | str, meta: CSVMetadata, provider_name: str = "csv") -> PriceSeries:
    raise NotImplementedError


class CSVProvider:
    name = "csv"
    description = "Archivos CSV cargados por el usuario (columnas date, close)."

    def __init__(self, directory: Path | str | None = None):
        from mrlab.config import get_settings

        self.directory = Path(directory) if directory else get_settings().uploads_dir

    def register(self, content: bytes | str, meta: CSVMetadata) -> PriceSeries:
        raise NotImplementedError

    def list_symbols(self) -> list[str]:
        raise NotImplementedError

    def search(self, query: str) -> list[AssetInfo]:
        raise NotImplementedError

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d") -> PriceSeries:
        raise NotImplementedError

    def get_metadata(self, symbol: str) -> AssetInfo:
        raise NotImplementedError
