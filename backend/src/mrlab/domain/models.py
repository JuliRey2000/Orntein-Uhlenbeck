"""Modelos de dominio compartidos por todo el núcleo (contratos estables).

Convenciones:
- Fechas: `pandas.DatetimeIndex` sin zona horaria, normalizado a medianoche y
  llamado ``date``. Para bolsa es la fecha local de la bolsa; para cripto en
  Yahoo es la fecha UTC del cierre diario.
- Los valores de los enums que aparecen en exportaciones están en español.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import date, datetime
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


class AssetClass(StrEnum):
    EQUITY = "accion"
    ETF = "etf"
    INDEX = "indice"
    FUTURE = "futuro"
    CRYPTO = "cripto"
    FX = "divisa"
    OTHER = "otro"


#: Clases que no se pueden comprar/vender directamente en la simulación.
NON_EXECUTABLE_CLASSES = frozenset({AssetClass.INDEX, AssetClass.FUTURE})


class PriceType(StrEnum):
    ADJUSTED_CLOSE = "cierre_ajustado"  # ajustado por splits y dividendos
    CLOSE = "cierre"
    SYNTHETIC = "sintetico"


PRICE_TYPE_LABELS = {
    PriceType.ADJUSTED_CLOSE: "Cierre ajustado (splits y dividendos)",
    PriceType.CLOSE: "Cierre sin ajustar",
    PriceType.SYNTHETIC: "SINTÉTICO (generado)",
}


class DataOrigin(StrEnum):
    HISTORICAL = "HISTORICO"
    SYNTHETIC = "SINTETICO"


class CalendarType(StrEnum):
    EXCHANGE = "bolsa"  # dos activos de bolsa: intersección de días de negociación
    CRYPTO_7D = "cripto_7d"  # dos criptomonedas: calendario de 7 días
    MIXED = "mixto"  # cripto + bolsa: solo días de negociación de la bolsa


class SpreadType(StrEnum):
    LOG = "log"
    PRICE = "precio"


class Segment(StrEnum):
    TRAIN = "formacion"
    VALIDATION = "validacion"
    TEST = "prueba"


class AssetInfo(BaseModel):
    """Identidad económica de un activo (sección 2.2)."""

    model_config = ConfigDict(frozen=True)

    symbol: str  # símbolo de la serie analizada en el proveedor (p. ej. ^GSPC)
    name: str  # nombre económico (p. ej. S&P 500)
    asset_class: AssetClass
    currency: str | None  # None si el proveedor no la informa (bloquea el análisis)
    provider: str
    exchange: str | None = None
    timezone: str | None = None
    origin: DataOrigin = DataOrigin.HISTORICAL
    executable: bool = True  # False para índices y futuros → exige proxy
    suggested_proxy: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _indices_and_futures_are_not_executable(cls, data):
        """Un índice o un futuro nunca es ejecutable, lo declare quien lo declare (D1)."""
        if isinstance(data, dict) and AssetClass(data.get("asset_class", AssetClass.OTHER)) in NON_EXECUTABLE_CLASSES:
            data = {**data, "executable": False}
        return data

    @property
    def is_synthetic(self) -> bool:
        return self.origin == DataOrigin.SYNTHETIC


class SeriesMeta(BaseModel):
    """Metadatos de una serie descargada/cargada (sección 2.2)."""

    model_config = ConfigDict(frozen=True)

    asset: AssetInfo
    price_type: PriceType
    interval: str = "1d"
    snapshot_id: str | None = None  # asignado por la caché (direccionado por contenido)
    retrieved_at: datetime | None = None  # UTC; None para sintéticos (deterministas)
    requested_start: date | None = None
    requested_end: date | None = None
    coverage_start: date | None = None
    coverage_end: date | None = None
    n_obs: int = 0
    source_note: str = ""
    seed: int | None = None  # semilla del generador (solo datos sintéticos)

    @property
    def is_synthetic(self) -> bool:
        return self.asset.is_synthetic

    @property
    def provider(self) -> str:
        return self.asset.provider

    @property
    def symbol(self) -> str:
        return self.asset.symbol


@dataclass(frozen=True)
class PriceSeries:
    """Serie de precios cruda tal como la entrega el proveedor.

    ``data`` tiene índice ``DatetimeIndex`` (nombre ``date``, sin zona horaria,
    normalizado) y al menos la columna ``close``; opcionalmente ``open``,
    ``high``, ``low``, ``volume``. Puede contener duplicados, desorden o NaN:
    ``mrlab.data.validation.clean_series`` los detecta y reporta.
    """

    meta: SeriesMeta
    data: pd.DataFrame

    @property
    def close(self) -> pd.Series:
        s = self.data["close"]
        s.name = self.meta.asset.symbol
        return s

    @property
    def symbol(self) -> str:
        return self.meta.asset.symbol

    @property
    def is_synthetic(self) -> bool:
        return self.meta.is_synthetic

    def with_meta(self, **changes) -> PriceSeries:
        return dc_replace(self, meta=self.meta.model_copy(update=changes))

    def with_data(self, data: pd.DataFrame) -> PriceSeries:
        return dc_replace(self, data=data)


def coverage_of(data: pd.DataFrame) -> tuple[date | None, date | None, int]:
    """(inicio, fin, n_obs) de un DataFrame indexado por fecha."""
    if data.empty:
        return None, None, 0
    idx = data.index
    return idx.min().date(), idx.max().date(), int(len(idx))
