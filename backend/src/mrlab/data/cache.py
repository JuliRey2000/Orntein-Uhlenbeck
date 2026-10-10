"""Caché local de instantáneas inmutables en Parquet (sección 1 y 15).

Por qué instantáneas inmutables: Yahoo reajusta TODO el histórico con cada
dividendo/split. Mezclar descargas de fechas distintas corrompería la serie, y
reproducir una ejecución exige usar exactamente los mismos precios.

CONTRATO (a implementar):
- ``content_hash(df)``: sha256 de una representación canónica del DataFrame
  (índice ISO AAAA-MM-DD, columnas en orden fijo close, open, high, low, volume
  si existen, floats con ``repr``). Independiente del orden de inserción de columnas.
- ``snapshot_id`` = ``f"{provider}_{simbolo_seguro}_{interval}_{hash[:12]}"``
  (``simbolo_seguro`` = símbolo con todo lo que no sea [A-Za-z0-9.-] → ``_``,
  p. ej. ``^GSPC`` → ``_GSPC``, ``GC=F`` → ``GC_F``). Direccionado por contenido:
  la misma descarga produce el mismo id (idempotente).
- ``put(series)``: escribe ``root/<provider>/<simbolo_seguro>/<snapshot_id>.parquet``
  con los metadatos (SeriesMeta en JSON) dentro de los metadatos del esquema
  Parquet (clave ``mrlab_meta``), y actualiza ``root/index.json`` de forma
  atómica (archivo temporal + ``os.replace``). Devuelve la serie con
  ``meta.snapshot_id`` asignado y coverage/n_obs actualizados. Si el id ya existe,
  no reescribe datos (conserva el ``retrieved_at`` original).
- ``get(snapshot_id)`` → PriceSeries idéntica a la guardada (mismo índice,
  columnas y dtypes float64). Inexistente → DataError(code="SNAPSHOT_NO_ENCONTRADO")
  explicando que re-descargar puede dar precios distintos.
- ``list(provider=None, symbol=None)`` → SeriesMeta ordenados por retrieved_at desc.
- ``latest(provider, symbol, interval)`` → el más reciente o None.
- Funciona en Windows/macOS/Linux (pathlib; sin caracteres prohibidos en nombres).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mrlab.domain.models import PriceSeries, SeriesMeta


def safe_symbol(symbol: str) -> str:
    raise NotImplementedError


def content_hash(df: pd.DataFrame) -> str:
    raise NotImplementedError


class SnapshotCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def put(self, series: PriceSeries) -> PriceSeries:
        raise NotImplementedError

    def get(self, snapshot_id: str) -> PriceSeries:
        raise NotImplementedError

    def has(self, snapshot_id: str) -> bool:
        raise NotImplementedError

    def list(self, provider: str | None = None, symbol: str | None = None) -> list[SeriesMeta]:
        raise NotImplementedError

    def latest(self, provider: str, symbol: str, interval: str = "1d") -> SeriesMeta | None:
        raise NotImplementedError
