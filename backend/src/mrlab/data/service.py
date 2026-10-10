"""Servicio de datos: proveedor + caché agresiva.

CONTRATO (a implementar):
- ``load(provider, symbol, start, end, interval="1d", refresh=False)``:
  si existe una instantánea cuyo rango pedido (requested_start/end) cubre
  [start, end] y no se pide ``refresh``, se usa (sin red). Si no, se descarga
  el rango pedido, se guarda con ``cache.put`` (requested_start/end fijados) y
  se usa. La serie devuelta se recorta a [start, end] inclusive, conserva el
  ``snapshot_id`` de la instantánea completa y actualiza coverage/n_obs.
- ``load_snapshot(snapshot_id, start=None, end=None)``: solo caché (para
  reproducir un run_config); nunca descarga.
- Los proveedores sintéticos también pasan por la caché (son deterministas:
  mismo contenido → mismo id).
"""

from __future__ import annotations

from datetime import date

from mrlab.data.cache import SnapshotCache
from mrlab.data.registry import ProviderRegistry
from mrlab.domain.models import PriceSeries


class DataService:
    def __init__(self, registry: ProviderRegistry, cache: SnapshotCache):
        self.registry = registry
        self.cache = cache

    def load(
        self,
        provider: str,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
        refresh: bool = False,
    ) -> PriceSeries:
        raise NotImplementedError

    def load_snapshot(self, snapshot_id: str, start: date | None = None, end: date | None = None) -> PriceSeries:
        raise NotImplementedError
