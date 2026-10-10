"""Servicio de datos: proveedor + caché agresiva.

CONTRATO:
- ``load(provider, symbol, start, end, interval="1d", refresh=False)``:
  si existe una instantánea cuyo rango pedido (requested_start/end) cubre
  [start, end] y no se pide ``refresh``, se usa (sin red; la más reciente si hay
  varias). Si no, se descarga el rango pedido, se guarda con ``cache.put``
  (requested_start/end fijados) y se usa. La serie devuelta se recorta a
  [start, end] inclusive, conserva el ``snapshot_id`` de la instantánea completa
  y actualiza coverage/n_obs; ``requested_start/end`` pasan a ser los de ESTA
  petición, de modo que (snapshot_id, requested_start, requested_end) basta para
  reproducirla con ``load_snapshot``.
- ``load_snapshot(snapshot_id, start=None, end=None)``: solo caché (para
  reproducir un run_config); nunca descarga.
- Los proveedores sintéticos también pasan por la caché (son deterministas:
  mismo contenido → mismo id).
- Un proveedor con ``always_reload = True`` (el CSV: el usuario puede volver a
  subir un archivo corregido con el mismo símbolo) se relee siempre; la lectura
  es local y barata, y la caché igualmente guarda cada versión como instantánea
  propia (otro contenido → otro id) para poder reproducirla.
- El recorte conserva el orden y los duplicados crudos (los reporta la
  validación). Un recorte vacío → DataError(code="SIN_DATOS_EN_RANGO").
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from mrlab.data.cache import SnapshotCache
from mrlab.data.providers.base import check_interval
from mrlab.data.registry import ProviderRegistry
from mrlab.domain.errors import DataError
from mrlab.domain.models import PriceSeries, SeriesMeta, coverage_of


def _check_range(start: date | None, end: date | None) -> None:
    if start is not None and end is not None and start > end:
        raise DataError(
            f"Rango de fechas inválido: el inicio ({start.isoformat()}) es posterior al fin "
            f"({end.isoformat()}).",
            code="RANGO_FECHAS_INVALIDO",
            details={"start": start.isoformat(), "end": end.isoformat()},
        )


def _covers(meta: SeriesMeta, start: date, end: date) -> bool:
    return (
        meta.requested_start is not None
        and meta.requested_end is not None
        and meta.requested_start <= start
        and meta.requested_end >= end
    )


def trim_series(series: PriceSeries, start: date | None, end: date | None) -> PriceSeries:
    """Recorta a [start, end] inclusive sin reordenar ni rellenar; actualiza la cobertura."""
    if start is None and end is None:
        return series
    idx = series.data.index
    mask = np.ones(len(idx), dtype=bool)
    if start is not None:
        mask &= idx >= pd.Timestamp(start)
    if end is not None:
        mask &= idx <= pd.Timestamp(end)
    data = series.data[mask]
    if data.empty:
        span = f"{start.isoformat() if start else '…'} a {end.isoformat() if end else '…'}"
        raise DataError(
            f"La serie «{series.symbol}» no tiene observaciones entre {span}. "
            f"Los datos disponibles cubren de {series.meta.coverage_start} a "
            f"{series.meta.coverage_end}. Revisa el rango de fechas.",
            code="SIN_DATOS_EN_RANGO",
            details={"symbol": series.symbol, "snapshot_id": series.meta.snapshot_id},
        )
    cov_start, cov_end, n_obs = coverage_of(data)
    changes: dict = {"coverage_start": cov_start, "coverage_end": cov_end, "n_obs": n_obs}
    if start is not None:
        changes["requested_start"] = start
    if end is not None:
        changes["requested_end"] = end
    return series.with_data(data).with_meta(**changes)


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
        _check_range(start, end)
        check_interval(interval)
        source = self.registry.get(provider)
        reload = refresh or getattr(source, "always_reload", False)
        cached = None if reload else self.find_covering(provider, symbol, start, end, interval)
        if cached is not None:
            full = self.cache.get(cached.snapshot_id)
        else:
            downloaded = source.get_history(symbol, start, end, interval)
            downloaded = downloaded.with_meta(
                requested_start=start, requested_end=end, interval=interval
            )
            full = self.cache.put(downloaded)
        return trim_series(full, start, end)

    def load_snapshot(
        self, snapshot_id: str, start: date | None = None, end: date | None = None
    ) -> PriceSeries:
        _check_range(start, end)
        return trim_series(self.cache.get(snapshot_id), start, end)

    def find_covering(
        self, provider: str, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> SeriesMeta | None:
        """Instantánea más reciente cuyo rango pedido cubre [start, end], o None."""
        return next(
            (
                m
                for m in self.cache.list(provider, symbol)
                if m.interval == interval and m.snapshot_id and _covers(m, start, end)
            ),
            None,
        )
