"""Interfaz abstracta de proveedores de datos (sección 2.1).

Añadir un proveedor = implementar esta interfaz + registrarlo en ``providers.toml``.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from mrlab.domain.errors import ProviderError
from mrlab.domain.models import AssetInfo, PriceSeries

SUPPORTED_INTERVALS = ("1d",)  # v1: solo diario (sección 2.5)


@runtime_checkable
class DataProvider(Protocol):
    name: str
    description: str  # en español; se muestra en la interfaz

    def search(self, query: str) -> list[AssetInfo]: ...

    def get_history(
        self, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> PriceSeries: ...

    def get_metadata(self, symbol: str) -> AssetInfo: ...


def check_interval(interval: str) -> None:
    if interval not in SUPPORTED_INTERVALS:
        raise ProviderError(
            f"Frecuencia «{interval}» no soportada en la v1. Solo se admiten datos diarios (1d).",
            code="INTERVALO_NO_SOPORTADO",
            details={"interval": interval, "soportados": list(SUPPORTED_INTERVALS)},
        )
