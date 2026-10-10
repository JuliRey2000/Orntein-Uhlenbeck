"""Registro de proveedores leído de ``providers.toml``.

CONTRATO (a implementar):
- ``ProviderRegistry.from_toml(path, overrides=None)``: lee ``default`` y las
  secciones ``[providers.<nombre>]`` con ``class = "modulo:Clase"`` y
  ``enabled``. Instancia solo los habilitados; ``overrides`` permite inyectar
  instancias ya construidas (p. ej. un YahooProvider con factorías falsas en
  pruebas) sin importar la clase. Errores de configuración → ConfigError en español.
- ``get(name)`` → proveedor o ConfigError(code="PROVEEDOR_DESCONOCIDO") con la
  lista de disponibles. ``names()``, ``default_name``, ``register(name, provider)``.
"""

from __future__ import annotations

from pathlib import Path

from mrlab.data.providers.base import DataProvider


class ProviderRegistry:
    def __init__(self, providers: dict[str, DataProvider] | None = None, default_name: str | None = None):
        self._providers: dict[str, DataProvider] = dict(providers or {})
        self.default_name = default_name

    @classmethod
    def from_toml(cls, path: Path | str, overrides: dict[str, DataProvider] | None = None) -> ProviderRegistry:
        raise NotImplementedError

    def register(self, name: str, provider: DataProvider) -> None:
        raise NotImplementedError

    def get(self, name: str) -> DataProvider:
        raise NotImplementedError

    def names(self) -> list[str]:
        raise NotImplementedError
