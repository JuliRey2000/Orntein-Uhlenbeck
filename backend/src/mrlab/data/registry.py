"""Registro de proveedores leído de ``providers.toml``.

CONTRATO:
- ``ProviderRegistry.from_toml(path, overrides=None)``: lee ``default`` y las
  secciones ``[providers.<nombre>]`` con ``class = "modulo:Clase"`` y
  ``enabled`` (por defecto true). Las demás claves de la sección se pasan como
  argumentos con nombre al constructor (p. ej. ``api_key_env``). Instancia solo
  los habilitados; ``overrides`` permite inyectar instancias ya construidas
  (p. ej. un YahooProvider con factorías falsas en pruebas) sin importar la
  clase: una instancia inyectada siempre queda registrada. Errores de
  configuración → ConfigError en español.
- ``get(name)`` → proveedor o ConfigError(code="PROVEEDOR_DESCONOCIDO") con la
  lista de disponibles. ``names()``, ``default_name``, ``register(name, provider)``.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Any

from mrlab.data.providers.base import DataProvider
from mrlab.domain.errors import ConfigError

_RESERVED_KEYS = {"class", "enabled"}


def _load_class(name: str, spec: Any, path: Path) -> type:
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ConfigError(
            f'En {path.name}, el proveedor «{name}» debe indicar class = "modulo:Clase" '
            f'(p. ej. "mrlab.data.providers.yahoo:YahooProvider").',
            code="CONFIG_PROVEEDORES_INVALIDA",
            details={"proveedor": name, "class": spec},
        )
    module_name, class_name = spec.split(":")
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(
            f"No se pudo cargar la clase «{spec}» del proveedor «{name}» ({path.name}). "
            "Revisa que el módulo exista y que el nombre de la clase esté bien escrito.",
            code="CONFIG_PROVEEDORES_INVALIDA",
            details={"proveedor": name, "class": spec, "error": str(exc)},
        ) from exc


class ProviderRegistry:
    def __init__(
        self, providers: dict[str, DataProvider] | None = None, default_name: str | None = None
    ):
        self._providers: dict[str, DataProvider] = {}
        for name, provider in (providers or {}).items():
            self.register(name, provider)
        self.default_name = default_name

    @classmethod
    def from_toml(
        cls, path: Path | str, overrides: dict[str, DataProvider] | None = None
    ) -> ProviderRegistry:
        path = Path(path)
        overrides = dict(overrides or {})
        try:
            config = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(
                f"No se encontró el archivo de proveedores {path}.",
                code="CONFIG_PROVEEDORES_NO_ENCONTRADA",
                details={"path": str(path)},
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(
                f"El archivo de proveedores {path.name} no es un TOML válido: {exc}.",
                code="CONFIG_PROVEEDORES_INVALIDA",
                details={"path": str(path)},
            ) from exc

        sections = config.get("providers", {})
        if not isinstance(sections, dict):
            raise ConfigError(
                f"En {path.name}, «providers» debe ser una tabla [providers.<nombre>].",
                code="CONFIG_PROVEEDORES_INVALIDA",
            )
        registry = cls()
        for name, section in sections.items():
            if not isinstance(section, dict):
                raise ConfigError(
                    f"En {path.name}, [providers.{name}] debe ser una tabla.",
                    code="CONFIG_PROVEEDORES_INVALIDA",
                    details={"proveedor": name},
                )
            enabled = section.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ConfigError(
                    f"En {path.name}, «enabled» del proveedor «{name}» debe ser true o false.",
                    code="CONFIG_PROVEEDORES_INVALIDA",
                    details={"proveedor": name, "enabled": enabled},
                )
            if name in overrides:
                registry.register(name, overrides.pop(name))
                continue
            if not enabled:
                continue
            provider_cls = _load_class(name, section.get("class"), path)
            kwargs = {k: v for k, v in section.items() if k not in _RESERVED_KEYS}
            try:
                instance = provider_cls(**kwargs)
            except TypeError as exc:
                raise ConfigError(
                    f"No se pudo crear el proveedor «{name}» con las opciones de {path.name}: {exc}.",
                    code="CONFIG_PROVEEDORES_INVALIDA",
                    details={"proveedor": name, "opciones": sorted(kwargs)},
                ) from exc
            registry.register(name, instance)
        for name, provider in overrides.items():
            registry.register(name, provider)

        default = config.get("default")
        if default is not None and default not in registry._providers:
            raise ConfigError(
                f"El proveedor por defecto «{default}» de {path.name} no está habilitado. "
                f"Disponibles: {', '.join(registry.names()) or 'ninguno'}.",
                code="CONFIG_PROVEEDORES_INVALIDA",
                details={"default": default, "disponibles": registry.names()},
            )
        registry.default_name = default or next(iter(registry._providers), None)
        return registry

    def register(self, name: str, provider: DataProvider) -> None:
        if not isinstance(provider, DataProvider):
            raise ConfigError(
                f"El objeto registrado como «{name}» no implementa la interfaz de proveedor "
                "(name, description, search, get_history, get_metadata).",
                code="PROVEEDOR_INVALIDO",
                details={"proveedor": name, "tipo": type(provider).__name__},
            )
        self._providers[name] = provider

    def get(self, name: str) -> DataProvider:
        try:
            return self._providers[name]
        except KeyError:
            available = self.names()
            raise ConfigError(
                f"Proveedor de datos desconocido: «{name}». Disponibles: "
                f"{', '.join(available) or 'ninguno'}.",
                code="PROVEEDOR_DESCONOCIDO",
                details={"proveedor": name, "disponibles": available},
            ) from None

    def names(self) -> list[str]:
        return list(self._providers)
