"""Registro de proveedores desde providers.toml."""

from __future__ import annotations

import pytest

from mrlab.config import BACKEND_ROOT
from mrlab.data.providers.csv_provider import CSVProvider
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.data.providers.yahoo import YahooProvider
from mrlab.data.registry import ProviderRegistry
from mrlab.domain.errors import ConfigError

PROVIDERS_TOML = BACKEND_ROOT / "providers.toml"


def write(tmp_path, text: str):
    path = tmp_path / "providers.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_providers_toml_with_overrides(tmp_path):
    csv = CSVProvider(tmp_path / "uploads")
    fake_yahoo = YahooProvider(ticker_factory=lambda s: None, sleep=lambda s: None)
    registry = ProviderRegistry.from_toml(
        PROVIDERS_TOML, overrides={"csv": csv, "yahoo": fake_yahoo}
    )

    assert registry.names() == ["yahoo", "csv", "synthetic"]
    assert registry.default_name == "yahoo"
    assert registry.get("csv") is csv
    assert registry.get("yahoo") is fake_yahoo
    assert isinstance(registry.get("synthetic"), SyntheticProvider)


def test_real_providers_toml_instantiates_classes():
    registry = ProviderRegistry.from_toml(PROVIDERS_TOML)
    assert isinstance(registry.get("yahoo"), YahooProvider)
    assert isinstance(registry.get("csv"), CSVProvider)


def test_unknown_provider_lists_available():
    registry = ProviderRegistry.from_toml(PROVIDERS_TOML)
    with pytest.raises(ConfigError) as exc_info:
        registry.get("tiingo")
    err = exc_info.value
    assert err.code == "PROVEEDOR_DESCONOCIDO"
    assert "«tiingo»" in err.message
    for name in ("yahoo", "csv", "synthetic"):
        assert name in err.message
    assert err.details["disponibles"] == ["yahoo", "csv", "synthetic"]


def test_disabled_providers_are_not_imported(tmp_path):
    path = write(
        tmp_path,
        """
default = "synthetic"
[providers.synthetic]
class = "mrlab.data.providers.synthetic:SyntheticProvider"
[providers.tiingo]
class = "mrlab.data.providers.tiingo:TiingoProvider"
enabled = false
api_key_env = "TIINGO_API_KEY"
""",
    )
    registry = ProviderRegistry.from_toml(path)
    assert registry.names() == ["synthetic"] and registry.default_name == "synthetic"


def test_extra_keys_are_constructor_options(tmp_path):
    path = write(
        tmp_path,
        """
[providers.yahoo]
class = "mrlab.data.providers.yahoo:YahooProvider"
max_retries = 2
backoff_base = 0.5
""",
    )
    registry = ProviderRegistry.from_toml(path)
    yahoo = registry.get("yahoo")
    assert (yahoo.max_retries, yahoo.backoff_base) == (2, 0.5)
    assert registry.default_name == "yahoo"


def test_override_is_registered_even_if_absent_from_file(tmp_path):
    path = write(
        tmp_path,
        'default = "synthetic"\n[providers.synthetic]\nclass = "mrlab.data.providers.synthetic:SyntheticProvider"\n',
    )
    extra = CSVProvider(tmp_path)
    registry = ProviderRegistry.from_toml(path, overrides={"csv": extra})
    assert registry.get("csv") is extra


@pytest.mark.parametrize(
    "text",
    [
        '[providers.x]\nclass = "mrlab.no_existe:Clase"\n',
        '[providers.x]\nclass = "mrlab.data.providers.yahoo:NoExiste"\n',
        '[providers.x]\nclass = "sin_dos_puntos"\n',
        "[providers.x]\nenabled = true\n",
        '[providers.x]\nclass = "mrlab.data.providers.yahoo:YahooProvider"\nenabled = "si"\n',
        '[providers.x]\nclass = "mrlab.data.providers.yahoo:YahooProvider"\nopcion_rara = 1\n',
        'default = "yahoo"\n[providers.x]\nclass = "mrlab.data.providers.synthetic:SyntheticProvider"\n',
        "providers = 3\n",
        "esto no es [toml",
    ],
)
def test_configuration_errors_are_spanish_config_errors(tmp_path, text):
    with pytest.raises(ConfigError) as exc_info:
        ProviderRegistry.from_toml(write(tmp_path, text))
    assert exc_info.value.code == "CONFIG_PROVEEDORES_INVALIDA"


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError) as exc_info:
        ProviderRegistry.from_toml(tmp_path / "no_existe.toml")
    assert exc_info.value.code == "CONFIG_PROVEEDORES_NO_ENCONTRADA"


def test_register_requires_provider_interface():
    registry = ProviderRegistry()
    registry.register("synthetic", SyntheticProvider())
    assert registry.names() == ["synthetic"]
    with pytest.raises(ConfigError) as exc_info:
        registry.register("roto", object())  # type: ignore[arg-type]
    assert exc_info.value.code == "PROVEEDOR_INVALIDO"
