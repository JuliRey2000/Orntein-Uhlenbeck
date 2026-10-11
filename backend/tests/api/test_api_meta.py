"""API meta: salud, proveedores (sin secretos), ejemplos, proxies, glosario, CORS y /docs."""

from __future__ import annotations

from datetime import date

import pytest
from api_support import FakeYahoo
from fastapi.testclient import TestClient

from mrlab import DISCLAIMER
from mrlab.api.main import create_app
from mrlab.api.routers.meta import HIDDEN, safe_description
from mrlab.data.presets import PRESET_PAIRS
from mrlab.data.providers.csv_provider import CSVProvider
from mrlab.data.proxies import PROXY_SUGGESTIONS
from mrlab.data.registry import ProviderRegistry
from mrlab.domain.errors import ProviderError

SECRET = "sk-test-0123456789abcdef"
ENV_SECRET = "env-secreto-9876543210"


class ToyProvider:
    """Proveedor de juguete con una clave de API (como lo sería Tiingo o Polygon)."""

    name = "juguete"
    api_key_env = "MRLAB_JUGUETE_API_KEY"

    def __init__(self) -> None:
        self.api_key = SECRET
        self.token = "tok-" + "x" * 20
        # Una descripción descuidada que incluye la clave y el valor de la variable de entorno.
        self.description = (
            f"Proveedor de juguete (clave {SECRET}, entorno {ENV_SECRET}, token {self.token})."
        )

    def search(self, query: str):
        return []

    def get_metadata(self, symbol: str):
        raise ProviderError("sin datos", code="SIMBOLO_SIN_DATOS")

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d"):
        raise ProviderError("sin datos", code="SIMBOLO_SIN_DATOS")


# --------------------------------------------------------------------------- health


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["code_version"]
    assert body["disclaimer"] == DISCLAIMER


# --------------------------------------------------------------------------- proveedores


def test_providers_lists_registry_with_spanish_notices(client):
    r = client.get("/api/providers")
    assert r.status_code == 200
    providers = {p["name"]: p for p in r.json()}
    assert set(providers) == {"yahoo", "csv", "synthetic"}
    assert providers["yahoo"]["is_default"] is True
    assert providers["yahoo"]["kind"] == "en_linea"
    assert any("no oficial" in n for n in providers["yahoo"]["notices"])
    assert providers["csv"]["kind"] == "archivo"
    assert providers["synthetic"]["kind"] == "sintetico"
    assert providers["synthetic"]["synthetic"] is True
    assert all(not p["synthetic"] for n, p in providers.items() if n != "synthetic")
    assert providers["synthetic"]["label"] == "Datos SINTÉTICOS"
    for p in providers.values():
        assert set(p) == {
            "name",
            "label",
            "description",
            "kind",
            "is_default",
            "synthetic",
            "notices",
        }


def test_providers_never_expose_api_keys_or_environment(lab_settings, fake_yahoo, monkeypatch):
    monkeypatch.setenv("MRLAB_JUGUETE_API_KEY", ENV_SECRET)
    toy = ToyProvider()
    registry = ProviderRegistry.from_toml(
        lab_settings.providers_file,
        overrides={
            "yahoo": fake_yahoo,
            "csv": CSVProvider(lab_settings.uploads_dir),
            "juguete": toy,
        },
    )
    with TestClient(create_app(lab_settings, registry)) as c:
        r = c.get("/api/providers")
    assert r.status_code == 200
    text = r.text
    assert SECRET not in text
    assert ENV_SECRET not in text
    assert toy.token not in text
    assert "api_key" not in text
    toy_out = next(p for p in r.json() if p["name"] == "juguete")
    assert HIDDEN in toy_out["description"]
    assert toy_out["kind"] == "en_linea"
    assert toy_out["notices"]  # aviso genérico en español para una fuente externa


def test_safe_description_hides_secrets_and_truncates(monkeypatch):
    monkeypatch.setenv("OTRO_SECRET", "valor-de-entorno-123")

    class P:
        api_key = "clave-de-clase-123456"
        description = "a clave-de-clase-123456 b valor-de-entorno-123 " + "x" * 600

    out = safe_description(P())
    assert "clave-de-clase-123456" not in out
    assert "valor-de-entorno-123" not in out
    assert out.count(HIDDEN) == 2
    assert len(out) <= 500
    assert out.endswith("…")


def test_safe_description_handles_missing_description():
    class P:
        pass

    assert safe_description(P()) == ""


# --------------------------------------------------------------------------- ejemplos y proxies


def test_examples_flag_synthetic_and_availability(client):
    r = client.get("/api/examples")
    assert r.status_code == 200
    examples = r.json()
    assert [e["key"] for e in examples] == [p.key for p in PRESET_PAIRS]
    by_key = {e["key"]: e for e in examples}
    assert by_key["sint_coi"]["synthetic"] is True
    assert by_key["ko_pep"]["synthetic"] is False
    assert all(e["available"] for e in examples)
    # Ninguna descripción promete un resultado.
    for e in examples:
        text = (e["title"] + " " + e["description"]).lower()
        for banned in ("rentable", "ganadora", "garantizad"):
            assert banned not in text


def test_examples_unavailable_when_provider_disabled(lab_settings, fake_yahoo):
    registry = ProviderRegistry({"yahoo": fake_yahoo}, default_name="yahoo")
    with TestClient(create_app(lab_settings, registry)) as c:
        examples = {e["key"]: e for e in c.get("/api/examples").json()}
    assert examples["ko_pep"]["available"] is True
    assert examples["sint_coi"]["available"] is False


def test_proxies_are_suggestions(client):
    r = client.get("/api/proxies")
    assert r.status_code == 200
    proxies = {p["reference"]: p for p in r.json()}
    assert set(proxies) == set(PROXY_SUGGESTIONS)
    assert proxies["^GSPC"]["proxy"] == "SPY"
    assert all(p["note"] for p in proxies.values())


def test_glossary_has_spanish_texts(client):
    r = client.get("/api/glossary")
    assert r.status_code == 200
    body = r.json()
    assert body["disclaimer"] == DISCLAIMER
    assert len(body["concepts"]) == 3
    assert "nunca prueba la cointegración" in body["correlation_note"]
    assert body["status_labels"]
    assert body["metric_help"]
    assert "t+1" in body["execution_convention"]
    assert "SINTÉTICOS" in body["synthetic_note"]


# --------------------------------------------------------------------------- CORS


def test_cors_allows_configured_origin(client):
    r = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


@pytest.mark.parametrize("origin", ["http://evil.example", "http://localhost:8080", "null"])
def test_cors_rejects_other_origins(client, origin):
    r = client.options(
        "/api/health", headers={"Origin": origin, "Access-Control-Request-Method": "GET"}
    )
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers
    r = client.get("/api/health", headers={"Origin": origin})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_cors_origin_list_comes_from_settings(lab_settings, registry):
    settings = lab_settings.model_copy(
        update={"cors_origins": " http://a.example , http://b.example ,"}
    )
    with TestClient(create_app(settings, registry)) as c:
        for origin in ("http://a.example", "http://b.example"):
            r = c.get("/api/health", headers={"Origin": origin})
            assert r.headers["access-control-allow-origin"] == origin
        r = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" not in r.headers


# --------------------------------------------------------------------------- documentación

#: Rutas de este grupo de routers (meta, activos, datos, simulación y trabajos).
OWN_PATHS = (
    "/api/health",
    "/api/providers",
    "/api/examples",
    "/api/proxies",
    "/api/glossary",
    "/api/assets/search",
    "/api/assets/metadata",
    "/api/assets/{provider}/{symbol}",
    "/api/data/load",
    "/api/data/csv",
    "/api/data/synthetic",
    "/api/data/cache",
    "/api/data/series",
    "/api/data/validate",
    "/api/ou/simulate",
    "/api/jobs",
    "/api/jobs/{job_id}",
)
ENGLISH_DEFAULTS = ("Successful Response", "Validation Error")


def test_openapi_documents_own_endpoints_in_spanish(client):
    spec = client.get("/openapi.json").json()
    for path in OWN_PATHS:
        assert path in spec["paths"], path
        for method, op in spec["paths"][path].items():
            where = f"{method.upper()} {path}"
            assert op.get("summary"), where
            assert op.get("description"), where
            assert op.get("tags"), where
            for status, resp in op["responses"].items():
                assert resp["description"] not in ENGLISH_DEFAULTS, (where, status)
                ref = str(resp.get("content", {}))
                assert "HTTPValidationError" not in ref, (where, status)
            for param in op.get("parameters", []):
                assert param.get("description"), (where, param["name"])


def test_openapi_request_and_response_models_have_descriptions(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for name in (
        "LoadRequest",
        "LoadResponse",
        "SyntheticRequest",
        "SyntheticOut",
        "SimulateRequest",
        "SimulationOut",
        "ValidateRequest",
        "ValidationOut",
        "SeriesMetaOut",
        "AssetInfoOut",
        "AssetMetadataOut",
        "ProviderOut",
        "HealthOut",
        "JobOut",
    ):
        props = schemas[name]["properties"]
        missing = [
            p for p, spec in props.items() if not spec.get("description") and "$ref" not in spec
        ]
        assert not missing, (name, missing)


def test_docs_page_is_served(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower()
    spec = client.get("/openapi.json").json()
    assert DISCLAIMER in spec["info"]["description"]


def test_fake_yahoo_is_used_without_network(client, fake_yahoo: FakeYahoo):
    client.get("/api/health")
    client.get("/api/providers")
    assert fake_yahoo.calls == []
