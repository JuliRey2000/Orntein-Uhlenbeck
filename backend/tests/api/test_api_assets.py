"""API de activos: búsqueda y metadatos (índices con proxy SUGERIDO, nunca sustituido)."""

from __future__ import annotations

import pytest
from api_support import FakeYahoo


def test_search_default_provider(client, fake_yahoo: FakeYahoo):
    r = client.get("/api/assets/search", params={"q": "ko"})
    assert r.status_code == 200
    symbols = [a["symbol"] for a in r.json()]
    assert "KO" in symbols
    assert fake_yahoo.calls == []  # buscar no descarga precios


def test_search_marks_indices_as_not_executable(client):
    r = client.get("/api/assets/search", params={"q": "S&P", "provider": "yahoo"})
    assert r.status_code == 200
    by_symbol = {a["symbol"]: a for a in r.json()}
    assert by_symbol["^GSPC"]["executable"] is False
    assert by_symbol["^GSPC"]["suggested_proxy"] == "SPY"
    assert by_symbol["SPY"]["executable"] is True


def test_search_synthetic_provider(client):
    r = client.get("/api/assets/search", params={"q": "SINT-COI", "provider": "synthetic"})
    assert r.status_code == 200
    assets = r.json()
    assert {a["symbol"] for a in assets} == {"SINT-COI-A", "SINT-COI-B"}
    assert all(a["origin"] == "SINTETICO" for a in assets)
    assert all(a["name"].startswith("SINTÉTICO") for a in assets)


@pytest.mark.parametrize(
    ("params", "field"),
    [({}, "q"), ({"q": ""}, "q"), ({"q": "   "}, "q"), ({"q": "x" * 101}, "q")],
)
def test_search_rejects_invalid_query(client, params, field):
    r = client.get("/api/assets/search", params=params)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "PETICION_INVALIDA"
    assert err["details"]["campos"][0]["campo"] == field


def test_search_unknown_provider_is_404(client):
    r = client.get("/api/assets/search", params={"q": "KO", "provider": "tiingo"})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "PROVEEDOR_DESCONOCIDO"
    assert "tiingo" in err["message"]


def test_metadata_of_index_suggests_proxy_without_substituting(client, fake_yahoo: FakeYahoo):
    r = client.get("/api/assets/metadata", params={"symbol": "^GSPC"})
    assert r.status_code == 200
    body = r.json()
    assert body["asset"]["symbol"] == "^GSPC"  # el símbolo pedido, no el proxy
    assert body["asset"]["executable"] is False
    assert body["requires_proxy"] is True
    assert body["proxy"]["proxy"] == "SPY"
    assert "no se aplica automáticamente" in body["note"]
    assert body["synthetic"] is False
    assert fake_yahoo.calls == []


def test_metadata_of_executable_asset(client):
    r = client.get("/api/assets/metadata", params={"symbol": "KO", "provider": "yahoo"})
    assert r.status_code == 200
    body = r.json()
    assert body["requires_proxy"] is False
    assert body["proxy"] is None
    assert "ejecutable" in body["note"]


def test_metadata_of_crypto_explains_shorts(client):
    body = client.get("/api/assets/metadata", params={"symbol": "BTC-USD"}).json()
    assert body["requires_proxy"] is False
    assert "corto" in body["note"]


def test_metadata_of_synthetic_asset(client):
    body = client.get(
        "/api/assets/metadata", params={"symbol": "SINT-COI-A", "provider": "synthetic"}
    ).json()
    assert body["synthetic"] is True
    assert "SINTÉTICA" in body["note"]


def test_metadata_path_form_accepts_encoded_symbols(client):
    r = client.get("/api/assets/yahoo/%5EGSPC")
    assert r.status_code == 200
    body = r.json()
    assert body["asset"]["symbol"] == "^GSPC"
    assert body["requires_proxy"] is True
    assert body["proxy"]["proxy"] == "SPY"
    assert client.get("/api/assets/synthetic/SINT-COI-B").json()["synthetic"] is True


@pytest.mark.parametrize(
    ("symbol", "status", "code"),
    [
        ("NOPE", 404, "SIMBOLO_SIN_DATOS"),
        ("FALLA_RED", 503, "ERROR_RED"),
        ("LIMITE", 429, "LIMITE_PETICIONES"),
    ],
)
def test_metadata_provider_errors(client, symbol, status, code):
    r = client.get("/api/assets/metadata", params={"symbol": symbol})
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    if status == 429:
        assert int(r.headers["retry-after"]) > 0


def test_metadata_requires_symbol(client):
    r = client.get("/api/assets/metadata", params={"symbol": "  "})
    assert r.status_code == 422
    assert "símbolo" in r.json()["error"]["message"]
    r = client.get("/api/assets/metadata")
    assert r.status_code == 422
    assert r.json()["error"]["details"]["campos"][0]["campo"] == "symbol"
