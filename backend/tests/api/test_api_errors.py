"""Formato de errores: todo error sale como {"error": {code, message, details}} en español.

Nunca aparece el texto de pydantic, FastAPI o Starlette en inglés ni una traza."""

from __future__ import annotations

import math
from datetime import date

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from mrlab.api.errors import (
    RETRY_AFTER_SECONDS,
    format_number,
    json_safe,
    lab_error_status,
    validation_summary,
)
from mrlab.domain.errors import (
    AnalysisBlocked,
    ConfigError,
    DataError,
    LabError,
    ModelError,
    ProviderError,
)

#: Fragmentos típicos de los mensajes en inglés de pydantic, FastAPI y Starlette.
ENGLISH_MARKERS = (
    "Input should",
    "Field required",
    "field required",
    "Extra inputs",
    "JSON decode",
    "should be",
    "should have",
    "is not a valid",
    "Value error",
    "value is not",
    "There was an error",
    "Not Found",
    "Method Not Allowed",
    "Internal Server Error",
    "Unprocessable",
    "Traceback",
    "Expecting value",
    "Invalid",
    "invalid",
    "Unable to",
)
J = {"content-type": "application/json"}


def assert_error_format(r, status: int, code: str | None = None) -> dict:
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert set(body) == {"error"}
    err = body["error"]
    assert set(err) == {"code", "message", "details"}
    assert isinstance(err["details"], dict)
    assert err["code"] and err["code"] == err["code"].upper()
    if code is not None:
        assert err["code"] == code
    texts = [err["message"]]
    texts += [c["motivo"] for c in err["details"].get("campos", [])]
    for text in texts:
        for marker in ENGLISH_MARKERS:
            assert marker not in text, (marker, text)
    return err


# --------------------------------------------------------------------------- HTTP


def test_unknown_route_404(client):
    err = assert_error_format(client.get("/api/no-existe"), 404, "RUTA_NO_ENCONTRADA")
    assert "no existe" in err["message"]
    assert_error_format(client.get("/fuera-de-la-api"), 404, "RUTA_NO_ENCONTRADA")


def test_method_not_allowed_405(client):
    r = client.delete("/api/health")
    err = assert_error_format(r, 405, "METODO_NO_PERMITIDO")
    assert "no permitido" in err["message"]
    assert "GET" in r.headers["allow"]
    assert_error_format(client.put("/api/data/load"), 405, "METODO_NO_PERMITIDO")


# --------------------------------------------------------------------------- 422 de validación

INVALID_BODIES = [
    # (descripción, ruta, kwargs, campos esperados)
    (
        "tipo incorrecto",
        "/api/ou/simulate",
        {"json": {"lam": "rápido", "sigma": [1], "x0": {"a": 1}}},
        {"lam", "sigma", "x0"},
    ),
    (
        "entero con decimales",
        "/api/ou/simulate",
        {"json": {"lam": 1, "sigma": 1, "x0": 1, "horizon": 10.5}},
        {"horizon"},
    ),
    (
        "enum inválido",
        "/api/data/synthetic",
        {"json": {"scenario": "aleatorio"}},
        {"scenario"},
    ),
    (
        "enum inválido anidado",
        "/api/data/validate",
        {
            "json": {
                "a": {"provider": "yahoo", "symbol": "KO", "snapshot_id": "x"},
                "b": {"provider": "yahoo", "symbol": "PEP", "snapshot_id": "y"},
                "spread_type": "cuadratico",
            }
        },
        {"spread_type"},
    ),
    (
        "fecha inválida",
        "/api/data/load",
        {"json": {"symbols": ["KO"], "start": "2020-02-30", "end": "31/12/2020"}},
        {"start", "end"},
    ),
    (
        "fecha como número",
        "/api/data/load",
        {"json": {"symbols": ["KO"], "start": 20200101, "end": "2020-12-31"}},
        {"start"},
    ),
    (
        "fecha con hora",
        "/api/data/load",
        {"json": {"symbols": ["KO"], "start": "2020-01-01T10:30:00", "end": "2020-12-31"}},
        {"start"},
    ),
    (
        "fecha anidada",
        "/api/data/validate",
        {
            "json": {
                "a": {"provider": "yahoo", "symbol": "KO", "start": "ayer", "end": "2020-01-01"},
                "b": {"provider": "yahoo", "symbol": "PEP", "snapshot_id": "y"},
            }
        },
        {"a.start"},
    ),
    (
        "JSON mal formado",
        "/api/data/load",
        {"content": b'{"symbols": ["KO",', "headers": J},
        {"(cuerpo de la petición)"},
    ),
    (
        "JSON con comillas simples",
        "/api/ou/simulate",
        {"content": b"{'lam': 1}", "headers": J},
        {"(cuerpo de la petición)"},
    ),
    (
        "campo faltante",
        "/api/ou/simulate",
        {"json": {"lam": 0.1, "sigma": 0.02}},
        {"x0"},
    ),
    (
        "cuerpo vacío",
        "/api/data/load",
        {"content": b"", "headers": J},
        {"(cuerpo de la petición)"},
    ),
    (
        "cuerpo null",
        "/api/data/load",
        {"content": b"null", "headers": J},
        {"(cuerpo de la petición)"},
    ),
    (
        "cuerpo lista",
        "/api/data/load",
        {"json": [1, 2, 3]},
        {"(cuerpo de la petición)"},
    ),
    (
        "sin Content-Type JSON",
        "/api/data/load",
        {
            "content": b'{"symbols": ["KO"], "start": "2020-01-01", "end": "2020-12-31"}',
            "headers": {"content-type": "text/plain"},
        },
        {"(cuerpo de la petición)"},
    ),
    (
        "campo no permitido",
        "/api/data/load",
        {"json": {"symbols": ["KO"], "start": "2020-01-01", "end": "2020-12-31", "x": 1}},
        {"x"},
    ),
    (
        "lista vacía y booleano inválido",
        "/api/data/load",
        {"json": {"symbols": [], "start": "2020-01-01", "end": "2020-12-31", "refresh": "sí"}},
        {"symbols", "refresh"},
    ),
    (
        "elementos de lista de otro tipo",
        "/api/data/load",
        {"json": {"symbols": [1, None], "start": "2020-01-01", "end": "2020-12-31"}},
        {"symbols.0", "symbols.1"},
    ),
    (
        "número no finito",
        "/api/data/synthetic",
        {"content": b'{"alpha": NaN, "beta": -Infinity}', "headers": J},
        {"alpha", "beta"},
    ),
    (
        "fuera de rango",
        "/api/ou/simulate",
        {"json": {"lam": -1, "sigma": 0.1, "x0": 1, "n_paths": 9999}},
        {"lam", "n_paths"},
    ),
    (
        "parámetro de consulta inválido",
        "/api/data/series?snapshot_id=x&start=mañana",
        None,
        {"start"},
    ),
    (
        "parámetro de consulta faltante",
        "/api/assets/search",
        None,
        {"q"},
    ),
    (
        "formulario incompleto",
        "/api/data/csv",
        {"data": {"symbol": "KO"}, "files": {"file": ("a.csv", b"date,close\n")}},
        {"currency", "asset_class", "price_type"},
    ),
]


@pytest.mark.parametrize(
    ("label", "path", "kwargs", "fields"), INVALID_BODIES, ids=[c[0] for c in INVALID_BODIES]
)
def test_validation_errors_are_spanish_with_field_paths(client, label, path, kwargs, fields):
    r = client.get(path) if kwargs is None else client.post(path, **kwargs)
    err = assert_error_format(r, 422, "PETICION_INVALIDA")
    assert err["message"].startswith("La petición tiene datos no válidos.")
    campos = err["details"]["campos"]
    assert fields <= {c["campo"] for c in campos}, campos
    for c in campos:
        assert set(c) == {"campo", "motivo", "recibido"}
        assert c["motivo"]
        assert c["campo"] in err["message"]


def test_specific_spanish_reasons(client):
    def reasons(**kwargs):
        r = client.post("/api/data/load", **kwargs)
        return {c["campo"]: c["motivo"] for c in r.json()["error"]["details"]["campos"]}

    got = reasons(json={"symbols": ["KO"], "start": "2020-02-30", "end": 5})
    assert got["start"] == "fecha inválida: usa el formato AAAA-MM-DD"
    assert got["end"] == "fecha inválida: usa el formato AAAA-MM-DD (como texto)"
    got = reasons(content=b"", headers=J)
    assert got["(cuerpo de la petición)"].startswith("falta el cuerpo de la petición")
    got = reasons(content=b'{"a": 1', headers=J)
    assert got["(cuerpo de la petición)"].startswith("el cuerpo no es un JSON válido")
    got = reasons(content=b"{}", headers={"content-type": "text/plain"})
    assert "Content-Type: application/json" in got["(cuerpo de la petición)"]
    r = client.post("/api/data/synthetic", json={"scenario": "x", "rho": 3})
    got = {c["campo"]: c["motivo"] for c in r.json()["error"]["details"]["campos"]}
    assert got["scenario"].startswith("opción no permitida; valores válidos: 'cointegrado'")
    assert " o 'independientes'" in got["scenario"]
    assert got["rho"] == "debe ser menor o igual que 1"


def test_received_value_is_short_and_never_the_raw_body(client):
    r = client.post(
        "/api/data/load",
        content=b'{"symbols": ["KO"], "start": "2020-01-01", "end": "2020-12-31"}',
        headers={"content-type": "text/plain"},
    )
    campos = r.json()["error"]["details"]["campos"]
    assert campos[0]["recibido"] is None
    r = client.post(
        "/api/data/load", json={"symbols": [None], "start": "x" * 200, "end": "2020-01-01"}
    )
    campos = {c["campo"]: c["recibido"] for c in r.json()["error"]["details"]["campos"]}
    assert campos["symbols.0"] is None
    assert len(campos["start"]) <= 60


def test_many_errors_are_summarized(client):
    r = client.post(
        "/api/data/synthetic",
        json={
            "scenario": "x",
            "n_obs": "x",
            "seed": "x",
            "alpha": "x",
            "beta": "x",
            "ou_lambda": "x",
            "ou_mu": "x",
            "ou_sigma": "x",
            "b_drift": "x",
            "b_vol": "x",
        },
    )
    err = assert_error_format(r, 422, "PETICION_INVALIDA")
    assert len(err["details"]["campos"]) == 10
    assert err["message"].endswith("y 2 problema(s) más.")


# --------------------------------------------------------------------------- LabError, HTTP y 500


@pytest.fixture
def raising_client(app):
    """Cliente con rutas de prueba que lanzan errores (sin red, sin estado)."""

    def raiser(exc: BaseException):
        def endpoint():
            raise exc

        return endpoint

    cases = {
        "dato": DataError("Datos inválidos de prueba.", code="DATOS_INVALIDOS"),
        "config": ConfigError("Configuración inválida de prueba."),
        "bloqueado": AnalysisBlocked("Análisis bloqueado de prueba."),
        "modelo": ModelError("Modelo no válido de prueba."),
        "proveedor": ProviderError("Fallo del proveedor de prueba."),
        "limite": ProviderError("Límite de prueba.", code="LIMITE_PETICIONES"),
        "red": ProviderError("Sin red de prueba.", code="ERROR_RED"),
        "noexiste": DataError("No existe de prueba.", code="SNAPSHOT_NO_ENCONTRADO"),
        "base": LabError("Error base de prueba."),
        "detalles": DataError(
            "Con detalles.",
            code="CON_DETALLES",
            details={"x": float("nan"), "d": date(2020, 1, 2), "n": [math.inf, 1]},
        ),
        "http_ingles": HTTPException(status_code=403, detail="Forbidden: you shall not pass"),
        "http_dict": HTTPException(
            status_code=409, detail={"code": "CONFLICTO_PRUEBA", "message": "Conflicto de prueba."}
        ),
        "http_413": StarletteHTTPException(status_code=413, detail="Request Entity Too Large"),
        "http_raro": HTTPException(status_code=418, detail="I'm a teapot"),
        "interno": RuntimeError("detalle interno: contraseña=1234 en /srv/app.py"),
        "clave": KeyError("api_key"),
    }
    for name, exc in cases.items():
        app.add_api_route(f"/api/_prueba/{name}", raiser(exc), methods=["GET"])
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.mark.parametrize(
    ("name", "status", "code"),
    [
        ("dato", 422, "DATOS_INVALIDOS"),
        ("config", 400, "CONFIGURACION_INVALIDA"),
        ("bloqueado", 422, "ANALISIS_BLOQUEADO"),
        ("modelo", 422, "MODELO_NO_VALIDO"),
        ("proveedor", 502, "ERROR_PROVEEDOR"),
        ("limite", 429, "LIMITE_PETICIONES"),
        ("red", 503, "ERROR_RED"),
        ("noexiste", 404, "SNAPSHOT_NO_ENCONTRADO"),
        ("base", 400, "ERROR"),
    ],
)
def test_lab_errors_map_to_http_status(raising_client, name, status, code):
    r = raising_client.get(f"/api/_prueba/{name}")
    err = assert_error_format(r, status, code)
    assert "de prueba" in err["message"]
    if status == 429:
        assert r.headers["retry-after"] == str(RETRY_AFTER_SECONDS)
    else:
        assert "retry-after" not in r.headers


def test_lab_error_details_are_json_safe(raising_client):
    r = raising_client.get("/api/_prueba/detalles")
    assert "NaN" not in r.text and "Infinity" not in r.text
    err = assert_error_format(r, 422, "CON_DETALLES")
    assert err["details"] == {"x": None, "d": "2020-01-02", "n": [None, 1]}


def test_http_exception_with_english_detail_is_translated(raising_client):
    err = assert_error_format(raising_client.get("/api/_prueba/http_ingles"), 403, "PROHIBIDO")
    assert "Forbidden" not in err["message"]
    err = assert_error_format(raising_client.get("/api/_prueba/http_413"), 413)
    assert err["code"] == "ARCHIVO_DEMASIADO_GRANDE"
    err = assert_error_format(raising_client.get("/api/_prueba/http_raro"), 418, "ERROR_HTTP")
    assert "teapot" not in err["message"]


def test_http_exception_with_structured_detail_is_kept(raising_client):
    err = assert_error_format(raising_client.get("/api/_prueba/http_dict"), 409)
    assert err == {"code": "CONFLICTO_PRUEBA", "message": "Conflicto de prueba.", "details": {}}


@pytest.mark.parametrize("name", ["interno", "clave"])
def test_unexpected_errors_are_500_without_trace(raising_client, name, caplog):
    r = raising_client.get(f"/api/_prueba/{name}")
    err = assert_error_format(r, 500, "ERROR_INTERNO")
    assert err["details"] == {}
    for leaked in ("contraseña", "1234", "/srv/app.py", "RuntimeError", "KeyError", "api_key"):
        assert leaked not in r.text
    assert "error interno inesperado" in err["message"]
    # Se registra en el log del servidor (para diagnosticar), no en la respuesta.
    assert any("Error no controlado" in rec.getMessage() for rec in caplog.records)


def test_malformed_multipart_is_400_in_spanish(client):
    r = client.post(
        "/api/data/csv",
        content=b"esto no es multipart",
        headers={"content-type": "multipart/form-data; boundary=zz"},
    )
    err = assert_error_format(r, 400, "PETICION_INVALIDA")
    assert "no se pudo interpretar el cuerpo" in err["message"]


# --------------------------------------------------------------------------- unidades


def test_lab_error_status_table():
    assert lab_error_status(ProviderError("x", code="SIMBOLO_SIN_DATOS")) == 404
    assert lab_error_status(ConfigError("x", code="PROVEEDOR_DESCONOCIDO")) == 404
    assert lab_error_status(ConfigError("x", code="TRABAJO_YA_TERMINADO")) == 400
    assert lab_error_status(ProviderError("x", code="RESPUESTA_INESPERADA")) == 502


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0, "0"),
        (1.0, "1"),
        (0.5, "0,5"),
        (2520, "2.520"),
        (10000, "10.000"),
        (-1234567, "-1.234.567"),
        (2**63 - 1, "9.223.372.036.854.775.807"),
        (1e-5, "1e-05"),
        ("abc", "abc"),
        (True, "True"),
    ],
)
def test_format_number(value, text):
    assert format_number(value) == text


def test_validation_summary_without_errors_of_pydantic_text():
    errors = [
        {"type": "missing", "loc": ("body", "x"), "msg": "Field required", "input": {}},
        {
            "type": "greater_than",
            "loc": ("query", "n"),
            "msg": "Input should be greater than 0",
            "input": -1,
            "ctx": {"gt": 0},
        },
        {"type": "tipo_nuevo_desconocido", "loc": ("body",), "msg": "Some new error", "input": 1},
    ]
    message, items = validation_summary(errors)
    assert "Field required" not in message and "Input should" not in message
    assert "Some new error" not in message
    assert [i["campo"] for i in items] == ["x", "n", "(cuerpo de la petición)"]
    assert items[1]["motivo"] == "debe ser mayor que 0"
    assert items[2]["motivo"] == "valor no válido"


def test_json_safe_handles_numpy_and_non_finite():
    import numpy as np

    out = json_safe({"a": np.float64("nan"), "b": np.int64(3), "c": (1, np.inf), 4: "x"})
    assert out == {"a": None, "b": 3, "c": [1, None], "4": "x"}
