"""Humo: la app arranca con el estado inyectado y los errores salen en español."""

from __future__ import annotations


def test_unknown_route_is_spanish_json(client):
    r = client.get("/api/no-existe")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "RUTA_NO_ENCONTRADA"
    assert "no existe" in body["error"]["message"]


def test_openapi_is_published(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"].startswith("Laboratorio de Reversión a la Media")
    assert client.get("/docs").status_code == 200
