"""API de exportación: /export (ZIP determinista) y /runs/replay (solo caché, sin red)."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pytest
from api_support import pair_config
from fastapi.testclient import TestClient

from mrlab.api.main import create_app
from mrlab.domain.errors import ConfigError
from mrlab.reporting.run_config import parse_run_config

EXPECTED_FILES = [
    "operaciones.csv",
    "diario.csv",
    "pruebas_estadisticas.csv",
    "run_config.json",
    "LEEME.txt",
]


def strict_json(response) -> Any:
    def reject(token: str) -> Any:
        raise AssertionError(f"La respuesta contiene {token}, que no es JSON válido")

    return json.loads(response.text, parse_constant=reject)


def export(client, config: dict, excel_es: bool = False):
    r = client.post("/api/export", json={"config": config, "excel_es": excel_es})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    return r


def members(content: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def run_config_of(content: bytes) -> tuple[str, bytes]:
    name, data = next((k, v) for k, v in members(content).items() if k.endswith("run_config.json"))
    return name, data


def replay(client, name: str, data: bytes, path: str = "/api/runs/replay", **form):
    return client.post(path, files={"file": (name, data, "application/json")}, data=form)


# --------------------------------------------------------------------------- /export


def test_export_is_deterministic_and_named(client) -> None:
    first = export(client, pair_config())
    second = export(client, pair_config())
    assert first.content == second.content  # misma configuración + misma caché = mismos bytes
    assert first.headers["content-disposition"] == 'attachment; filename="KO_PEP.zip"'
    files = members(first.content)
    assert list(files) == EXPECTED_FILES
    readme = files["LEEME.txt"].decode("utf-8")
    assert "SINTÉTICOS" not in readme and "Tipo de datos: HISTORICO" in readme
    assert files["diario.csv"].decode("utf-8").startswith("tipo_datos,")
    cfg = json.loads(files["run_config.json"])
    assert cfg["include_test"] is False and cfg["pair"]["a"]["snapshot_id"]


def test_export_rejects_include_test(client) -> None:
    r = client.post("/api/export", json={"config": pair_config(include_test=True)})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "PRUEBA_SOLO_EN_ESTUDIO" and "estudio" in err["message"]


def test_export_excel_es_format(client) -> None:
    files = members(export(client, pair_config(), excel_es=True).content)
    daily = files["diario.csv"]
    assert daily.startswith("﻿".encode())
    assert daily.decode("utf-8-sig").splitlines()[0].startswith("tipo_datos;")


def test_synthetic_pair_from_generator_is_flagged(client) -> None:
    r = client.post("/api/data/synthetic", json={"scenario": "cointegrado", "seed": 11})
    assert r.status_code == 200, r.text
    gen = r.json()
    refs = {
        leg: {
            "provider": gen[leg]["asset"]["provider"],
            "symbol": gen[leg]["asset"]["symbol"],
            "snapshot_id": gen[leg]["snapshot_id"],
        }
        for leg in ("a", "b")
    }
    resp = export(client, {"pair": refs})
    disposition = resp.headers["content-disposition"]
    assert 'filename="SINTETICO_' in disposition
    files = members(resp.content)
    assert all(name.startswith("SINTETICO_") for name in files)
    readme = next(v for k, v in files.items() if k.endswith("LEEME.txt")).decode("utf-8")
    assert "DATOS SINTÉTICOS" in readme
    daily = next(v for k, v in files.items() if k.endswith("diario.csv")).decode("utf-8")
    assert all(line.startswith("SINTETICO,") for line in daily.splitlines()[1:])


def test_synthetic_preset_provider_is_flagged(client) -> None:
    cfg = pair_config("SINT-COI-A", "SINT-COI-B", provider="synthetic")
    for leg in ("a", "b"):
        cfg["pair"][leg].update(start="2018-01-01", end="2023-06-30")
    r = client.post("/api/pairs/analyze", json=cfg)
    assert r.status_code == 200, r.text
    body = strict_json(r)
    assert body["synthetic"] is True and body["data_origin"] == "SINTETICO"
    resp = export(client, cfg)
    assert resp.headers["content-disposition"].startswith(
        'attachment; filename="SINTETICO_SINT-COI-A_SINT-COI-B'
    )


# --------------------------------------------------------------------------- /runs/replay


def test_replay_reproduces_the_same_zip_without_network(client, fake_yahoo) -> None:
    original = export(client, pair_config()).content
    name, data = run_config_of(original)
    calls = len(fake_yahoo.calls)
    r = replay(client, name, data)
    assert r.status_code == 200, r.text
    assert r.content == original
    assert r.headers["content-disposition"] == 'attachment; filename="KO_PEP.zip"'
    assert len(fake_yahoo.calls) == calls  # nunca descarga
    # También con el formato de Excel en español.
    excel = export(client, pair_config(), excel_es=True).content
    r = replay(client, *run_config_of(excel), excel_es="true")
    assert r.content == excel


def test_replay_json_returns_analysis_and_backtest(client) -> None:
    name, data = run_config_of(export(client, pair_config()).content)
    r = replay(client, name, data, path="/api/runs/replay/json")
    assert r.status_code == 200, r.text
    body = strict_json(r)
    assert body["test_included"] is False and body["test_contaminated"] is None
    assert body["analysis"]["analysis"]["status"] == "evidencia"
    assert list(body["backtest"]["segments"]) == ["formacion", "validacion"]
    assert body["analysis"]["config"]["config_hash"] == json.loads(data)["config_hash"]


def test_replay_with_missing_snapshot_is_spanish_404(client, lab_settings, registry, tmp_path):
    name, data = run_config_of(export(client, pair_config()).content)
    empty = lab_settings.model_copy(
        update={
            "data_dir": tmp_path / "otra",
            "cache_dir": tmp_path / "otra" / "cache",
            "studies_dir": tmp_path / "otra" / "studies",
        }
    )
    with TestClient(create_app(empty, registry), raise_server_exceptions=False) as other:
        r = replay(other, name, data)
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "SNAPSHOT_NO_ENCONTRADO"
    assert "No se encontró la instantánea" in err["message"]


def test_replay_rejects_edited_file(client) -> None:
    name, data = run_config_of(export(client, pair_config()).content)
    cfg = json.loads(data)
    cfg["strategy"]["entry_z"] = 2.5  # editado a mano, con el hash original
    r = replay(client, name, json.dumps(cfg).encode())
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "RUN_CONFIG_HASH_DISTINTO" and "se modificó" in err["message"]


def test_replay_requires_pinned_snapshots(client) -> None:
    r = replay(client, "run_config.json", json.dumps(pair_config()).encode())
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "RUN_CONFIG_SIN_INSTANTANEAS"
    assert "KO (a)" in err["message"] and "nunca vuelve a descargar" in err["message"]


def test_replay_cannot_run_a_new_test_evaluation(client) -> None:
    """Un run_config.json con include_test escrito a mano (sin historial) no ejecuta la prueba."""
    name, data = run_config_of(export(client, pair_config()).content)
    cfg = json.loads(data)
    cfg["include_test"] = True
    cfg.pop("config_hash")
    r = replay(client, name, json.dumps(cfg).encode())
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "PRUEBA_SOLO_EN_ESTUDIO"


def test_replay_of_study_export_keeps_test_and_contamination(client) -> None:
    created = client.post("/api/studies", json={"config": pair_config()}).json()
    study_id = created["study"]["id"]
    client.post(f"/api/studies/{study_id}/test")
    client.put(f"/api/studies/{study_id}/config", json={"strategy": {"entry_z": 2.5}})
    exported = client.get(f"/api/studies/{study_id}/export").content
    name, data = run_config_of(exported)
    assert json.loads(data)["include_test"] is True
    r = replay(client, name, data, path="/api/runs/replay/json")
    assert r.status_code == 200, r.text
    body = strict_json(r)
    assert body["test_included"] is True and body["test_contaminated"] is True
    assert "prueba" in body["backtest"]["segments"]
    assert body["analysis"]["analysis"]["chart_end"] is None  # la prueba ya se vio
    zipped = replay(client, name, data)
    readme = members(zipped.content)["LEEME.txt"].decode("utf-8")
    assert "Prueba contaminada: SÍ" in readme


def test_replay_rejects_invalid_json_in_spanish(client) -> None:
    r = replay(client, "run_config.json", b"{no es json")
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "RUN_CONFIG_INVALIDO" and "no es un JSON válido" in err["message"]
    r = client.post("/api/runs/replay")
    assert r.status_code == 422 and r.json()["error"]["code"] == "PETICION_INVALIDA"


def test_parse_run_config_matches_load_run_config(tmp_path) -> None:
    with pytest.raises(ConfigError) as err:
        parse_run_config(b"[1, 2]", "SINTETICO_run_config.json")
    assert err.value.code == "RUN_CONFIG_INVALIDO"
    assert "SINTETICO_run_config.json" in err.value.message
    assert err.value.details["path"] == "SINTETICO_run_config.json"
    cfg = parse_run_config(json.dumps(pair_config()))
    assert cfg.config_hash and cfg.pair.a.symbol == "KO"


def test_openapi_documents_exports(client) -> None:
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/api/export"]["post"]
    assert "application/zip" in op["responses"]["200"]["content"]
    assert "mismos bytes" in op["description"]
    replay_op = spec["paths"]["/api/runs/replay"]["post"]
    assert "nunca descarga" in replay_op["description"]
