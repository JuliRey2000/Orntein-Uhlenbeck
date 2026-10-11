"""API de estudios: flujo completo por HTTP, prueba única y prueba contaminada (sin red)."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pytest
from api_support import pair_config

from mrlab import DISCLAIMER
from mrlab.api.main import create_app
from mrlab.studies import STUDY_RULE_NOTE


def strict_json(response) -> Any:
    def reject(token: str) -> Any:
        raise AssertionError(f"La respuesta contiene {token}, que no es JSON válido")

    return json.loads(response.text, parse_constant=reject)


def create(client, name: str = "", **kwargs) -> dict:
    r = client.post("/api/studies", json={"config": pair_config(**kwargs), "name": name})
    assert r.status_code == 201, r.text
    return strict_json(r)


def zip_texts(content: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        return {name: zf.read(name).decode("utf-8-sig") for name in zf.namelist()}


def test_full_study_flow_over_http(client) -> None:
    # 1. Crear: configuración congelada (instantáneas y división), prueba aún no vista.
    created = create(client, name="KO frente a PEP")
    study_id = created["study"]["id"]
    assert created["note"] == STUDY_RULE_NOTE
    assert created["test_viewed"] is False and created["contaminated"] is False
    assert "aún no se ha ejecutado" in created["contamination_text"]
    cfg = created["study"]["config"]
    assert cfg["pair"]["a"]["snapshot_id"] and cfg["frozen_split"]["n_test"] > 0
    assert cfg["include_test"] is False

    listed = client.get("/api/studies").json()
    assert [s["id"] for s in listed] == [study_id]
    assert listed[0]["symbols"] == ["KO", "PEP"] and listed[0]["name"] == "KO frente a PEP"
    assert client.get(f"/api/studies/{study_id}").json()["study"]["id"] == study_id

    # 2. Ajustar umbrales ANTES de ver la prueba: no contamina; solo cambian los campos enviados.
    r = client.put(f"/api/studies/{study_id}/config", json={"strategy": {"entry_z": 2.5}})
    assert r.status_code == 200, r.text
    tuned = r.json()
    assert tuned["contaminated"] is False and tuned["warning"] is None
    strategy = tuned["study"]["config"]["strategy"]
    assert strategy["entry_z"] == 2.5 and strategy["exit_z"] == 0.5 and strategy["stop_z"] == 3.5
    (change,) = tuned["study"]["changes"]
    assert change["fields"] == ["strategy.entry_z"] and change["after_test"] is False

    # 3. Ejecutar la prueba: segmento «prueba» fuera de muestra, registrado.
    r = client.post(f"/api/studies/{study_id}/test")
    assert r.status_code == 200, r.text
    tested = strict_json(r)
    assert tested["contaminated"] is False
    assert tested["contamination_text"].startswith("Prueba no contaminada")
    bt = tested["backtest"]
    assert bt["disclaimer"] == DISCLAIMER
    assert list(bt["segments"]) == ["formacion", "validacion", "prueba"]
    test_seg = bt["segments"]["prueba"]
    assert test_seg["in_sample"] is False and test_seg["label"] == "prueba"
    split = bt["config"]["frozen_split"]
    assert (test_seg["start"], test_seg["end"]) == (split["test_start"], split["test_end"])
    assert bt["test_included"] is True and bt["test_contaminated"] is False
    assert tested["study"]["test_viewed"] is True
    assert [run["source"] for run in tested["study"]["study"]["test_runs"]] == ["prueba"]

    # 4. Exportar: el ZIP incluye la prueba y dice que no está contaminada.
    r = client.get(f"/api/studies/{study_id}/export")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert f'filename="KO_PEP_estudio_{study_id[:8]}.zip"' in r.headers["content-disposition"]
    readme = zip_texts(r.content)["LEEME.txt"]
    assert "Prueba contaminada: no — según el registro persistente del estudio" in readme
    assert "- Prueba:" in readme and "backtest incluido" in readme.split("- Prueba:")[1]

    # 5. Cambiar umbrales DESPUÉS de verla: contaminación permanente, con aviso.
    r = client.put(f"/api/studies/{study_id}/config", json={"strategy": {"exit_z": 0.25}})
    changed = r.json()
    assert changed["contaminated"] is True
    assert "prueba contaminada" in changed["warning"]
    assert "strategy.exit_z" in changed["contamination_reason"]
    assert changed["contamination_text"].startswith("Prueba contaminada.")

    # 6. Volver a la configuración anterior no quita la marca.
    r = client.put(f"/api/studies/{study_id}/config", json={"strategy": {"exit_z": 0.5}})
    assert r.json()["contaminated"] is True

    # 7. Volver a ejecutar la prueba: contaminada, también en el backtest y en la exportación.
    tested2 = client.post(f"/api/studies/{study_id}/test").json()
    assert tested2["contaminated"] is True and tested2["backtest"]["test_contaminated"] is True
    assert "strategy.exit_z" in tested2["contamination_text"]
    texts = zip_texts(client.get(f"/api/studies/{study_id}/export").content)
    readme = texts["LEEME.txt"]
    assert "Prueba contaminada: SÍ — según el registro persistente del estudio" in readme
    assert "strategy.exit_z" in readme
    stats = texts["pruebas_estadisticas.csv"]
    assert "HISTORICO,prueba,par,contaminada,sí" in stats
    assert "HISTORICO,prueba,par,ejecutada,sí" in stats

    # 8. Eliminar.
    assert client.delete(f"/api/studies/{study_id}").status_code == 204
    r = client.get(f"/api/studies/{study_id}")
    assert r.status_code == 404 and r.json()["error"]["code"] == "ESTUDIO_NO_ENCONTRADO"
    assert client.get("/api/studies").json() == []


def test_export_before_test_excludes_it(client) -> None:
    study_id = create(client)["study"]["id"]
    readme = zip_texts(client.get(f"/api/studies/{study_id}/export").content)["LEEME.txt"]
    assert "la prueba aún no se ha ejecutado" in readme
    assert "backtest NO ejecutado" in readme.split("- Prueba:")[1].splitlines()[0]


def test_create_with_include_test_is_rejected(client) -> None:
    r = client.post("/api/studies", json={"config": pair_config(include_test=True)})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "PRUEBA_SOLO_EN_ESTUDIO" and "una sola vez" in err["message"]
    assert client.get("/api/studies").json() == []


@pytest.mark.parametrize("bad_id", ["abc", "..x", "A" * 32, "0" * 32, "0" * 31 + "g", "x" * 80])
def test_invalid_or_missing_ids_are_spanish_404(client, bad_id: str) -> None:
    for method, path in (
        ("get", f"/api/studies/{bad_id}"),
        ("post", f"/api/studies/{bad_id}/test"),
        ("get", f"/api/studies/{bad_id}/export"),
        ("delete", f"/api/studies/{bad_id}"),
    ):
        r = getattr(client, method)(path)
        assert r.status_code == 404, (method, path, r.text)
        err = r.json()["error"]
        assert err["code"] == "ESTUDIO_NO_ENCONTRADO"
        assert "No existe el estudio" in err["message"]
    r = client.put(f"/api/studies/{bad_id}/config", json={"strategy": {"entry_z": 2.5}})
    assert r.status_code == 404


def test_path_traversal_never_reaches_the_store(client) -> None:
    r = client.get("/api/studies/..%2F..%2Fsecreto")
    assert r.status_code == 404
    assert r.json()["error"]["code"] in ("ESTUDIO_NO_ENCONTRADO", "RUTA_NO_ENCONTRADA")


def test_update_validation_errors_are_spanish(client) -> None:
    study_id = create(client)["study"]["id"]
    url = f"/api/studies/{study_id}/config"
    r = client.put(url, json={"strategy": {"exit_z": 3.0}})  # salida > entrada (2,0)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UMBRALES_INCONSISTENTES"
    r = client.put(url, json={"strategy": {"entry_z": -1}})
    assert r.status_code == 422
    assert "strategy.entry_z: debe ser mayor que 0" in r.json()["error"]["message"]
    r = client.put(url, json={"strategy": {"no_existe": 1}})
    assert r.status_code == 422 and "campo no permitido" in r.json()["error"]["message"]
    r = client.put(
        url, json={"analysis": {"split": {"train": 0.5, "validation": 0.25, "test": 0.25}}}
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "DIVISION_CONGELADA"
    r = client.put(url, json={"costs": {"commission_bps": 10}})
    assert r.status_code == 200 and r.json()["study"]["config"]["costs"]["commission_bps"] == 10
    assert r.json()["study"]["config"]["costs"]["slippage_bps_equity"] == 5.0


def test_blocked_pair_test_is_422_and_not_recorded(client) -> None:
    study_id = create(client, a="XOM", b="CVX")["study"]["id"]
    r = client.post(f"/api/studies/{study_id}/test")
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "PRUEBA_NO_SIMULABLE" and "bloqueada" in err["message"]
    assert client.get(f"/api/studies/{study_id}").json()["test_viewed"] is False


def test_new_study_after_seeing_the_test_is_contaminated(client) -> None:
    first = create(client, name="original")["study"]["id"]
    client.post(f"/api/studies/{first}/test")
    tuned = create(client, name="ajustado", strategy={"entry_z": 2.5})
    assert tuned["test_viewed"] is True and tuned["contaminated"] is True
    assert "«original»" in tuned["contamination_reason"]
    r = client.post(f"/api/studies/{tuned['study']['id']}/test")
    assert r.json()["backtest"]["test_contaminated"] is True


def test_studies_survive_a_restart(client, lab_settings, registry) -> None:
    study_id = create(client)["study"]["id"]
    client.post(f"/api/studies/{study_id}/test")
    client.put(f"/api/studies/{study_id}/config", json={"strategy": {"entry_z": 2.25}})
    restarted = create_app(lab_settings, registry)  # nuevo proceso: estado desde disco
    from fastapi.testclient import TestClient

    with TestClient(restarted) as other:
        body = other.get(f"/api/studies/{study_id}").json()
        assert body["contaminated"] is True and body["test_viewed"] is True
        assert body["study"]["config"]["strategy"]["entry_z"] == 2.25
        tested = other.post(f"/api/studies/{study_id}/test").json()
        assert tested["backtest"]["test_contaminated"] is True


def test_openapi_documents_studies(client) -> None:
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    assert paths["/api/studies"]["post"]["summary"] == "Crear un estudio"
    test_op = paths["/api/studies/{study_id}/test"]["post"]
    assert "contaminada" in test_op["description"]
    update = spec["components"]["schemas"]["StrategyPatch"]["properties"]["entry_z"]
    assert update["description"]


def test_analysis_patch_keeps_other_fields_and_split(client) -> None:
    created = create(client)
    study_id = created["study"]["id"]
    before = created["study"]["config"]["analysis"]
    r = client.put(f"/api/studies/{study_id}/config", json={"analysis": {"alpha_level": 0.1}})
    assert r.status_code == 200, r.text
    after = r.json()["study"]["config"]["analysis"]
    assert after["alpha_level"] == 0.1
    assert {k: v for k, v in after.items() if k != "alpha_level"} == {
        k: v for k, v in before.items() if k != "alpha_level"
    }
    assert r.json()["study"]["changes"][0]["fields"] == ["analysis.alpha_level"]
