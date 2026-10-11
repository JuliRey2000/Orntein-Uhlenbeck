"""Integración de la fase 2: arranque sin efectos, rutas del servidor ocultas y uvicorn real."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from api_support import pair_config

from mrlab.api.errors import redact_paths, server_path_replacements
from mrlab.api.main import create_app
from mrlab.domain.errors import DataError

TIMEOUT = 10.0

NO_NETWORK = """
import socket

def _sin_red(*args, **kwargs):
    raise RuntimeError("importar la API no debe usar la red")

socket.socket.connect = _sin_red
socket.create_connection = _sin_red
socket.getaddrinfo = _sin_red
"""


def _lab_env(root: Path) -> dict[str, str]:
    data = root / "datos"
    return {
        **os.environ,
        "MRLAB_DATA_DIR": str(data),
        "MRLAB_CACHE_DIR": str(data / "cache"),
        "MRLAB_STUDIES_DIR": str(data / "studies"),
        "MRLAB_UPLOADS_DIR": str(data / "uploads"),
    }


def test_importing_main_creates_no_directories_and_uses_no_network(tmp_path):
    code = NO_NETWORK + "import mrlab.api.main as m\nassert m.app.title\n"
    work = tmp_path / "trabajo"
    work.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=work,
        env=_lab_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "datos").exists()
    assert list(work.iterdir()) == []


def test_directories_are_created_on_first_use(client, lab_settings):
    assert not lab_settings.data_dir.exists()
    assert client.get("/api/health").status_code == 200
    assert not lab_settings.data_dir.exists()
    r = client.post(
        "/api/data/load",
        json={"provider": "yahoo", "symbols": ["KO"], "start": "2016-01-04", "end": "2021-12-31"},
    )
    assert r.status_code == 200
    assert lab_settings.cache_dir.is_dir()


def test_providers_from_toml_use_the_app_settings(lab_settings):
    """Sin registro inyectado, los proveedores usan la configuración de la app, no la del
    entorno: los CSV se guardan en SU carpeta de subidas y Yahoo usa SUS reintentos."""
    settings = lab_settings.model_copy(
        update={"yahoo_max_retries": 1, "yahoo_backoff_base_seconds": 0.0}
    )
    state = create_app(settings).state.lab
    assert state.registry.get("csv").directory == settings.uploads_dir
    yahoo = state.registry.get("yahoo")
    assert yahoo.max_retries == 1
    assert not settings.data_dir.exists()


# --------------------------------------------------------------------------- rutas del servidor


def test_redact_paths_replaces_every_server_folder(lab_settings):
    paths = server_path_replacements(lab_settings)
    text = f"Archivo: {lab_settings.cache_dir / 'yahoo' / 'KO' / 'x.parquet'}"
    out = redact_paths({"m": text, "l": [str(lab_settings.studies_dir)], "n": 3}, paths)
    assert str(lab_settings.data_dir) not in str(out)
    assert out["m"].startswith("Archivo: [caché del servidor]")
    assert out["l"] == ["[estudios del servidor]"]
    assert out["n"] == 3


def test_redact_paths_ignores_relative_folders(lab_settings, monkeypatch, tmp_path):
    """Con carpetas relativas («data») solo se sustituye su ruta absoluta: nunca la palabra
    «data» de un mensaje ni una ruta relativa (que no revela nada del servidor)."""
    monkeypatch.chdir(tmp_path)
    settings = lab_settings.model_copy(
        update={
            "data_dir": Path("data"),
            "cache_dir": Path("data/cache"),
            "studies_dir": Path("data/studies"),
            "uploads_dir": Path("data/uploads"),
        }
    )
    paths = server_path_replacements(settings)
    assert all(Path(p).is_absolute() for p, _ in paths)
    text = f"metadata en data/cache; archivo {tmp_path / 'data' / 'cache' / 'x.parquet'}"
    assert redact_paths(text, paths) == (
        "metadata en data/cache; archivo [caché del servidor]/x.parquet"
    )


def test_snapshot_not_found_hides_the_cache_path(client, lab_settings):
    r = client.get("/api/data/series", params={"snapshot_id": "no_existe_1d_abc"})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "SNAPSHOT_NO_ENCONTRADO"
    assert str(lab_settings.cache_dir) not in r.text
    assert str(lab_settings.data_dir) not in r.text
    assert "[caché del servidor]" in err["message"]
    assert err["details"]["cache"] == "[caché del servidor]"


def test_corrupt_snapshot_hides_the_file_path(client, lab_settings):
    r = client.post(
        "/api/data/load",
        json={"provider": "yahoo", "symbols": ["KO"], "start": "2016-01-04", "end": "2021-12-31"},
    )
    sid = r.json()["series"][0]["snapshot_id"]
    files = list(lab_settings.cache_dir.rglob(f"{sid}.parquet"))
    assert len(files) == 1
    files[0].write_bytes(b"no es un parquet")
    r = client.get("/api/data/series", params={"snapshot_id": sid})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SNAPSHOT_CORRUPTO"
    assert str(lab_settings.data_dir) not in r.text
    assert "[caché del servidor]" in r.text


def test_job_errors_hide_server_paths(client, app, lab_settings):
    jobs = app.state.lab.jobs
    secret = str(lab_settings.cache_dir / "yahoo")

    def fn(report):
        raise DataError(f"Falta un archivo en {secret}.", details={"path": secret})

    job = jobs.submit("prueba", fn)
    jobs.wait(job.id, timeout=TIMEOUT)
    for url in (f"/api/jobs/{job.id}", "/api/jobs"):
        r = client.get(url)
        assert r.status_code == 200
        assert str(lab_settings.data_dir) not in r.text
        assert "[caché del servidor]" in r.text


def test_compare_job_items_hide_server_paths(client, app, lab_settings):
    """Un par con una instantánea inexistente falla en su fila; la ruta de la caché no sale."""
    bad = pair_config()
    bad["pair"]["a"]["snapshot_id"] = "no_existe_1d_abc"
    r = client.post("/api/backtest/compare", json={"configs": [bad, pair_config("XOM", "CVX")]})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    app.state.lab.jobs.wait(job_id, timeout=60)
    r = client.get(f"/api/jobs/{job_id}")
    body = r.json()
    assert body["status"] == "completado"
    first = body["result"]["items"][0]
    assert first["error"]["code"] == "SNAPSHOT_NO_ENCONTRADO"
    assert str(lab_settings.data_dir) not in r.text


# --------------------------------------------------------------------------- uvicorn real


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.slow
def test_uvicorn_serves_health_and_docs(tmp_path):
    """``uvicorn mrlab.api.main:app`` arranca, responde /api/health y /docs, y se detiene."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mrlab.api.main:app", "--port", str(port)],
        cwd=tmp_path,
        env=_lab_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        # trust_env=False: un proxy configurado en el entorno no debe interceptar localhost.
        with httpx.Client(base_url=base, trust_env=False, timeout=10) as http:
            deadline = time.monotonic() + 60
            health = None
            while time.monotonic() < deadline:
                try:
                    health = http.get("/api/health", timeout=2)
                    break
                except httpx.TransportError:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.2)
            assert health is not None, proc.stdout.read() if proc.stdout else ""
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            docs = http.get("/docs")
            assert docs.status_code == 200
            assert "swagger" in docs.text.lower()
            assert http.get("/openapi.json").json()["info"]["title"].startswith("Laboratorio")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    assert not (tmp_path / "datos").exists()
