"""API de trabajos en segundo plano: listado, detalle, progreso, cancelación y errores."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from mrlab.api.jobs import INTERNAL_ERROR_MESSAGE, JobManager
from mrlab.api.routers.jobs import CANCEL_REQUESTED_MESSAGE
from mrlab.domain.errors import DataError

TIMEOUT = 10.0


@pytest.fixture
def jobs(app) -> JobManager:
    return app.state.lab.jobs


@pytest.fixture
def gate() -> Iterator[threading.Event]:
    """Evento que libera los trabajos bloqueados (siempre se libera al terminar la prueba)."""
    event = threading.Event()
    yield event
    event.set()


def blocking_job(started: threading.Event, release: threading.Event, progress: float = 0.3):
    def fn(report):
        report(progress, "Analizando 3 de 10 pares")
        started.set()
        assert release.wait(TIMEOUT)
        report(0.9, "Casi listo")  # punto de control: aquí se detiene si se pidió cancelar
        return {"ok": True}

    return fn


def test_list_is_empty_at_start(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert r.json() == []


def test_completed_job_with_json_safe_result(client, jobs):
    def fn(report):
        report(0.5, "Mitad")
        return {"x": 1.5, "nan": float("nan"), "lista": [float("inf"), 2], "texto": "ñ"}

    job = jobs.submit("prueba", fn, title="Trabajo de prueba")
    jobs.wait(job.id, timeout=TIMEOUT)
    r = client.get(f"/api/jobs/{job.id}")
    assert r.status_code == 200
    assert "NaN" not in r.text and "Infinity" not in r.text
    body = r.json()
    assert body["id"] == job.id
    assert body["kind"] == "prueba"
    assert body["title"] == "Trabajo de prueba"
    assert body["status"] == "completado"
    assert body["progress"] == 1.0
    assert body["message"] == "Trabajo completado."
    assert body["result"] == {"x": 1.5, "nan": None, "lista": [None, 2], "texto": "ñ"}
    assert body["error"] is None
    assert body["started_at"] and body["finished_at"]


def test_pydantic_result_is_serialized(client, jobs):
    class Out(BaseModel):
        valor: float
        nombre: str

    job = jobs.submit("prueba", lambda report: Out(valor=2.5, nombre="par"))
    jobs.wait(job.id, timeout=TIMEOUT)
    assert client.get(f"/api/jobs/{job.id}").json()["result"] == {"valor": 2.5, "nombre": "par"}


def test_list_has_summaries_most_recent_first(client, jobs):
    first = jobs.submit("a", lambda report: 1)
    jobs.wait(first.id, timeout=TIMEOUT)
    second = jobs.submit("b", lambda report: 2)
    jobs.wait(second.id, timeout=TIMEOUT)
    r = client.get("/api/jobs")
    assert r.status_code == 200
    listed = r.json()
    assert [j["id"] for j in listed] == [second.id, first.id]
    for j in listed:
        assert "result" not in j
        assert j["status"] == "completado"


def test_progress_is_visible_while_running(client, jobs, gate):
    started = threading.Event()
    job = jobs.submit("ranking", blocking_job(started, gate), title="Ranking")
    assert started.wait(TIMEOUT)
    body = client.get(f"/api/jobs/{job.id}").json()
    assert body["status"] == "en_curso"
    assert body["progress"] == pytest.approx(0.3)
    assert body["message"] == "Analizando 3 de 10 pares"
    assert body["result"] is None
    gate.set()
    jobs.wait(job.id, timeout=TIMEOUT)
    assert client.get(f"/api/jobs/{job.id}").json()["status"] == "completado"


def test_unknown_job_is_404_in_spanish(client):
    r = client.get("/api/jobs/no-existe")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "TRABAJO_NO_ENCONTRADO"
    assert "No existe el trabajo" in err["message"]
    r = client.delete("/api/jobs/no-existe")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "TRABAJO_NO_ENCONTRADO"


def test_job_id_too_long_is_422(client):
    r = client.get("/api/jobs/" + "a" * 65)
    assert r.status_code == 422
    assert r.json()["error"]["details"]["campos"][0]["campo"] == "job_id"


def test_cancel_running_job(client, jobs, gate):
    started = threading.Event()
    job = jobs.submit("ranking", blocking_job(started, gate))
    assert started.wait(TIMEOUT)
    r = client.delete(f"/api/jobs/{job.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "en_curso"
    assert body["cancel_requested"] is True
    assert body["message"] == CANCEL_REQUESTED_MESSAGE
    gate.set()
    jobs.wait(job.id, timeout=TIMEOUT)
    done = client.get(f"/api/jobs/{job.id}").json()
    assert done["status"] == "cancelado"
    assert done["message"] == "Trabajo cancelado."
    assert done["result"] is None
    # Ya terminó (cancelado): cancelarlo de nuevo es un conflicto.
    again = client.delete(f"/api/jobs/{job.id}")
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "TRABAJO_YA_TERMINADO"


def test_cancel_pending_job(client, jobs, gate, lab_settings):
    started = [threading.Event() for _ in range(lab_settings.job_workers)]
    busy = [jobs.submit("ocupado", blocking_job(s, gate)) for s in started]
    for s in started:
        assert s.wait(TIMEOUT)
    pending = jobs.submit("en_espera", lambda report: "no debería ejecutarse")
    assert client.get(f"/api/jobs/{pending.id}").json()["status"] == "pendiente"
    r = client.delete(f"/api/jobs/{pending.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "cancelado"
    assert body["finished_at"]
    gate.set()
    for job in busy:
        jobs.wait(job.id, timeout=TIMEOUT)
    assert client.get(f"/api/jobs/{pending.id}").json()["result"] is None


def test_cancel_finished_job_is_409_in_spanish(client, jobs):
    job = jobs.submit("prueba", lambda report: {"ok": True})
    jobs.wait(job.id, timeout=TIMEOUT)
    r = client.delete(f"/api/jobs/{job.id}")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "TRABAJO_YA_TERMINADO"
    assert "ya terminó" in err["message"]
    assert err["details"] == {"job_id": job.id, "status": "completado"}
    assert client.get(f"/api/jobs/{job.id}").json()["status"] == "completado"


def test_lab_error_in_job_is_reported_in_spanish(client, jobs):
    def fn(report):
        raise DataError("Faltan datos para el par «KO/PEP».", code="DATOS_INSUFICIENTES")

    job = jobs.submit("prueba", fn)
    jobs.wait(job.id, timeout=TIMEOUT)
    body = client.get(f"/api/jobs/{job.id}").json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "DATOS_INSUFICIENTES"
    assert body["error"]["message"] == "Faltan datos para el par «KO/PEP»."
    listed = client.get("/api/jobs").json()[0]
    assert listed["error"]["code"] == "DATOS_INSUFICIENTES"


def test_unexpected_error_in_job_hides_internals(client, jobs):
    def fn(report):
        raise RuntimeError("detalle interno /ruta/secreta.py línea 42")

    job = jobs.submit("prueba", fn)
    jobs.wait(job.id, timeout=TIMEOUT)
    r = client.get(f"/api/jobs/{job.id}")
    body = r.json()
    assert body["status"] == "error"
    assert body["error"] == {
        "code": "ERROR_INTERNO",
        "message": INTERNAL_ERROR_MESSAGE,
        "details": {},
    }
    assert "secreta" not in r.text
    assert "Traceback" not in r.text


def test_cancel_rechecks_status_under_the_lock(client, jobs, monkeypatch):
    """Carrera: el trabajo termina entre la consulta inicial y el lock de ``cancel``.

    Se simula devolviendo desde ``get`` una copia «en_curso» ya desactualizada: ``cancel`` debe
    volver a mirar el estado dentro del lock y responder 409, sin marcar cancel_requested."""
    job = jobs.submit("prueba", lambda report: {"ok": True})
    jobs.wait(job.id, timeout=TIMEOUT)
    stale = jobs.get(job.id).model_copy(update={"status": "en_curso"})
    monkeypatch.setattr(jobs, "get", lambda job_id: stale)
    r = client.delete(f"/api/jobs/{job.id}")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "TRABAJO_YA_TERMINADO"
    monkeypatch.undo()
    done = jobs.get(job.id)
    assert done.status == "completado"
    assert done.cancel_requested is False
