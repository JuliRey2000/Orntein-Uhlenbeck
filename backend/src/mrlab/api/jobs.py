"""Trabajos en segundo plano con progreso (decisión I2: en proceso, sin Redis ni Celery).

- ``JobManager(max_workers)``: ThreadPoolExecutor. ``submit(kind, fn, *, title)`` donde
  ``fn(report)`` recibe un callback ``report(progress: float 0..1, message: str)`` y devuelve un
  resultado JSON-serializable (dict/list o modelo pydantic). Devuelve el ``Job`` creado.
- Estados: "pendiente" → "en_curso" → "completado" | "error" | "cancelado".
  Una excepción ``LabError`` deja ``error = {code, message, details}`` (en español); cualquier
  otra excepción deja un error genérico en español (code "ERROR_INTERNO") SIN exponer la traza.
- ``get(job_id)`` (KeyError → el router responde 404 en español), ``list()`` (más recientes
  primero), ``cancel(job_id)`` (solo si está pendiente; si ya corre, marca ``cancel_requested`` y
  el callback ``report`` lanza ``JobCancelled`` en la siguiente llamada).
- Retención: guarda como máximo ``max_jobs`` (default 200) trabajos terminados; los más viejos se
  descartan. Thread-safe (lock). ``shutdown()`` cancela los pendientes y espera a los activos.
- Marcas de tiempo UTC (``created_at``, ``started_at``, ``finished_at``) en ISO.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from mrlab.domain.errors import ConfigError, DataError, LabError

logger = logging.getLogger("mrlab.jobs")

INTERNAL_ERROR_MESSAGE = (
    "El trabajo falló por un error interno inesperado. Revisa los registros del servidor."
)

JobStatus = Literal["pendiente", "en_curso", "completado", "error", "cancelado"]
ProgressCallback = Callable[[float, str], None]
_FINISHED = frozenset({"completado", "error", "cancelado"})


def _now() -> datetime:
    return datetime.now(UTC)


class JobCancelled(Exception):
    """Se lanza dentro del trabajo cuando se pidió cancelarlo."""


class Job(BaseModel):
    id: str
    kind: str
    title: str
    status: JobStatus = "pendiente"
    progress: float = 0.0
    message: str = ""
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested: bool = False
    result: Any = None
    error: dict[str, Any] | None = None


class JobSummary(BaseModel):
    """Vista sin ``result`` (para listados)."""

    id: str
    kind: str
    title: str
    status: JobStatus
    progress: float
    message: str
    created_at: datetime
    finished_at: datetime | None = None
    error: dict[str, Any] | None = Field(default=None)


class JobManager:
    def __init__(self, max_workers: int = 2, max_jobs: int = 200):
        self.max_workers = max_workers
        self.max_jobs = max_jobs
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mrlab-job")

    # ------------------------------------------------------------------ helpers

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs[job_id] = job.model_copy(update=changes)

    def _prune(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in _FINISHED]
        excess = len(finished) - self.max_jobs
        if excess > 0:
            for job in sorted(finished, key=lambda j: j.created_at)[:excess]:
                self._jobs.pop(job.id, None)
                self._futures.pop(job.id, None)

    def _run(self, job_id: str, fn: Callable[[ProgressCallback], Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status == "cancelado":
                return
            self._jobs[job_id] = job.model_copy(update={"status": "en_curso", "started_at": _now()})

        def report(progress: float, message: str = "") -> None:
            with self._lock:
                current = self._jobs[job_id]
                if current.cancel_requested:
                    raise JobCancelled()
                self._jobs[job_id] = current.model_copy(
                    update={"progress": min(1.0, max(0.0, float(progress))), "message": message}
                )

        try:
            result = fn(report)
            if isinstance(result, BaseModel):
                result = result.model_dump(mode="json")
            self._update(
                job_id,
                status="completado",
                progress=1.0,
                result=result,
                finished_at=_now(),
                message="Trabajo completado.",
            )
        except JobCancelled:
            self._update(
                job_id, status="cancelado", finished_at=_now(), message="Trabajo cancelado."
            )
        except LabError as exc:
            self._update(
                job_id, status="error", error=exc.to_dict(), finished_at=_now(), message=exc.message
            )
        except Exception:  # noqa: BLE001 - nunca exponer la traza al cliente
            logger.exception("Error no controlado en el trabajo %s", job_id)
            self._update(
                job_id,
                status="error",
                finished_at=_now(),
                error={"code": "ERROR_INTERNO", "message": INTERNAL_ERROR_MESSAGE, "details": {}},
                message=INTERNAL_ERROR_MESSAGE,
            )
        finally:
            with self._lock:
                self._prune()

    # ------------------------------------------------------------------ API pública

    def submit(self, kind: str, fn: Callable[[ProgressCallback], Any], *, title: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, title=title or kind, created_at=_now())
        with self._lock:
            self._jobs[job.id] = job
            self._futures[job.id] = self._executor.submit(self._run, job.id, fn)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise DataError(
                f"No existe el trabajo «{job_id}» (puede haberse descartado por antigüedad).",
                code="TRABAJO_NO_ENCONTRADO",
                details={"job_id": job_id},
            )
        return job

    def list(self) -> list[JobSummary]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [
            JobSummary(**j.model_dump(exclude={"result", "started_at", "cancel_requested"}))
            for j in jobs
        ]

    def cancel(self, job_id: str) -> Job:
        self.get(job_id)  # 404 si no existe
        with self._lock:
            # El estado se comprueba DENTRO del lock: si el trabajo terminó entre la consulta y
            # este punto, se responde «ya terminó» en lugar de marcar cancel_requested.
            job = self._jobs.get(job_id)
            if job is None:
                raise DataError(
                    f"No existe el trabajo «{job_id}» (puede haberse descartado por antigüedad).",
                    code="TRABAJO_NO_ENCONTRADO",
                    details={"job_id": job_id},
                )
            if job.status in _FINISHED:
                raise ConfigError(
                    f"El trabajo ya terminó (estado: {job.status}); no se puede cancelar.",
                    code="TRABAJO_YA_TERMINADO",
                    details={"job_id": job_id, "status": job.status},
                )
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                self._jobs[job_id] = self._jobs[job_id].model_copy(
                    update={
                        "status": "cancelado",
                        "finished_at": _now(),
                        "message": "Trabajo cancelado.",
                    }
                )
            else:
                self._jobs[job_id] = self._jobs[job_id].model_copy(
                    update={"cancel_requested": True}
                )
            return self._jobs[job_id]

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        self.get(job_id)
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except CancelledError:
                pass
        return self.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status == "pendiente":
                    self._jobs[job_id] = job.model_copy(
                        update={
                            "status": "cancelado",
                            "finished_at": _now(),
                            "message": "Servidor detenido.",
                        }
                    )
