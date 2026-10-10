"""Trabajos en segundo plano con progreso (decisión I2: en proceso, sin Redis ni Celery).

CONTRATO (a implementar):
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

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["pendiente", "en_curso", "completado", "error", "cancelado"]
ProgressCallback = Callable[[float, str], None]


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

    def submit(self, kind: str, fn: Callable[[ProgressCallback], Any], *, title: str = "") -> Job:
        raise NotImplementedError

    def get(self, job_id: str) -> Job:
        raise NotImplementedError

    def list(self) -> list[JobSummary]:
        raise NotImplementedError

    def cancel(self, job_id: str) -> Job:
        raise NotImplementedError

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Espera a que termine (para pruebas y para la CLI)."""
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
