"""Trabajos en segundo plano.

CONTRATO:
- GET /jobs → list[JobSummaryOut] (más recientes primero, sin ``result``).
- GET /jobs/{id} → JobOut (con ``result`` cuando está completado). Inexistente → 404
  TRABAJO_NO_ENCONTRADO en español.
- DELETE /jobs/{id} → JobOut (cancelación: si estaba pendiente queda «cancelado»; si ya corre se
  marca ``cancel_requested`` y se detiene en el siguiente punto de control). Si ya terminó → 409
  TRABAJO_YA_TERMINADO en español.
Los resultados nunca contienen NaN ni infinitos (se envían como null).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import ConfigDict

from mrlab.api.errors import json_safe
from mrlab.api.jobs import Job, JobSummary
from mrlab.api.openapi_models import describe
from mrlab.api.schemas import ERROR_RESPONSES, ErrorResponse
from mrlab.api.state import AppState, get_state
from mrlab.domain.errors import ConfigError

router = APIRouter(tags=["trabajos"])

OK = "Respuesta correcta."
CANCEL_REQUESTED_MESSAGE = (
    "Cancelación solicitada: el trabajo se detendrá en el próximo punto de control."
)

JOB_FIELDS = {
    "id": "Identificador del trabajo.",
    "kind": "Tipo de trabajo (p. ej. «comparacion»).",
    "title": "Descripción corta.",
    "status": "Estado: «pendiente», «en_curso», «completado», «error» o «cancelado».",
    "progress": "Progreso de 0 a 1.",
    "message": "Último mensaje de progreso, en español.",
    "created_at": "Creación (UTC, ISO 8601).",
    "started_at": "Inicio de la ejecución (UTC).",
    "finished_at": "Fin de la ejecución (UTC).",
    "cancel_requested": "True si se pidió cancelarlo mientras se ejecutaba.",
    "result": "Resultado (solo si está completado).",
    "error": "Error en español {code, message, details} si terminó con error (sin trazas).",
}


class JobOut(Job):
    """Trabajo en segundo plano con su progreso y, al terminar, su resultado o su error."""

    model_config = ConfigDict(json_schema_extra=describe(JOB_FIELDS))


class JobSummaryOut(JobSummary):
    """Resumen de un trabajo (sin el resultado)."""

    model_config = ConfigDict(json_schema_extra=describe(JOB_FIELDS))


JOB_ID = Path(..., min_length=1, max_length=64, description="Identificador del trabajo.")


def _out(job: Job) -> JobOut:
    data = job.model_dump()
    data["result"] = json_safe(data.get("result"))
    data["error"] = json_safe(data.get("error"))
    return JobOut(**data)


@router.get(
    "/jobs",
    response_model=list[JobSummaryOut],
    summary="Listar trabajos",
    description="Trabajos en segundo plano, los más recientes primero (sin el resultado).",
    response_description=OK,
    responses={500: ERROR_RESPONSES[500]},
)
def list_jobs(state: AppState = Depends(get_state)) -> list[JobSummary]:
    return state.jobs.list()


@router.get(
    "/jobs/{job_id}",
    response_model=JobOut,
    summary="Consultar un trabajo",
    description=(
        "Estado, progreso y, cuando termina, el resultado o el error (en español, sin trazas). "
        "Consulta periódicamente esta ruta para mostrar el progreso."
    ),
    response_description=OK,
    responses={404: ERROR_RESPONSES[404], 500: ERROR_RESPONSES[500]},
)
def get_job(job_id: str = JOB_ID, state: AppState = Depends(get_state)) -> JobOut:
    return _out(state.jobs.get(job_id))


@router.delete(
    "/jobs/{job_id}",
    response_model=JobOut,
    summary="Cancelar un trabajo",
    description=(
        "Cancela un trabajo pendiente, o pide detener uno en curso (se detiene en el próximo "
        "punto de control y queda «cancelado»). Un trabajo que ya terminó no se puede cancelar "
        "(409)."
    ),
    response_description=OK,
    responses={
        404: ERROR_RESPONSES[404],
        409: {"model": ErrorResponse, "description": "El trabajo ya terminó"},
        500: ERROR_RESPONSES[500],
    },
)
def cancel_job(job_id: str = JOB_ID, state: AppState = Depends(get_state)) -> JobOut:
    try:
        job = state.jobs.cancel(job_id)
    except ConfigError as exc:
        if exc.code != "TRABAJO_YA_TERMINADO":
            raise
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": exc.message, "details": json_safe(exc.details)},
        ) from exc
    out = _out(job)
    if job.cancel_requested and job.status in ("pendiente", "en_curso"):
        out = out.model_copy(update={"message": CANCEL_REQUESTED_MESSAGE})
    return out
