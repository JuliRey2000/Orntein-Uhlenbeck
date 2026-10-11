"""Estudios: congelar una configuración y ejecutar la prueba fuera de muestra (una vez).

CONTRATO, sobre mrlab.studies.StudyStore:
- POST /studies  StudyCreate {config: RunConfig, name: str = ""} → StudyOut (201).
- GET /studies → list[StudySummaryOut] {id, name, symbols, created_at, updated_at,
  test_viewed, contaminated}.
- GET /studies/{id} → StudyOut {study (Study), test_viewed, contaminated, contamination_reason,
  note (español sobre la regla de la prueba)}.
- PUT /studies/{id}/config  StudyUpdate {strategy?, costs?, analysis?} → StudyOut. Si la prueba
  ya se vio, la respuesta incluye el aviso de contaminación.
- POST /studies/{id}/test → StudyTestOut {study: StudyOut, backtest: BacktestOut (con el
  segmento "prueba"), contaminated, contamination_text (texto veraz en español)}.
- GET /studies/{id}/export?excel_es=false → ZIP (ver exports.py) del estudio; incluye la prueba
  solo si ya se ejecutó (en este estudio); refleja la contaminación.
- DELETE /studies/{id} → 204.
Ids inválidos o inexistentes → 404 ESTUDIO_NO_ENCONTRADO en español.

Precisiones de la implementación:
- PUT /studies/{id}/config cambia SOLO los campos enviados de cada sección (los demás conservan
  su valor actual) y valida la sección completa resultante; los datos del par, las instantáneas
  y la división no cambian (cambiar «analysis.split» → 400 DIVISION_CONGELADA).
- StudyOut lleva ``contamination_text`` (texto veraz para la interfaz) y, en las respuestas a un
  cambio hecho después de ver la prueba, ``warning`` con el aviso de contaminación.
- La exportación del estudio se descarga como ``<base>_estudio_<8 primeros caracteres del
  id>.zip``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from starlette.responses import Response

from mrlab.analysis.pair_pipeline import AnalysisConfig
from mrlab.api.dto import (
    ANALYSIS_FIELDS,
    STRATEGY_EXTRA_FIELDS,
    BacktestOut,
    RunConfigIn,
    backtest_out,
    core_config,
)
from mrlab.api.errors import json_safe, validation_summary
from mrlab.api.routers.analysis import ANALYSIS_RESPONSES
from mrlab.api.routers.exports import ZIP_DESCRIPTION, ZIP_RESPONSES, zip_response
from mrlab.api.schemas import ERROR_RESPONSES
from mrlab.api.state import AppState, get_state
from mrlab.reporting.run_config import RunConfig
from mrlab.strategy.params import CostParams, StrategyParams
from mrlab.studies import STUDY_RULE_NOTE, Study, contamination_text

router = APIRouter(tags=["estudios"])

OK = "Respuesta correcta."
STUDY_ID = Path(..., description="Identificador del estudio (32 caracteres hexadecimales).")
STUDY_RESPONSES = {k: v for k, v in ERROR_RESPONSES.items() if k in (400, 404, 422, 500)}


# --------------------------------------------------------------------------- modelos


class StudyCreate(BaseModel):
    """Configuración a congelar y nombre opcional del estudio."""

    model_config = ConfigDict(extra="forbid")

    config: RunConfigIn = Field(
        description=(
            "Configuración del par (include_test debe ser false: la prueba se ejecuta después, "
            "una sola vez, con POST /api/studies/{id}/test)."
        )
    )
    name: str = Field("", max_length=120, description="Nombre del estudio (por defecto «A / B»).")


class StudyDocOut(Study):
    """Estudio guardado: configuración congelada, ejecuciones de la prueba y cambios."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Estudio guardado en data/studies/ (sobrevive a reinicios): configuración "
                "congelada, cada ejecución de la prueba y cada cambio de configuración."
            )
        }
    )


class StudyOut(BaseModel):
    """Estudio con el estado de su prueba."""

    study: StudyDocOut = Field(description="Estudio completo.")
    test_viewed: bool = Field(description="True si la prueba ya se ejecutó (y se vio).")
    contaminated: bool = Field(description="True si la prueba está contaminada (permanente).")
    contamination_reason: str | None = Field(None, description="Motivo de la contaminación.")
    contamination_text: str = Field(description="Estado de la prueba explicado en español.")
    warning: str | None = Field(
        None, description="Aviso tras un cambio hecho después de ver la prueba."
    )
    note: str = Field(STUDY_RULE_NOTE, description="Regla de la prueba fuera de muestra.")


class StudySummaryOut(BaseModel):
    """Resumen de un estudio para listados."""

    id: str = Field(description="Identificador del estudio.")
    name: str = Field(description="Nombre del estudio.")
    symbols: list[str] = Field(description="Símbolos [A, B] del par.")
    created_at: datetime = Field(description="Creación (UTC).")
    updated_at: datetime = Field(description="Última modificación (UTC).")
    test_viewed: bool = Field(description="True si la prueba ya se ejecutó.")
    contaminated: bool = Field(description="True si la prueba está contaminada.")


class StudyTestOut(BaseModel):
    """Resultado de ejecutar la prueba de un estudio."""

    study: StudyOut = Field(description="Estudio actualizado (con la ejecución registrada).")
    backtest: BacktestOut = Field(
        description="Backtest con los segmentos de formación, validación y PRUEBA."
    )
    contaminated: bool = Field(description="True si la prueba está contaminada.")
    contamination_text: str = Field(description="Texto veraz sobre el estado de la prueba.")


def _partial(model: type[BaseModel], name: str, doc: str, extra: dict[str, str]) -> type[BaseModel]:
    """Versión de ``model`` con todos los campos opcionales (para cambios parciales)."""
    fields: dict[str, Any] = {}
    for field_name, info in model.model_fields.items():
        description = info.description or extra.get(field_name) or ""
        fields[field_name] = (info.annotation | None, Field(None, description=description))
    return create_model(  # type: ignore[call-overload,no-any-return]
        name, __doc__=doc, __config__=ConfigDict(extra="forbid"), **fields
    )


StrategyPatch = _partial(
    StrategyParams,
    "StrategyPatch",
    "Cambios de la estrategia: solo los campos enviados; los demás conservan su valor.",
    STRATEGY_EXTRA_FIELDS,
)
CostsPatch = _partial(
    CostParams,
    "CostsPatch",
    "Cambios de los costes: solo los campos enviados; los demás conservan su valor.",
    {},
)
AnalysisPatch = _partial(
    AnalysisConfig,
    "AnalysisPatch",
    "Cambios del análisis: solo los campos enviados (la división «split» está congelada).",
    ANALYSIS_FIELDS,
)


class StudyUpdate(BaseModel):
    """Cambios de parámetros (los datos del par y la división no cambian)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"strategy": {"entry_z": 2.5}}]},
    )

    # Modelos creados con create_model: los verificadores de tipos no los ven como clases.
    strategy: StrategyPatch | None = Field(  # type: ignore[valid-type]
        None, description="Cambios de la estrategia."
    )
    costs: CostsPatch | None = Field(  # type: ignore[valid-type]
        None, description="Cambios de los costes."
    )
    analysis: AnalysisPatch | None = Field(  # type: ignore[valid-type]
        None, description="Cambios del análisis."
    )


# --------------------------------------------------------------------------- utilidades


def study_out(study: Study, warning: str | None = None) -> StudyOut:
    return StudyOut(
        study=StudyDocOut.model_validate(study.model_dump()),
        test_viewed=study.test_viewed,
        contaminated=study.contaminated,
        contamination_reason=study.contamination_reason,
        contamination_text=contamination_text(study),
        warning=warning,
    )


def _merge(section: str, current: BaseModel, patch: BaseModel | None, cls: type[BaseModel]) -> Any:
    """Sección actual + campos enviados, validada completa (errores 422 en español)."""
    if patch is None:
        return None
    changes = patch.model_dump(exclude_unset=True)
    if not changes:
        return None
    try:
        return cls.model_validate({**current.model_dump(), **changes})
    except ValidationError as exc:
        errors = [{**e, "loc": (section, *e.get("loc", ()))} for e in exc.errors()]
        message, items = validation_summary(errors)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "PETICION_INVALIDA",
                "message": message,
                "details": {"campos": json_safe(items)},
            },
        ) from None


# --------------------------------------------------------------------------- endpoints


@router.post(
    "/studies",
    response_model=StudyOut,
    status_code=201,
    summary="Crear un estudio",
    description=(
        "Congela la configuración de un par: fija las instantáneas de datos y las fechas de "
        "corte (formación/validación/prueba) y la guarda en data/studies/. Es la única vía para "
        "ejecutar la prueba fuera de muestra. La configuración se envía sin la prueba "
        "(include_test = false; si no → 400)."
    ),
    response_description="Estudio creado.",
    responses=ANALYSIS_RESPONSES,
)
def create_study(req: StudyCreate, state: AppState = Depends(get_state)) -> StudyOut:
    config: RunConfig = core_config(req.config)
    return study_out(state.studies.create(config, state.data, name=req.name))


@router.get(
    "/studies",
    response_model=list[StudySummaryOut],
    summary="Listar estudios",
    description="Estudios guardados, los más recientes primero.",
    response_description=OK,
    responses={500: ERROR_RESPONSES[500]},
)
def list_studies(state: AppState = Depends(get_state)) -> list[StudySummaryOut]:
    return [
        StudySummaryOut(
            id=s.id,
            name=s.name,
            symbols=list(s.symbols),
            created_at=s.created_at,
            updated_at=s.updated_at,
            test_viewed=s.test_viewed,
            contaminated=s.contaminated,
        )
        for s in state.studies.list()
    ]


@router.get(
    "/studies/{study_id}",
    response_model=StudyOut,
    summary="Consultar un estudio",
    description=(
        "Configuración congelada, ejecuciones de la prueba, cambios registrados y estado de la "
        "prueba (contaminada o no, con su motivo)."
    ),
    response_description=OK,
    responses=STUDY_RESPONSES,
)
def get_study(study_id: str = STUDY_ID, state: AppState = Depends(get_state)) -> StudyOut:
    return study_out(state.studies.get(study_id))


@router.put(
    "/studies/{study_id}/config",
    response_model=StudyOut,
    summary="Cambiar parámetros de un estudio",
    description=(
        "Cambia parámetros de la estrategia, de los costes o del análisis (solo los campos "
        "enviados; los demás conservan su valor). Los datos del par, las instantáneas y la "
        "división no cambian. Cada cambio queda registrado. Si la prueba ya se ejecutó, cualquier "
        "cambio marca el estudio como «prueba contaminada» de forma PERMANENTE (también en las "
        "exportaciones), aunque después se vuelva a la configuración original."
    ),
    response_description=OK,
    responses=STUDY_RESPONSES,
)
def update_study(
    req: StudyUpdate, study_id: str = STUDY_ID, state: AppState = Depends(get_state)
) -> StudyOut:
    before = state.studies.get(study_id)
    current = before.config
    strategy = _merge("strategy", current.strategy, req.strategy, StrategyParams)
    costs = _merge("costs", current.costs, req.costs, CostParams)
    analysis = _merge("analysis", current.analysis, req.analysis, AnalysisConfig)
    study = state.studies.update_config(study_id, strategy=strategy, costs=costs, analysis=analysis)
    warning = None
    if len(study.changes) > len(before.changes) and study.changes[-1].after_test:
        warning = (
            "Cambiaste parámetros después de ver la prueba: el estudio queda marcado como «prueba "
            "contaminada» de forma permanente. Sus resultados de prueba ya no son una evaluación "
            "fuera de muestra."
        )
    return study_out(study, warning)


@router.post(
    "/studies/{study_id}/test",
    response_model=StudyTestOut,
    summary="Ejecutar la prueba fuera de muestra",
    description=(
        "Ejecuta el segmento de PRUEBA con la configuración congelada del estudio y lo registra "
        "(hash y fecha). Devuelve el backtest con formación, validación y prueba, y el estado de "
        "la prueba: si la configuración cambió después de verla, o si se ejecuta con otra "
        "configuración, la prueba queda contaminada de forma permanente. Si la estrategia está "
        "bloqueada para el par (p. ej. OU no válido), responde 422 y no registra nada."
    ),
    response_description=OK,
    responses=STUDY_RESPONSES,
)
def run_study_test(study_id: str = STUDY_ID, state: AppState = Depends(get_state)) -> StudyTestOut:
    study, result = state.studies.run_test(study_id, state.data)
    out = study_out(study)
    return StudyTestOut(
        study=out,
        backtest=backtest_out(result),
        contaminated=study.contaminated,
        contamination_text=out.contamination_text,
    )


@router.get(
    "/studies/{study_id}/export",
    summary="Exportar un estudio (ZIP)",
    description=(
        "ZIP reproducible del estudio (mismo formato que /api/export). Incluye el segmento de "
        "prueba solo si ya se ejecutó en este estudio, y el LEEME y pruebas_estadisticas.csv "
        "dicen si la prueba está contaminada y por qué, según el registro persistente del "
        "estudio."
    ),
    response_class=Response,
    response_description=ZIP_DESCRIPTION,
    responses={
        200: ZIP_RESPONSES[200],
        **{k: v for k, v in ERROR_RESPONSES.items() if k in (400, 404, 422, 500)},
    },
)
def export_study(
    study_id: str = STUDY_ID,
    excel_es: bool = Query(
        False, description="CSV para Excel en español (separador «;», coma decimal, BOM)."
    ),
    state: AppState = Depends(get_state),
) -> Response:
    study = state.studies.get(study_id)
    result = state.studies.result(study_id, state.data, include_test=study.test_run_here)
    return zip_response(result, excel_es, suffix=f"_estudio_{study.id[:8]}")


@router.delete(
    "/studies/{study_id}",
    status_code=204,
    summary="Eliminar un estudio",
    description=(
        "Quita el estudio de la lista. Si su prueba ya se ejecutó, su registro se conserva en "
        "data/studies/eliminados/: un estudio nuevo del mismo par, con un periodo de prueba que "
        "se solape, sabrá que la prueba ya se vio y nacerá contaminado si usa otra "
        "configuración. Borrar y volver a crear un estudio no «lava» la marca."
    ),
    response_class=Response,
    response_description="Estudio eliminado (sin contenido).",
    responses=STUDY_RESPONSES,
)
def delete_study(study_id: str = STUDY_ID, state: AppState = Depends(get_state)) -> Response:
    state.studies.delete(study_id)
    return Response(status_code=204)
