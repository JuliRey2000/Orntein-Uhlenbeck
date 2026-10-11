"""Exportación reproducible y reproducción desde run_config.json (sección 15).

CONTRATO:
- POST /export  ExportRequest {config: RunConfig, excel_es: bool=False} → application/zip
  (export_zip_bytes) con Content-Disposition: attachment; filename="<base>.zip" (prefijo
  SINTETICO_ si hay datos sintéticos). include_test=True → 400 (la prueba se exporta desde el
  estudio). Misma configuración + misma caché → mismos bytes.
- POST /runs/replay  multipart file=run_config.json (+ excel_es) → ZIP reproducido EXACTAMENTE
  (solo caché, nunca descarga: si falta una instantánea → 404 SNAPSHOT_NO_ENCONTRADO en
  español). Un run_config con include_test=True se reproduce tal cual (es una reproducción, no
  una nueva evaluación) y conserva su test_history / marca de contaminación.
- POST /runs/replay/json  mismo, pero devuelve AnalysisOut + BacktestOut en JSON (para la UI).
- Helper público ``zip_response(result, excel_es) -> Response`` reutilizable por studies.py.

Precisiones de la implementación:
- Reproducir exige que TODAS las series del run_config.json estén fijadas a una instantánea
  (``snapshot_id``): sin ella habría que descargar, y reproducir nunca descarga en silencio
  (400 RUN_CONFIG_SIN_INSTANTANEAS).
- Un run_config.json con ``include_test = true`` solo se reproduce si su propio hash figura en su
  ``test_history`` (es decir, si registra que la prueba ya se ejecutó con ESA configuración). Si
  no (p. ej. un archivo escrito a mano, sin hash, con otros umbrales), reproducirlo sería una
  evaluación nueva de la prueba fuera de un estudio → 400 PRUEBA_SOLO_EN_ESTUDIO.
- El archivo se lee con un límite de tamaño (``max_upload_mb``) y se verifica como
  ``load_run_config`` (``parse_run_config``): un archivo editado después de exportarlo se rechaza
  (RUN_CONFIG_HASH_DISTINTO).
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from mrlab.analysis.runner import RunResult, run_pair
from mrlab.api.dto import AnalysisOut, BacktestOut, RunConfigIn, analysis_out, backtest_out
from mrlab.api.routers.analysis import ANALYSIS_RESPONSES, run_without_test
from mrlab.api.routers.data import UploadLimitRoute, max_upload_bytes, read_limited
from mrlab.api.schemas import ErrorResponse
from mrlab.api.state import AppState, get_state
from mrlab.domain.errors import ConfigError
from mrlab.reporting.export import export_basename, export_zip_bytes
from mrlab.reporting.run_config import RunConfig, parse_run_config

router = APIRouter(tags=["exportación"])
#: Rutas con archivo: el tamaño del cuerpo se limita antes de recibirlo entero (ver data.py).
_upload_router = APIRouter(route_class=UploadLimitRoute)

ZIP_MEDIA_TYPE = "application/zip"
ZIP_DESCRIPTION = (
    "Archivo ZIP con operaciones.csv, diario.csv, pruebas_estadisticas.csv, run_config.json y "
    "LEEME.txt (prefijo SINTETICO_ en todos si los datos son sintéticos)."
)
ZIP_RESPONSES = {
    200: {
        "content": {ZIP_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
        "description": ZIP_DESCRIPTION,
    },
    **{k: v for k, v in ANALYSIS_RESPONSES.items() if k != 200},
}
#: Reproducir: solo caché (sin errores del proveedor) y un archivo con tamaño máximo.
REPLAY_RESPONSES = {
    **{k: v for k, v in ZIP_RESPONSES.items() if k not in (429, 502, 503)},
    413: {"model": ErrorResponse, "description": "El archivo supera el tamaño máximo"},
}


def zip_response(result: RunResult, excel_es: bool = False, suffix: str = "") -> Response:
    """ZIP determinista de la exportación como descarga (``<base><suffix>.zip``)."""
    name = f"{export_basename(result)}{suffix}.zip"
    ascii_name = name.encode("ascii", "replace").decode("ascii").replace("?", "_").replace('"', "")
    disposition = f'attachment; filename="{ascii_name}"'
    if ascii_name != name:
        disposition += f"; filename*=UTF-8''{quote(name)}"
    return Response(
        content=export_zip_bytes(result, excel_es),
        media_type=ZIP_MEDIA_TYPE,
        headers={"Content-Disposition": disposition},
    )


# --------------------------------------------------------------------------- exportar


class ExportRequest(BaseModel):
    """Configuración a exportar y formato de los CSV."""

    model_config = ConfigDict(extra="forbid")

    config: RunConfigIn = Field(description="Configuración (include_test debe ser false).")
    excel_es: bool = Field(
        False,
        description=(
            "CSV para Excel en español: separador «;», coma decimal y UTF-8 con BOM. Por defecto, "
            "separador «,» y punto decimal."
        ),
    )


@router.post(
    "/export",
    summary="Exportar un análisis (ZIP)",
    description=(
        "Ejecuta el análisis y el backtest de formación y validación con la configuración dada y "
        "devuelve un ZIP reproducible: operaciones, serie diaria, pruebas estadísticas, "
        "run_config.json (con su hash y las instantáneas de datos usadas) y LEEME.txt. La misma "
        "configuración con la misma caché produce exactamente los mismos bytes. Cada fila indica "
        "si los datos son históricos o SINTÉTICOS. La prueba no se exporta desde aquí "
        "(include_test = true → 400): se exporta desde su estudio."
    ),
    response_class=Response,
    response_description=ZIP_DESCRIPTION,
    responses=ZIP_RESPONSES,
)
def export(req: ExportRequest, state: AppState = Depends(get_state)) -> Response:
    result = run_without_test(req.config, "La exportación", state)
    return zip_response(result, req.excel_es)


# --------------------------------------------------------------------------- reproducir


def _unpinned(config: RunConfig) -> list[str]:
    refs = {
        "a": config.pair.a,
        "b": config.pair.b,
        "exec_a": config.pair.exec_a,
        "exec_b": config.pair.exec_b,
    }
    return [
        f"{ref.symbol} ({role})"
        for role, ref in refs.items()
        if ref is not None and not ref.snapshot_id
    ]


def _replay(file: UploadFile, state: AppState) -> RunResult:
    settings = state.settings
    content = read_limited(file.file, max_upload_bytes(settings), settings)
    config = parse_run_config(content, file.filename or "run_config.json")
    missing = _unpinned(config)
    if missing:
        raise ConfigError(
            "El run_config.json no fija la instantánea de datos (snapshot_id) de: "
            f"{', '.join(missing)}. Reproducir exige exactamente los mismos precios y el "
            "laboratorio nunca vuelve a descargar en silencio. Usa el run_config.json que exportó "
            "el laboratorio, o analiza el par con /api/pairs/analyze para obtener una "
            "configuración fijada.",
            code="RUN_CONFIG_SIN_INSTANTANEAS",
            details={"series": missing},
        )
    if config.include_test and config.config_hash not in config.test_history:
        raise ConfigError(
            "Este run_config.json pide el periodo de prueba, pero su historial (test_history) no "
            "registra ninguna ejecución de la prueba con esta misma configuración: reproducirlo "
            "sería una evaluación nueva de la prueba fuera de un estudio. La prueba se ejecuta "
            "una sola vez, dentro de un estudio (POST /api/studies y POST "
            "/api/studies/{id}/test).",
            code="PRUEBA_SOLO_EN_ESTUDIO",
            details={"include_test": True, "config_hash": config.config_hash},
        )
    return run_pair(config, state.data)


REPLAY_FILE = File(
    ...,
    description="run_config.json exportado por el laboratorio (con o sin prefijo SINTETICO_).",
)
REPLAY_EXCEL = Form(
    False, description="CSV para Excel en español (separador «;», coma decimal, BOM)."
)


@_upload_router.post(
    "/runs/replay",
    summary="Reproducir un run_config.json (ZIP)",
    description=(
        "Reproduce exactamente un análisis exportado a partir de su run_config.json, usando SOLO "
        "las instantáneas de la caché local (nunca descarga): con la misma caché, el ZIP es "
        "idéntico byte a byte al original. Si falta una instantánea responde 404 "
        "(SNAPSHOT_NO_ENCONTRADO). Un archivo modificado después de exportarlo se rechaza. Si el "
        "run_config.json incluía la prueba ya ejecutada, se reproduce tal cual (no es una nueva "
        "evaluación) y conserva su marca de prueba contaminada."
    ),
    response_class=Response,
    response_description=ZIP_DESCRIPTION,
    responses=REPLAY_RESPONSES,
)
def replay(
    file: UploadFile = REPLAY_FILE,
    excel_es: bool = REPLAY_EXCEL,
    state: AppState = Depends(get_state),
) -> Response:
    return zip_response(_replay(file, state), excel_es)


class ReplayOut(BaseModel):
    """Reproducción de un run_config.json en JSON (para la interfaz)."""

    analysis: AnalysisOut = Field(description="Análisis del par reproducido.")
    backtest: BacktestOut = Field(description="Backtest reproducido (con la prueba si la incluía).")
    test_included: bool = Field(description="True si el run_config.json incluía la prueba.")
    test_contaminated: bool | None = Field(
        None, description="Prueba contaminada según el historial del archivo (null si no aplica)."
    )


@_upload_router.post(
    "/runs/replay/json",
    response_model=ReplayOut,
    summary="Reproducir un run_config.json (JSON)",
    description=(
        "Igual que /runs/replay, pero devuelve el análisis y el backtest en JSON para mostrarlos "
        "en la interfaz. Solo usa la caché local (nunca descarga)."
    ),
    response_description="Respuesta correcta.",
    responses={k: v for k, v in REPLAY_RESPONSES.items() if k != 200},
)
def replay_json(
    file: UploadFile = REPLAY_FILE,
    state: AppState = Depends(get_state),
) -> ReplayOut:
    result = _replay(file, state)
    return ReplayOut(
        analysis=analysis_out(result),
        backtest=backtest_out(result),
        test_included=result.test_included,
        test_contaminated=result.test_contaminated,
    )


router.include_router(_upload_router)
