"""Backtest de formación y validación, y comparación entre pares.

CONTRATO:
- POST /backtest  RunConfig (include_test=True → 400 en español: la prueba solo por estudios)
  → BacktestOut {config (fijado), status, status_label, strategy_allowed, illustrative_only,
  illustrative_banner, discard_reasons, segments: {formacion, validacion} → BacktestSegmentOut,
  synthetic, disclaimer, execution_convention}. Si la estrategia no está permitida: segments
  vacío y ``blocked_reason`` en español (no es un error HTTP).
  BacktestSegmentOut {segment, daily: ColumnsOut (columnas DAILY_COLUMNS), trades: list[Trade],
  events: list[Event], metrics: Metrics, metric_help (METRIC_HELP), warnings, max_holding_bars,
  in_sample: bool (True para formación), legs {y: {symbol, asset_class}, x: {...}}, beta,
  spread_type, calendar, periods_per_year, error (si el segmento no se pudo simular)}.
- POST /backtest/compare  CompareRequest {configs: list[RunConfig] (2..10)} → JobAccepted
  {job_id}: trabajo en segundo plano (JobManager) que analiza y simula cada par con su propio
  capital; resultado: list[CompareItem] {label, status, status_label, strategy_allowed,
  illustrative_banner, validation_metrics: Metrics | None, error: ErrorBody | None} — nunca
  ordenado por PnL (orden de entrada). Progreso por par.

Precisiones de la implementación:
- /backtest/compare valida TODAS las configuraciones antes de crear el trabajo (include_test →
  400). Dentro del trabajo, el error de un par (p. ej. el proveedor no responde) queda en su
  ``error`` y los demás pares siguen; un par bloqueado (p. ej. monedas distintas, ⛔) no es un
  error: su fila dice por qué no se simuló.
- El resultado del trabajo es ``CompareResult`` {items, note, synthetic, disclaimer}; cada fila
  lleva su ``index`` (posición en la petición) y la configuración fijada para reproducirla.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from mrlab.analysis.pair_pipeline import Cause, PairStatus
from mrlab.analysis.runner import run_pair
from mrlab.api.dto import (
    BacktestOut,
    RunConfigIn,
    backtest_out,
    blocked_reason,
    core_config,
)
from mrlab.api.errors import json_safe
from mrlab.api.jobs import ProgressCallback
from mrlab.api.openapi_models import EducationalOut
from mrlab.api.routers.analysis import ANALYSIS_RESPONSES, run_without_test, study_only_error
from mrlab.api.schemas import ERROR_RESPONSES, ErrorBody
from mrlab.api.state import AppState, get_state
from mrlab.backtest.metrics import METRIC_HELP, Metrics
from mrlab.data.service import DataService
from mrlab.domain.errors import LabError
from mrlab.domain.models import DataOrigin, Segment
from mrlab.reporting.run_config import RunConfig

router = APIRouter(tags=["backtest"])

OK = "Respuesta correcta."
MIN_COMPARE = 2
MAX_COMPARE = 10
COMPARE_KIND = "comparacion"

COMPARE_NOTE = (
    "Cada par se simula por separado con su propio capital inicial. Las filas siguen el orden de "
    "la petición: nunca se ordenan por PnL, porque ordenar por resultado retrospectivo invita a "
    "elegir el par que mejor salió por azar. Las métricas son de VALIDACIÓN (fuera de la muestra "
    "de formación); la prueba solo se ejecuta dentro de un estudio."
)


@router.post(
    "/backtest",
    response_model=BacktestOut,
    summary="Backtest de formación y validación",
    description=(
        "Simula la estrategia en FORMACIÓN (dentro de muestra, solo ilustrativa) y en VALIDACIÓN, "
        "cada segmento por separado, con el capital inicial y los parámetros fijos de formación. "
        "La señal se calcula con el cierre de t y se ejecuta al cierre de t+1. Devuelve la serie "
        "diaria (posiciones, PnL bruto, costes por tipo, PnL neto, capital, exposición y "
        "apalancamiento), las operaciones, los eventos y las métricas con su ayuda. Si la "
        "estrategia está bloqueada (p. ej. OU no válido o índice sin proxy ejecutable), no hay "
        "segmentos y «blocked_reason» explica por qué. include_test = true responde 400: la "
        "prueba solo se ejecuta en un estudio."
    ),
    response_description=OK,
    responses=ANALYSIS_RESPONSES,
)
def backtest(cfg: RunConfigIn, state: AppState = Depends(get_state)) -> BacktestOut:
    return backtest_out(run_without_test(cfg, "El backtest", state))


# --------------------------------------------------------------------------- comparación


class CompareRequest(BaseModel):
    """Pares a comparar, cada uno con su configuración completa (y su propio capital)."""

    model_config = ConfigDict(extra="forbid")

    configs: list[RunConfigIn] = Field(
        min_length=MIN_COMPARE,
        max_length=MAX_COMPARE,
        description=(
            f"Configuraciones a comparar (de {MIN_COMPARE} a {MAX_COMPARE}). Cada par usa su "
            "propio capital inicial (strategy.initial_capital)."
        ),
    )


class JobAccepted(BaseModel):
    """Trabajo creado: consulta su progreso y su resultado en GET /api/jobs/{job_id}."""

    job_id: str = Field(description="Identificador del trabajo.")
    status: str = Field(description="Estado inicial del trabajo («pendiente»).")
    status_url: str = Field(description="Ruta para consultar el progreso y el resultado.")
    message: str = Field(description="Explicación en español.")


class CompareItem(BaseModel):
    """Resultado de un par de la comparación (orden de la petición)."""

    index: int = Field(description="Posición del par en la petición (desde 0).")
    label: str = Field(description="Par, p. ej. «KO / PEP» (con el proxy ejecutable si lo hay).")
    symbol_a: str = Field(description="Símbolo de la serie A.")
    symbol_b: str = Field(description="Símbolo de la serie B.")
    status: PairStatus | None = Field(None, description="Estado del par (null si hubo un error).")
    status_label: str | None = Field(None, description="Etiqueta del estado en español.")
    strategy_allowed: bool = Field(False, description="True si se pudo simular la estrategia.")
    illustrative_only: bool = Field(True, description="True si el resultado es solo ilustrativo.")
    illustrative_banner: str | None = Field(None, description="Banner «solo ilustrativo».")
    discard_reasons: list[Cause] = Field(
        default_factory=list, description="Panel «¿Por qué se descartó este par?»."
    )
    blocked_reason: str | None = Field(None, description="Por qué no se simuló (si aplica).")
    initial_capital: float = Field(description="Capital inicial de ESTE par.")
    validation_metrics: Metrics | None = Field(
        None, description="Métricas del segmento de VALIDACIÓN (null si no se simuló)."
    )
    validation_error: str | None = Field(
        None, description="Por qué no se pudo simular la validación (si aplica)."
    )
    synthetic: bool = Field(False, description="True si el par usa datos SINTÉTICOS.")
    data_origin: DataOrigin | None = Field(None, description="HISTORICO o SINTETICO.")
    config: RunConfig | None = Field(
        None, description="Configuración fijada (para reproducir este par con /backtest)."
    )
    error: ErrorBody | None = Field(
        None, description="Error en español si el par no se pudo analizar (los demás siguen)."
    )


class CompareResult(EducationalOut):
    """Resultado del trabajo de comparación (en GET /api/jobs/{id} → result)."""

    items: list[CompareItem] = Field(description="Un resultado por par, en el orden de entrada.")
    metric_help: dict[str, str] = Field(
        default_factory=lambda: dict(METRIC_HELP), description="Ayuda de una línea por métrica."
    )
    note: str = Field(COMPARE_NOTE, description="Cómo leer la comparación.")


def _label(config: RunConfig) -> str:
    pair = config.pair
    a = pair.a.symbol + (f" (opera {pair.exec_a.symbol})" if pair.exec_a else "")
    b = pair.b.symbol + (f" (opera {pair.exec_b.symbol})" if pair.exec_b else "")
    return f"{a} / {b}"


def _compare_item(index: int, config: RunConfig, data: DataService) -> CompareItem:
    base: dict[str, Any] = {
        "index": index,
        "label": _label(config),
        "symbol_a": config.pair.a.symbol,
        "symbol_b": config.pair.b.symbol,
        "initial_capital": config.strategy.initial_capital,
    }
    try:
        result = run_pair(config, data, include_test=False)
    except LabError as exc:
        return CompareItem(
            **base,
            error=ErrorBody(code=exc.code, message=exc.message, details=json_safe(exc.details)),
        )
    an = result.analysis
    validation = result.backtests.get(Segment.VALIDATION)
    return CompareItem(
        **base,
        status=an.status,
        status_label=an.status_label,
        strategy_allowed=an.strategy_allowed,
        illustrative_only=an.illustrative_only,
        illustrative_banner=an.illustrative_banner,
        discard_reasons=list(an.discard_reasons),
        blocked_reason=blocked_reason(an),
        validation_metrics=validation.metrics if validation is not None else None,
        validation_error=result.segment_errors.get(Segment.VALIDATION),
        synthetic=result.is_synthetic,
        data_origin=result.data_origin,
        config=result.config,
    )


def compare_job(
    configs: list[RunConfig], data: DataService
) -> Callable[[ProgressCallback], CompareResult]:
    """Función del trabajo: un par tras otro, con progreso; nunca reordena."""

    def run(report: ProgressCallback) -> CompareResult:
        items: list[CompareItem] = []
        n = len(configs)
        for i, config in enumerate(configs):
            report(i / n, f"Analizando el par {i + 1} de {n}: {_label(config)}.")
            items.append(_compare_item(i, config, data))
        report(1.0, f"Comparación terminada: {n} pares.")
        return CompareResult(items=items, synthetic=any(it.synthetic for it in items))

    return run


@router.post(
    "/backtest/compare",
    response_model=JobAccepted,
    status_code=202,
    summary="Comparar pares (en segundo plano)",
    description=(
        "Crea un trabajo en segundo plano que analiza y simula cada par con su propio capital "
        "(formación y validación; nunca la prueba). Consulta el progreso y el resultado en "
        "GET /api/jobs/{job_id}: el resultado conserva el orden de la petición y nunca se ordena "
        "por PnL. El error de un par (en español) no detiene a los demás."
    ),
    response_description="Trabajo creado.",
    responses={k: v for k, v in ERROR_RESPONSES.items() if k in (400, 422, 500)},
)
def compare(req: CompareRequest, state: AppState = Depends(get_state)) -> JobAccepted:
    for i, cfg in enumerate(req.configs):
        if cfg.include_test:
            exc = study_only_error("La comparación de pares")
            exc.details["indice"] = i
            raise exc
    configs = [core_config(cfg) for cfg in req.configs]
    title = "Comparación: " + "; ".join(_label(c) for c in configs)
    job = state.jobs.submit(COMPARE_KIND, compare_job(configs, state.data), title=title)
    return JobAccepted(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/jobs/{job.id}",
        message=(
            f"Comparación de {len(configs)} pares en curso. Consulta el progreso en "
            f"/api/jobs/{job.id}."
        ),
    )
