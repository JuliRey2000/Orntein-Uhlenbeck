"""Análisis: correlaciones y análisis completo de un par (cointegración, spread, OU).

CONTRATO:
- POST /correlations  CorrelationRequest {series: list[SeriesRef] (2..30), window: int=63,
  pair: [i, j] | None (índices del par para la correlación móvil, los rendimientos y la
  dispersión; por defecto el primero con el segundo)} → CorrelationOut {symbols,
  pearson: list[list[float|None]], spearman, nobs: list[list[int]], normalized: {dates,
  series: {symbol: values}} (base 100 en la intersección), rolling: SeriesOut, returns:
  {dates, a, b} (log-rendimientos del par sobre fechas comunes), concepts (textos de los tres
  conceptos), note (la correlación es un filtro exploratorio, nunca prueba de cointegración),
  synthetic, disclaimer}.
- POST /pairs/analyze  RunConfig (include_test debe ser False; si es True → 400 en español que
  remite a /api/studies) → AnalysisOut {config: RunConfig (fijado: snapshot_id y split),
  analysis: PairAnalysisOut, synthetic, disclaimer}.
  PairAnalysisOut serializa PairAnalysis (ver ``mrlab.api.dto``): bands y zscore son None si el
  OU NO es válido, y el OU no válido no tiene λ, μ, σ, σ_eq ni vida media (sección 17).

Precisiones de la implementación:
- /pairs/analyze usa ``run_pair`` (sin la prueba), igual que /backtest: así el panel «¿Por qué se
  descartó?» incluye también las razones que solo se conocen al simular la validación (costes que
  superan el PnL bruto, segmento no simulable; decisión O5a) y el análisis nunca contradice al
  backtest de la misma configuración.
- Las series para gráficos terminan al final de la validación: el periodo de prueba solo se
  muestra al ejecutar la prueba dentro de un estudio (``chart_note`` lo explica).
- /correlations carga cada serie como /data/validate (caché o proveedor; con ``snapshot_id`` solo
  caché). Nunca rellena fechas: cada par usa la intersección de sus fechas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mrlab.analysis.runner import RunResult, run_pair
from mrlab.api.dto import AnalysisOut, RunConfigIn, analysis_out, core_config
from mrlab.api.errors import RequestValueError
from mrlab.api.openapi_models import EducationalOut, SeriesMetaOut
from mrlab.api.routers.data import SeriesQueryIn, load_ref
from mrlab.api.schemas import ERROR_RESPONSES, ErrorResponse, SeriesOut, clean_float, series_out
from mrlab.api.state import AppState, get_state
from mrlab.data.calendar import CLOSE_TIME_MISMATCH_WARNING
from mrlab.domain.errors import ConfigError
from mrlab.domain.models import AssetClass, DataOrigin, PriceSeries
from mrlab.reporting.run_config import RunConfig
from mrlab.stats.correlation import (
    CONCEPTS,
    correlation_matrix,
    normalized_prices,
    rolling_correlation,
)

router = APIRouter(tags=["análisis"])

OK = "Respuesta correcta."
MAX_CORRELATION_SERIES = 30
MAX_ROLLING_WINDOW = 2520

CORRELATION_NOTE = (
    "La correlación es un filtro exploratorio opcional, nunca una prueba de cointegración: dos "
    "activos pueden estar muy correlacionados y no estar cointegrados (y al revés). La "
    "cointegración se evalúa con Engle–Granger en /api/pairs/analyze."
)

ANALYSIS_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    429: {"model": ErrorResponse, "description": "El proveedor limitó las peticiones"},
    503: {"model": ErrorResponse, "description": "Sin conexión con el proveedor de datos"},
}


def study_only_error(where: str) -> ConfigError:
    """Error 400 cuando se pide la prueba fuera de un estudio (sección 8.1)."""
    return ConfigError(
        f"{where} no ejecuta el periodo de prueba (include_test debe ser false). La prueba fuera "
        "de muestra se ejecuta una sola vez y con la configuración congelada, dentro de un "
        "estudio: crea uno con POST /api/studies y ejecútala con POST /api/studies/{id}/test.",
        code="PRUEBA_SOLO_EN_ESTUDIO",
        details={"include_test": True, "ruta_estudios": "/api/studies"},
    )


def run_without_test(cfg: RunConfigIn, where: str, state: AppState) -> RunResult:
    """``run_pair`` sin la prueba; ``include_test=True`` → 400 PRUEBA_SOLO_EN_ESTUDIO."""
    if cfg.include_test:
        raise study_only_error(where)
    config: RunConfig = core_config(cfg)
    return run_pair(config, state.data, include_test=False)


# --------------------------------------------------------------------------- correlaciones


class CorrelationRequest(BaseModel):
    """Series a comparar y ventana de la correlación móvil."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "series": [
                        {
                            "provider": "yahoo",
                            "symbol": "KO",
                            "start": "2016-01-04",
                            "end": "2021-12-31",
                        },
                        {
                            "provider": "yahoo",
                            "symbol": "PEP",
                            "start": "2016-01-04",
                            "end": "2021-12-31",
                        },
                    ],
                    "window": 63,
                }
            ]
        },
    )

    series: list[SeriesQueryIn] = Field(
        min_length=2,
        max_length=MAX_CORRELATION_SERIES,
        description=f"Series a comparar (de 2 a {MAX_CORRELATION_SERIES}).",
    )
    window: int = Field(
        63,
        ge=2,
        le=MAX_ROLLING_WINDOW,
        description="Ventana de la correlación móvil, en observaciones comunes (por defecto 63).",
    )
    pair: list[int] | None = Field(
        None,
        min_length=2,
        max_length=2,
        description=(
            "Índices (desde 0) de las dos series para la correlación móvil, los rendimientos y la "
            "dispersión. Por defecto [0, 1]."
        ),
    )

    @model_validator(mode="after")
    def _check_pair(self) -> CorrelationRequest:
        if self.pair is not None:
            i, j = self.pair
            n = len(self.series)
            if not (0 <= i < n and 0 <= j < n):
                raise RequestValueError(
                    f"los índices del par deben estar entre 0 y {n - 1} (se recibió [{i}, {j}])"
                )
            if i == j:
                raise RequestValueError("el par debe tener dos series distintas")
        return self


class NormalizedOut(BaseModel):
    """Precios normalizados (base 100) sobre las fechas comunes."""

    dates: list[str] = Field(description="Fechas comunes a todas las series (intersección).")
    series: dict[str, list[float | None]] = Field(
        description="Precios en base 100 desde la primera fecha común, por serie."
    )


class ReturnsOut(BaseModel):
    """Log-rendimientos del par elegido (para el gráfico de dispersión)."""

    symbol_a: str = Field(description="Primera serie del par elegido.")
    symbol_b: str = Field(description="Segunda serie del par elegido.")
    dates: list[str] = Field(description="Fechas de los rendimientos (fechas comunes del par).")
    a: list[float | None] = Field(description="Log-rendimientos de la primera serie.")
    b: list[float | None] = Field(description="Log-rendimientos de la segunda serie.")


class CorrelationOut(EducationalOut):
    """Explorador de relaciones (sección 3): correlación de log-rendimientos."""

    data_origin: DataOrigin = Field(description="HISTORICO o SINTETICO (si alguna serie lo es).")
    symbols: list[str] = Field(description="Etiqueta de cada serie (filas y columnas).")
    pearson: list[list[float | None]] = Field(
        description="Matriz de Pearson de log-rendimientos (cada par sobre sus fechas comunes)."
    )
    spearman: list[list[float | None]] = Field(description="Matriz de Spearman.")
    nobs: list[list[int]] = Field(description="Rendimientos usados en cada celda.")
    pair: list[int] = Field(description="Índices del par elegido.")
    pair_symbols: list[str] = Field(description="Etiquetas del par elegido.")
    window: int = Field(description="Ventana de la correlación móvil.")
    normalized: NormalizedOut = Field(description="Precios normalizados (base 100).")
    rolling: SeriesOut = Field(
        description="Correlación móvil (Pearson, solo hacia atrás; null hasta completar la ventana)."
    )
    returns: ReturnsOut = Field(description="Log-rendimientos del par (para la dispersión).")
    concepts: dict[str, str] = Field(
        description="Correlación, cointegración y aptitud para OU: tres conceptos distintos."
    )
    note: str = Field(description="La correlación no es una prueba de cointegración.")
    warnings: list[str] = Field(default_factory=list, description="Avisos (calendarios, etc.).")
    series_meta: list[SeriesMetaOut] = Field(
        description="Metadatos de cada serie, con snapshot_id."
    )


def _labels(series: list[PriceSeries]) -> list[str]:
    """Símbolo de cada serie; si se repite, con el proveedor y, si aún se repite, con su número."""
    symbols = [s.symbol for s in series]
    labels = [
        f"{s.symbol} ({s.meta.provider})" if symbols.count(s.symbol) > 1 else s.symbol
        for s in series
    ]
    return [lab if labels.count(lab) == 1 else f"{lab} #{i + 1}" for i, lab in enumerate(labels)]


def _matrix(frame: Any, labels: list[str], as_int: bool = False) -> list[list[Any]]:
    if as_int:
        return [[int(frame.loc[r, c]) for c in labels] for r in labels]
    return [[clean_float(frame.loc[r, c]) for c in labels] for r in labels]


@router.post(
    "/correlations",
    response_model=CorrelationOut,
    summary="Correlaciones de rendimientos",
    description=(
        "Matriz de correlación de log-rendimientos (Pearson y Spearman) con el número de "
        "observaciones de cada celda, precios normalizados (base 100), correlación móvil y "
        "rendimientos del par elegido. Cada par de series usa la intersección de sus fechas "
        "(nunca se rellenan fechas). Es un filtro exploratorio: nunca una prueba de "
        "cointegración."
    ),
    response_description=OK,
    responses=ANALYSIS_RESPONSES,
)
def correlations(req: CorrelationRequest, state: AppState = Depends(get_state)) -> CorrelationOut:
    loaded = [load_ref(state.data, ref, f"#{i + 1}") for i, ref in enumerate(req.series)]
    labels = _labels(loaded)
    closes = {lab: s.data["close"].rename(lab) for lab, s in zip(labels, loaded, strict=True)}
    matrix = correlation_matrix(closes)
    i, j = req.pair or [0, 1]
    la, lb = labels[i], labels[j]
    norm = normalized_prices(closes)
    rolling = rolling_correlation(closes[la], closes[lb], req.window)
    # Log-rendimientos del par sobre SUS fechas comunes (la base 100 no cambia los rendimientos).
    returns = np.log(normalized_prices({la: closes[la], lb: closes[lb]})).diff().iloc[1:]
    ra, rb = returns[la], returns[lb]
    classes = {s.meta.asset.asset_class for s in loaded}
    warnings = []
    if AssetClass.CRYPTO in classes and classes != {AssetClass.CRYPTO}:
        warnings.append(CLOSE_TIME_MISMATCH_WARNING)
    synthetic = any(s.is_synthetic for s in loaded)
    return CorrelationOut(
        synthetic=synthetic,
        data_origin=DataOrigin.SYNTHETIC if synthetic else DataOrigin.HISTORICAL,
        symbols=labels,
        pearson=_matrix(matrix.pearson, labels),
        spearman=_matrix(matrix.spearman, labels),
        nobs=_matrix(matrix.nobs, labels, as_int=True),
        pair=[i, j],
        pair_symbols=[la, lb],
        window=req.window,
        normalized=NormalizedOut(
            dates=[t.date().isoformat() for t in norm.index],
            series={lab: [clean_float(v) for v in norm[lab].to_numpy()] for lab in labels},
        ),
        rolling=series_out(rolling),  # type: ignore[arg-type]
        returns=ReturnsOut(
            symbol_a=la,
            symbol_b=lb,
            dates=[t.date().isoformat() for t in ra.index],
            a=[clean_float(v) for v in ra.to_numpy()],
            b=[clean_float(v) for v in rb.to_numpy()],
        ),
        concepts=dict(CONCEPTS),
        note=CORRELATION_NOTE,
        warnings=warnings,
        series_meta=[s.meta for s in loaded],  # type: ignore[misc]
    )


# --------------------------------------------------------------------------- análisis del par


@router.post(
    "/pairs/analyze",
    response_model=AnalysisOut,
    summary="Analizar un par",
    description=(
        "Análisis completo de un par con la configuración dada: validación de datos, ADF/KPSS, "
        "Engle–Granger en ambas direcciones con corrección por pruebas múltiples, estabilidad "
        "por ventanas, hedge ratio por MCO (solo formación), spread y ajuste OU, con el estado "
        "del par (uno de cinco) y el panel «¿Por qué se descartó?». Con un OU no válido la "
        "respuesta no incluye λ, vida media, bandas ni z. La configuración devuelta queda fijada "
        "(instantáneas y fechas de corte) para reproducir el análisis. No ejecuta el periodo de "
        "prueba: include_test = true responde 400 (la prueba solo se ejecuta en un estudio)."
    ),
    response_description=OK,
    responses=ANALYSIS_RESPONSES,
)
def analyze(cfg: RunConfigIn, state: AppState = Depends(get_state)) -> AnalysisOut:
    return analysis_out(run_without_test(cfg, "El análisis de un par", state))
