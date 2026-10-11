"""Simulación SINTÉTICA del OU (modo educativo, sección 6.3).

CONTRATO:
- POST /ou/simulate  SimulateRequest {lam, mu, sigma, x0, horizon (≤ MAX_SIM_HORIZON),
  n_paths (1..500), seed, paths_returned: int = 50 (máx. trayectorias devueltas para dibujar)}
  → SimulationOut: el resumen de mrlab.model.ou.simulate_paths_summary calculado con TODAS las
  trayectorias, pero ``paths`` recortado a paths_returned (con n_paths_returned y una nota en
  español); label "SINTÉTICO", synthetic=True, disclaimer. Además, el histograma del tiempo de
  cruce: crossing_hist {bins, counts} (bins = bordes; len(bins) = len(counts) + 1) y
  n_not_crossed.
- Límites: n_paths > 500, horizonte fuera de [1, MAX_SIM_HORIZON], λ o σ ≤ 0 o valores no
  finitos → 422 PETICION_INVALIDA en español; X₀ = μ → 400 X0_IGUAL_A_MU (del núcleo).
- La simulación NUNCA toca datos históricos ni la caché. Nunca devuelve NaN ni infinitos.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from mrlab.api.openapi_models import EducationalOut
from mrlab.api.schemas import ERROR_RESPONSES, clean_float
from mrlab.model.ou import MAX_SIM_HORIZON, MAX_SIM_PATHS, TIME_UNIT_DEFAULT, simulate_paths_summary

router = APIRouter(tags=["simulación"])

OK = "Respuesta correcta."
SYNTHETIC_LABEL = "SINTÉTICO"
DEFAULT_PATHS_RETURNED = 50
#: Número máximo de intervalos del histograma de tiempos de cruce.
MAX_HIST_BINS = 40


class SimulateRequest(BaseModel):
    """Parámetros de la simulación SINTÉTICA de un proceso OU."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "lam": 0.05,
                    "mu": 0.0,
                    "sigma": 0.02,
                    "x0": 0.1,
                    "horizon": 252,
                    "n_paths": 200,
                    "seed": 42,
                    "paths_returned": 50,
                }
            ]
        },
    )

    lam: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="λ: velocidad de reversión (1/día de negociación). Vida media = ln(2)/λ.",
    )
    mu: float = Field(0.0, allow_inf_nan=False, description="μ: nivel de referencia del proceso.")
    sigma: float = Field(
        gt=0, allow_inf_nan=False, description="σ: volatilidad (unidades del spread por √día)."
    )
    x0: float = Field(
        allow_inf_nan=False,
        description="X₀: valor inicial. Debe ser distinto de μ (se mide el tiempo hasta cruzar μ).",
    )
    horizon: int = Field(
        252,
        ge=1,
        le=MAX_SIM_HORIZON,
        description=f"Horizonte en días de negociación (de 1 a {MAX_SIM_HORIZON}).",
    )
    n_paths: int = Field(
        200,
        ge=1,
        le=MAX_SIM_PATHS,
        description=f"Trayectorias simuladas (de 1 a {MAX_SIM_PATHS}); todas entran en las estadísticas.",
    )
    seed: int = Field(
        42,
        ge=0,
        le=2**63 - 1,
        description="Semilla (entero ≥ 0): misma semilla, mismas trayectorias.",
    )
    paths_returned: int = Field(
        DEFAULT_PATHS_RETURNED,
        ge=0,
        le=MAX_SIM_PATHS,
        description=(
            "Trayectorias que se devuelven para dibujar (como máximo n_paths). No cambia las "
            "estadísticas, que usan todas las trayectorias."
        ),
    )


class HistogramOut(BaseModel):
    """Histograma: número de trayectorias según el día en que cruzaron μ por primera vez."""

    bins: list[float] = Field(
        description="Bordes de los intervalos (len(bins) = len(counts) + 1), en días."
    )
    counts: list[int] = Field(description="Trayectorias que cruzaron μ en cada intervalo.")


class SimulationOut(EducationalOut):
    """Simulación SINTÉTICA de un proceso OU con la transición exacta."""

    synthetic: bool = Field(True, description="Siempre True: la simulación es SINTÉTICA.")
    label: str = Field(SYNTHETIC_LABEL, description="Etiqueta permanente: «SINTÉTICO».")
    lam: float = Field(description="λ usado.")
    mu: float = Field(description="μ usado.")
    sigma: float = Field(description="σ usado.")
    x0: float = Field(description="X₀ usado.")
    horizon: int = Field(description="Horizonte en días de negociación.")
    n_paths: int = Field(description="Trayectorias simuladas (todas usadas en las estadísticas).")
    seed: int = Field(description="Semilla.")
    time_unit: str = Field(TIME_UNIT_DEFAULT, description="Unidad de tiempo.")
    sigma_eq: float | None = Field(description="σ_eq = σ/√(2λ): desviación estacionaria.")
    half_life_theoretical: float | None = Field(description="Vida media teórica ln(2)/λ.")
    half_life_empirical: float | None = Field(
        description=(
            "Vida media empírica: primer h en que la media de las trayectorias cae a la mitad de "
            "(X₀ − μ); null si no ocurre dentro del horizonte."
        )
    )
    time_grid: list[float | None] = Field(description="Instantes 0..horizonte.")
    paths: list[list[float | None]] = Field(
        description="Trayectorias devueltas para dibujar (las primeras n_paths_returned)."
    )
    n_paths_returned: int = Field(description="Número de trayectorias devueltas en «paths».")
    paths_note: str = Field(description="Explicación del recorte de trayectorias.")
    mean_empirical: list[float | None] = Field(description="Media de TODAS las trayectorias.")
    mean_theoretical: list[float | None] = Field(description="E[X_t] = μ + (X₀ − μ)e^{−λt}.")
    band_lower_theoretical: list[float | None] = Field(
        description="Media condicional − 2 desviaciones estándar condicionales."
    )
    band_upper_theoretical: list[float | None] = Field(
        description="Media condicional + 2 desviaciones estándar condicionales."
    )
    stationary_lower: list[float | None] = Field(description="μ − 2·σ_eq.")
    stationary_upper: list[float | None] = Field(description="μ + 2·σ_eq.")
    crossing_times: list[float | None] = Field(
        description="Tiempo hasta cruzar μ por trayectoria (null si no cruzó)."
    )
    crossing_time_median: float | None = Field(
        description="Mediana del tiempo de cruce (null si la mitad o más no cruzó)."
    )
    crossing_hist: HistogramOut = Field(description="Histograma del tiempo hasta cruzar μ.")
    n_not_crossed: int = Field(description="Trayectorias que no cruzaron μ dentro del horizonte.")
    notes: list[str] = Field(description="Notas en español (sesgo del cruce discreto, bandas…).")


def _clean_list(values: list[Any]) -> list[float | None]:
    return [clean_float(v) for v in values]


def crossing_histogram(times: list[float | None]) -> HistogramOut:
    """Histograma de los tiempos de cruce (enteros ≥ 1) con intervalos alineados a días."""
    crossed = np.array([t for t in times if t is not None and math.isfinite(t)], dtype=float)
    if crossed.size == 0:
        return HistogramOut(bins=[], counts=[])
    top = int(math.ceil(crossed.max()))
    width = max(1, math.ceil(top / MAX_HIST_BINS))
    n_bins = math.ceil(top / width)
    edges = 0.5 + width * np.arange(n_bins + 1)
    counts, _ = np.histogram(crossed, bins=edges)
    return HistogramOut(bins=[float(e) for e in edges], counts=[int(c) for c in counts])


@router.post(
    "/ou/simulate",
    response_model=SimulationOut,
    summary="Simular un proceso OU (SINTÉTICO)",
    description=(
        "Simula trayectorias de un proceso Ornstein–Uhlenbeck con la transición EXACTA (no "
        "Euler): trayectorias, media y bandas teóricas, histograma del tiempo hasta cruzar μ y "
        "vida media teórica frente a empírica. Todo se calcula con las n_paths trayectorias; solo "
        "se devuelven paths_returned para dibujar. Es un modo educativo: los datos son "
        "SINTÉTICOS y nunca modifican datos históricos ni la caché."
    ),
    response_description=OK,
    responses={code: ERROR_RESPONSES[code] for code in (400, 422, 500)},
)
def simulate(req: SimulateRequest) -> SimulationOut:
    summary = simulate_paths_summary(
        lam=req.lam,
        mu=req.mu,
        sigma=req.sigma,
        x0=req.x0,
        horizon=req.horizon,
        n_paths=req.n_paths,
        seed=req.seed,
    )
    n_returned = min(req.paths_returned, req.n_paths)
    if n_returned < req.n_paths:
        paths_note = (
            f"Se muestran {n_returned} de {req.n_paths} trayectorias; las medias, las bandas, el "
            f"histograma y la vida media empírica usan las {req.n_paths}."
        )
    else:
        paths_note = f"Se muestran las {req.n_paths} trayectorias simuladas."
    return SimulationOut(
        lam=summary.lam,
        mu=summary.mu,
        sigma=summary.sigma,
        x0=summary.x0,
        horizon=summary.horizon,
        n_paths=summary.n_paths,
        seed=summary.seed,
        sigma_eq=clean_float(summary.sigma_eq),
        half_life_theoretical=clean_float(summary.half_life_theoretical),
        half_life_empirical=clean_float(summary.half_life_empirical),
        time_grid=_clean_list(summary.time_grid),
        paths=[_clean_list(p) for p in summary.paths[:n_returned]],
        n_paths_returned=n_returned,
        paths_note=paths_note,
        mean_empirical=_clean_list(summary.mean_empirical),
        mean_theoretical=_clean_list(summary.mean_theoretical),
        band_lower_theoretical=_clean_list(summary.band_lower_theoretical),
        band_upper_theoretical=_clean_list(summary.band_upper_theoretical),
        stationary_lower=_clean_list(summary.stationary_lower),
        stationary_upper=_clean_list(summary.stationary_upper),
        crossing_times=_clean_list(summary.crossing_times),
        crossing_time_median=clean_float(summary.crossing_time_median),
        crossing_hist=crossing_histogram(summary.crossing_times),
        n_not_crossed=summary.n_not_crossed,
        notes=[*summary.notes, paths_note],
        synthetic=True,
        label=summary.label or SYNTHETIC_LABEL,
    )
