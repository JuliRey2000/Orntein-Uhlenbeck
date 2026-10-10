"""Orquestación del análisis de un par: validar → pruebas → spread → OU → estado y causas.

Estados posibles (4.5), EXACTAMENTE estos, con la causa explicada:
  ✅ Evidencia de cointegración en la ventana evaluada
  ⚠️ Evidencia insuficiente
  ⚠️ Supuestos dudosos
  ⚠️ Resultado inestable entre ventanas
  ⛔ Datos insuficientes   (también cubre bloqueos por datos no aptos: monedas
                            distintas o precios ≤ 0 con spread logarítmico)
Precedencia del estado principal: ⛔ > supuestos dudosos > evidencia
insuficiente > inestable > ✅. Se listan TODAS las causas activas.
Nunca usar "cointegración garantizada" ni "par rentable".

CONTRATO (a implementar) ``analyze_pair(a, b, config, exec_a=None, exec_b=None)``:
 1. ``validate_pair`` (tipo de spread, mínimo 500 en formación). Bloqueado →
    PairAnalysis con status INSUFFICIENT_DATA, causas, y el resto en None.
 2. ``compute_split`` sobre las fechas alineadas (o ``config.frozen_split`` si se
    pasa: entonces se respetan esas fechas).
 3. Transformar (log/precio). Con SOLO formación: ADF/KPSS de cada serie,
    Engle–Granger en ambas direcciones, BH sobre las 2 direcciones (o sobre la
    familia que pase el llamador en ``family_pvalues`` para universos), elección
    de dirección (o la fijada), estabilidad por ventanas (dirección elegida).
 4. Spread: ``fit_hedge_ratio`` con formación (dependiente = A de la dirección
    elegida), ``compute_spread`` sobre TODA la muestra con α, β fijos.
 5. OU con formación: ``fit_ou``; si válido → z con μ, σ_eq fijos sobre toda la
    muestra (o z móvil si zscore_mode="movil", solo pedagógico), pronósticos a
    1, 5, 20 desde el último dato de formación, diagnóstico de residuos,
    estabilidad del OU por ventanas.
 6. Estado + causas + ``discard_reasons`` (para el panel "¿Por qué se descartó
    este par?"): p ajustado > α, inestabilidad, b fuera de (0,1), vida media
    fuera de rango (>126 o <1), datos insuficientes, supuestos dudosos,
    filtro de correlación (si se activó y no se cumple).
 7. ``strategy_allowed`` = no bloqueado y OU válido. ``illustrative_only`` =
    estado distinto de ✅ (el backtest se permite con aviso "solo ilustrativo").
 8. Si se pasan exec_a/exec_b (proxy ejecutable para índices), se alinean con
    las series de análisis; si no, se ejecuta sobre las mismas series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

from mrlab.backtest.split import SampleSplit, SplitConfig
from mrlab.data.validation import AlignedPair, ValidationReport
from mrlab.domain.models import CalendarType, PriceSeries, SpreadType
from mrlab.model.diagnostics import OUStability, ResidualDiagnostics
from mrlab.model.ou import OUFit, OUForecast
from mrlab.model.spread import HedgeFit
from mrlab.stats.cointegration import Direction, EGResult, EGStability
from mrlab.stats.integration import IntegrationResult


class PairStatus(str, Enum):
    EVIDENCE = "evidencia"
    INSUFFICIENT_EVIDENCE = "evidencia_insuficiente"
    DOUBTFUL_ASSUMPTIONS = "supuestos_dudosos"
    UNSTABLE = "inestable"
    INSUFFICIENT_DATA = "datos_insuficientes"


STATUS_LABELS = {
    PairStatus.EVIDENCE: "✅ Evidencia de cointegración en la ventana evaluada",
    PairStatus.INSUFFICIENT_EVIDENCE: "⚠️ Evidencia insuficiente",
    PairStatus.DOUBTFUL_ASSUMPTIONS: "⚠️ Supuestos dudosos",
    PairStatus.UNSTABLE: "⚠️ Resultado inestable entre ventanas",
    PairStatus.INSUFFICIENT_DATA: "⛔ Datos insuficientes",
}

STATUS_PRECEDENCE = (
    PairStatus.INSUFFICIENT_DATA,
    PairStatus.DOUBTFUL_ASSUMPTIONS,
    PairStatus.INSUFFICIENT_EVIDENCE,
    PairStatus.UNSTABLE,
    PairStatus.EVIDENCE,
)


class AnalysisConfig(BaseModel):
    spread_type: SpreadType = SpreadType.LOG
    alpha_level: float = Field(0.05, gt=0, lt=1)
    fdr_q: float = Field(0.05, gt=0, lt=1)
    fdr_method: str = "bh"
    maxlag: int = 12
    direction: Direction | None = None  # None = automática (menor p-value)
    split: SplitConfig = SplitConfig()
    min_train_obs: int = 500
    stability_window: int = 252
    stability_step: int = 63
    max_fail_frac: float = 0.40
    half_life_max: float = 126.0
    half_life_min: float = 1.0
    correlation_filter: bool = False
    correlation_min: float = 0.5
    forecast_horizons: tuple[int, ...] = (1, 5, 20)
    forecast_level: float = 0.95


class Cause(BaseModel):
    code: str
    status: PairStatus | None  # estado que provoca (None = solo informativa/descartes)
    message: str


@dataclass
class PairAnalysis:
    config: AnalysisConfig
    symbol_a: str
    symbol_b: str
    validation: ValidationReport
    status: PairStatus
    causes: list[Cause]
    discard_reasons: list[Cause]
    strategy_allowed: bool
    illustrative_only: bool
    aligned: AlignedPair | None = None
    exec_prices: pd.DataFrame | None = None  # columnas "a", "b" ejecutables, mismas fechas
    calendar: CalendarType | None = None
    split: SampleSplit | None = None
    integration: dict[str, IntegrationResult] = field(default_factory=dict)
    eg: dict[Direction, EGResult] = field(default_factory=dict)
    eg_p_adjusted: dict[Direction, float] = field(default_factory=dict)
    chosen_direction: Direction | None = None
    direction_was_auto: bool = True
    stability: EGStability | None = None
    hedge: HedgeFit | None = None
    spread: pd.Series | None = None  # toda la muestra, α y β fijos de formación
    ou: OUFit | None = None
    zscore: pd.Series | None = None
    forecasts: list[OUForecast] = field(default_factory=list)
    residuals: ResidualDiagnostics | None = None
    ou_stability: OUStability | None = None
    correlation_train: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def dependent_symbol(self) -> str | None:
        return self.hedge.dependent if self.hedge else None


def analyze_pair(
    a: PriceSeries,
    b: PriceSeries,
    config: AnalysisConfig | None = None,
    *,
    exec_a: PriceSeries | None = None,
    exec_b: PriceSeries | None = None,
    frozen_split: SampleSplit | None = None,
    family_pvalues: list[float] | None = None,
) -> PairAnalysis:
    raise NotImplementedError
