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

CONTRATO ``analyze_pair(a, b, config, exec_a=None, exec_b=None)``:
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

Precisiones de la implementación:
- ``causes`` son las razones del estado principal, ordenadas por precedencia
  (todas las activas, no solo la que decide). Si no hay ninguna, el estado es ✅
  y ``causes`` contiene una única causa con la decisión de Engle–Granger.
- La validez del OU NO es un estado (la especificación fija cinco): un OU no
  válido bloquea la estrategia y aparece en ``discard_reasons`` con
  ``status=None``, igual que la vida media fuera de rango, ``near_unit_root`` y
  el filtro de correlación.
- ``zscore`` es siempre el z con μ y σ_eq FIJOS de formación. El z móvil
  (``StrategyParams.zscore_mode="movil"``) lo calcula el runner a partir de
  ``spread``: es solo una comparación pedagógica.
- Si con ``frozen_split`` (o al alinear el proxy ejecutable) la formación queda
  por debajo de ``min_train_obs``, el estado es ⛔ con los números.
- Los errores de datos durante los cálculos (p. ej. una serie constante en
  formación) se convierten en una causa ⛔ ``CALCULO_NO_POSIBLE`` en lugar de
  propagarse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mrlab.backtest.split import SampleSplit, SplitConfig, compute_split, segment_slice
from mrlab.data.calendar import time_unit_label
from mrlab.data.validation import (
    AlignedPair,
    Severity,
    ValidationReport,
    clean_series,
    format_int,
    validate_pair,
)
from mrlab.domain.errors import AnalysisBlocked, DataError
from mrlab.domain.models import CalendarType, PriceSeries, Segment, SpreadType
from mrlab.model.diagnostics import (
    OUStability,
    ResidualDiagnostics,
    residual_diagnostics,
    rolling_ou_stability,
)
from mrlab.model.ou import (
    MU_REFERENCE_NOTE,
    PARAMETER_UNCERTAINTY_NOTE,
    OUFit,
    OUForecast,
    OUStatus,
    fit_ou,
    forecast,
    zscore,
)
from mrlab.model.spread import HedgeFit, compute_spread, fit_hedge_ratio, transform
from mrlab.stats.cointegration import (
    Direction,
    EGDecision,
    EGResult,
    EGStability,
    eg_decision,
    engle_granger_both,
    rolling_eg_stability,
)
from mrlab.stats.correlation import log_returns
from mrlab.stats.integration import (
    INTEGRATION_LABELS,
    IntegrationResult,
    format_pvalue,
    integration_tests,
)
from mrlab.stats.multitest import FDR_METHOD_LABELS


class PairStatus(StrEnum):
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

#: Banner permanente de la decisión O5 cuando el par no es ✅ pero el OU es válido.
ILLUSTRATIVE_BANNER = "El par no superó los filtros: resultado solo ilustrativo."

SYNTHETIC_WARNING = (
    "Datos SINTÉTICOS: el par fue generado por el laboratorio con parámetros conocidos; no son "
    "precios históricos."
)

STRATEGY_BLOCKED_NOTE = (
    "Sin un OU válido no hay μ ni σ_eq, así que no se puede calcular el z-score: la estrategia "
    "queda bloqueada para este par."
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
    eg_decision: EGDecision | None = None  # dirección, p ajustados y mensaje de la decisión
    exec_symbol_a: str | None = None  # instrumento ejecutable de cada pierna (None = el mismo)
    exec_symbol_b: str | None = None

    @property
    def dependent_symbol(self) -> str | None:
        return self.hedge.dependent if self.hedge else None

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def blocked(self) -> bool:
        return self.status == PairStatus.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- formato


def _fmt(x: float, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles (convención de la interfaz)."""
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _fmt_p(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else _fmt(p)


def _plural(time_unit: str) -> str:
    return "días" + time_unit[3:] if time_unit.startswith("día") else time_unit


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


# --------------------------------------------------------------------------- estado


def resolve_status(causes: list[Cause]) -> PairStatus:
    """Estado principal según la precedencia de la decisión E6 (sin causas → ✅)."""
    active = {c.status for c in causes if c.status is not None}
    return next((s for s in STATUS_PRECEDENCE if s in active), PairStatus.EVIDENCE)


def _by_precedence(causes: list[Cause]) -> list[Cause]:
    order = {s: i for i, s in enumerate(STATUS_PRECEDENCE)}
    return sorted(causes, key=lambda c: order.get(c.status, len(order)))


# --------------------------------------------------------------------------- piezas


def _executable_prices(
    aligned: AlignedPair,
    series: tuple[PriceSeries, PriceSeries],
    execs: tuple[PriceSeries | None, PriceSeries | None],
    warnings: list[str],
    causes: list[Cause],
) -> tuple[AlignedPair, pd.DataFrame, tuple[str | None, str | None]]:
    """Precios ejecutables (columnas "a", "b") en las mismas fechas que el análisis.

    Sin proxy, cada pierna se ejecuta sobre su propia serie. Con proxy, solo se
    usan las fechas en que también cotiza el proxy (nunca se rellena).
    """
    prices = aligned.prices
    columns: dict[str, pd.Series] = {"a": prices["a"], "b": prices["b"]}
    index = prices.index
    exec_symbols: list[str | None] = []
    for leg, analysis_series, exec_series in zip(("a", "b"), series, execs, strict=True):
        asset = analysis_series.meta.asset
        if exec_series is None:
            exec_symbols.append(None)
            if not asset.executable:
                proxy = (
                    f" El proxy sugerido es {asset.suggested_proxy}."
                    if asset.suggested_proxy
                    else ""
                )
                warnings.append(
                    f"«{asset.symbol}» no se puede comprar ni vender directamente (índice o futuro "
                    f"continuo): el backtest lo opera como si fuera ejecutable, solo como "
                    f"aproximación.{proxy} Elige un instrumento ejecutable para simular con más "
                    "realismo; nunca se sustituye en silencio."
                )
            continue
        exec_symbols.append(exec_series.symbol)
        clean, _ = clean_series(exec_series)
        close = clean.data["close"].astype(float)
        bad = close[~(close > 0)]
        if not bad.empty:
            warnings.append(
                f"El instrumento ejecutable «{exec_series.symbol}» tiene {len(bad)} precio(s) "
                "menores o iguales a cero: esas fechas se excluyen del análisis."
            )
            close = close[close > 0]
        cur_exec = exec_series.meta.asset.currency
        if cur_exec != asset.currency:
            causes.append(
                Cause(
                    code="MONEDA_PROXY_DISTINTA",
                    status=PairStatus.INSUFFICIENT_DATA,
                    message=(
                        f"«{asset.symbol}» cotiza en {asset.currency} y su instrumento ejecutable "
                        f"«{exec_series.symbol}» en {cur_exec}: deben estar en la misma moneda."
                    ),
                )
            )
        warnings.append(
            f"Se analiza «{asset.symbol}» y se opera «{exec_series.symbol}»: la señal y el PnL "
            "provienen de instrumentos distintos (riesgo de base; por ejemplo, un índice de precios "
            "no incluye dividendos y un ETF ajustado sí)."
        )
        index = index.intersection(close.index)
        columns[leg] = close

    dropped = len(prices.index) - len(index)
    if dropped:
        warnings.append(
            f"{format_int(dropped)} fecha(s) comunes del par no tienen precio del instrumento "
            "ejecutable: se excluyen del análisis (no se rellenan)."
        )
    exec_prices = pd.DataFrame({k: v.reindex(index) for k, v in columns.items()}, index=index)
    exec_prices.index.name = "date"
    if dropped:
        aligned = dc_replace(aligned, prices=prices.loc[index])
    return aligned, exec_prices, (exec_symbols[0], exec_symbols[1])


def _integration_cause(res: IntegrationResult) -> Cause:
    return Cause(
        code="SERIE_NO_I1",
        status=PairStatus.DOUBTFUL_ASSUMPTIONS,
        message=(
            f"«{res.symbol}»: {INTEGRATION_LABELS[res.classification].lower()} en formación (no es "
            f"compatible con I(1)). En niveles: ADF {format_pvalue(res.level_adf)}, KPSS "
            f"{format_pvalue(res.level_kpss)}; en diferencias: ADF {format_pvalue(res.diff_adf)}, "
            f"KPSS {format_pvalue(res.diff_kpss)}. Engle–Granger supone que ambas series son I(1), "
            "así que su resultado debe tomarse con cautela."
        ),
    )


def _evidence_cause(decision: EGDecision, results: dict[Direction, EGResult]) -> Cause:
    chosen = results[decision.direction]
    method = FDR_METHOD_LABELS.get(decision.fdr_method, decision.fdr_method)
    p_adj = decision.p_adjusted[decision.direction]
    p_raw = decision.p_raw[decision.direction]
    detail = (
        f"(p original {_fmt_p(p_raw)}; corrección de {method} sobre m = {decision.family_size} "
        "pruebas)"
    )
    where = f"en la dirección elegida ({chosen.dependent} sobre {chosen.independent})"
    if decision.evidence:
        return Cause(
            code="EVIDENCIA_COINTEGRACION",
            status=PairStatus.EVIDENCE,
            message=(
                f"El p-value ajustado de Engle–Granger {where} es {_fmt_p(p_adj)} ≤ "
                f"{_fmt(decision.alpha_level, 2)} {detail}: hay evidencia de cointegración en la "
                "ventana evaluada (formación). No garantiza que la relación se mantenga."
            ),
        )
    if not math.isfinite(chosen.statistic):
        return Cause(
            code="EG_NO_FIABLE",
            status=PairStatus.INSUFFICIENT_EVIDENCE,
            message=f"Engle–Granger no es fiable para este par: {chosen.warning or decision.message}",
        )
    return Cause(
        code="EVIDENCIA_INSUFICIENTE",
        status=PairStatus.INSUFFICIENT_EVIDENCE,
        message=(
            f"El p-value ajustado de Engle–Granger {where} es {_fmt_p(p_adj)} > "
            f"{_fmt(decision.alpha_level, 2)} {detail}: no hay evidencia suficiente de "
            "cointegración en la ventana evaluada."
        ),
    )


def _stability_causes(stability: EGStability) -> list[Cause]:
    if not stability.unstable:
        return []
    causes: list[Cause] = []
    if stability.frac_fail is not None and stability.frac_fail > stability.max_fail_frac:
        causes.append(
            Cause(
                code="VENTANAS_FALLIDAS",
                status=PairStatus.UNSTABLE,
                message=stability.reasons[0],
            )
        )
    if stability.beta_sign_change:
        causes.append(
            Cause(
                code="BETA_CAMBIA_SIGNO",
                status=PairStatus.UNSTABLE,
                message=stability.reasons[-1],
            )
        )
    return causes


def _ou_discards(ou: OUFit, cfg: AnalysisConfig) -> list[Cause]:
    """Razones de descarte que vienen del modelo OU (no cambian el estado del par)."""
    if not ou.valid:
        codes = {
            OUStatus.NON_STATIONARY: "OU_NO_ESTACIONARIO",
            OUStatus.NON_POSITIVE_B: "OU_B_NO_POSITIVO",
            OUStatus.INSUFFICIENT: "OU_DATOS_INSUFICIENTES",
        }
        return [
            Cause(
                code=codes[ou.status],
                status=None,
                message=f"{ou.message} {STRATEGY_BLOCKED_NOTE}",
            )
        ]
    assert ou.half_life is not None and ou.b is not None
    units = _plural(ou.time_unit)
    out: list[Cause] = []
    if ou.half_life > cfg.half_life_max:
        out.append(
            Cause(
                code="VIDA_MEDIA_LARGA",
                status=None,
                message=(
                    f"Vida media de {_fmt(ou.half_life, 1)} {units}, por encima del máximo de "
                    f"{_fmt(cfg.half_life_max, 0)}: la reversión es demasiado lenta para operar con "
                    "este horizonte."
                ),
            )
        )
    if ou.half_life < cfg.half_life_min:
        out.append(
            Cause(
                code="VIDA_MEDIA_CORTA",
                status=None,
                message=(
                    f"Vida media de {_fmt(ou.half_life, 2)} {units}, por debajo del mínimo de "
                    f"{_fmt(cfg.half_life_min, 0)}: con datos diarios no es operable ni fiable."
                ),
            )
        )
    if ou.near_unit_root:
        df_txt = f" (estadístico de Dickey–Fuller {_fmt(ou.df_stat, 2)})" if ou.df_stat else ""
        out.append(
            Cause(
                code="RAIZ_UNITARIA_CERCANA",
                status=None,
                message=(
                    f"El AR(1) del spread tiene b = {_fmt(ou.b)}, que no se distingue "
                    f"estadísticamente de 1 al 5 %{df_txt}: el spread podría comportarse como un "
                    f"paseo aleatorio y la vida media de {_fmt(ou.half_life, 1)} {units} es muy "
                    "imprecisa."
                ),
            )
        )
    return out


def _train_correlation(train: pd.DataFrame) -> float | None:
    """Pearson de log-rendimientos en formación (None si hay precios ≤ 0)."""
    if (train <= 0).to_numpy().any():
        return None
    corr = log_returns(train["a"]).corr(log_returns(train["b"]))
    return float(corr) if np.isfinite(corr) else None


# --------------------------------------------------------------------------- análisis


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
    cfg = config or AnalysisConfig()
    sym_a, sym_b = a.symbol, b.symbol
    report, aligned = validate_pair(
        a, b, cfg.spread_type, min_train_obs=cfg.min_train_obs, train_fraction=cfg.split.train
    )
    out = PairAnalysis(
        config=cfg,
        symbol_a=sym_a,
        symbol_b=sym_b,
        validation=report,
        status=PairStatus.INSUFFICIENT_DATA,
        causes=[],
        discard_reasons=[],
        strategy_allowed=False,
        illustrative_only=True,
        aligned=aligned,
        calendar=report.calendar,
        direction_was_auto=cfg.direction is None,
    )
    warnings: list[str] = []
    if a.is_synthetic or b.is_synthetic:
        warnings.append(SYNTHETIC_WARNING)

    # 1. Bloqueos de la validación (moneda, precios ≤ 0 con log, < 500 en formación).
    if report.blocked or aligned is None:
        blocking = [
            Cause(code=i.code, status=PairStatus.INSUFFICIENT_DATA, message=i.message)
            for i in report.issues
            if i.severity == Severity.ERROR and i.symbol is None
        ] or [
            Cause(
                code="ANALISIS_BLOQUEADO",
                status=PairStatus.INSUFFICIENT_DATA,
                message=report.status_message,
            )
        ]
        return _finish(out, blocking, [], warnings)

    # 2. Precios ejecutables y división de la muestra (fechas congeladas si se pasan).
    causes: list[Cause] = []
    aligned, exec_prices, (exec_sym_a, exec_sym_b) = _executable_prices(
        aligned, (a, b), (exec_a, exec_b), warnings, causes
    )
    out.aligned, out.exec_prices = aligned, exec_prices
    out.exec_symbol_a, out.exec_symbol_b = exec_sym_a, exec_sym_b
    prices = aligned.prices
    if len(prices) == 0:
        causes.append(
            Cause(
                code="DATOS_INSUFICIENTES",
                status=PairStatus.INSUFFICIENT_DATA,
                message="No quedan fechas comunes entre las series de análisis y las ejecutables.",
            )
        )
        return _finish(out, causes, [], warnings)
    split = frozen_split or compute_split(prices.index, cfg.split)
    out.split = split
    train = segment_slice(prices, split, Segment.TRAIN)
    if len(train) < cfg.min_train_obs:
        causes.append(
            Cause(
                code="DATOS_INSUFICIENTES",
                status=PairStatus.INSUFFICIENT_DATA,
                message=(
                    f"Datos insuficientes: la muestra de formación ({split.train_start} a "
                    f"{split.train_end}) tiene {format_int(len(train))} observaciones comunes; se "
                    f"necesitan al menos {format_int(cfg.min_train_obs)}."
                ),
            )
        )
    if any(c.status == PairStatus.INSUFFICIENT_DATA for c in causes):
        return _finish(out, causes, [], warnings)

    # 3–5. Pruebas, spread y OU (solo formación para estimar).
    discards: list[Cause] = []
    try:
        _statistics(out, cfg, prices, train, causes, discards, warnings, family_pvalues)
    except (DataError, AnalysisBlocked) as exc:
        causes.append(
            Cause(
                code="CALCULO_NO_POSIBLE",
                status=PairStatus.INSUFFICIENT_DATA,
                message=f"No se pudo completar el análisis con estos datos: {exc.message}",
            )
        )
    return _finish(out, causes, discards, warnings)


def _statistics(
    out: PairAnalysis,
    cfg: AnalysisConfig,
    prices: pd.DataFrame,
    train: pd.DataFrame,
    causes: list[Cause],
    discards: list[Cause],
    warnings: list[str],
    family_pvalues: list[float] | None,
) -> None:
    sym = {"a": out.symbol_a, "b": out.symbol_b}
    f_train = {leg: transform(train[leg].rename(sym[leg]), cfg.spread_type) for leg in ("a", "b")}

    # Integración de cada serie (decisión E2).
    for leg in ("a", "b"):
        res = integration_tests(f_train[leg], sym[leg], alpha=cfg.alpha_level, maxlag=cfg.maxlag)
        out.integration[sym[leg]] = res
        if res.classification != "compatible_I1":
            causes.append(_integration_cause(res))

    # Engle–Granger en ambas direcciones + FDR + dirección (decisiones E3 y E4).
    results = engle_granger_both(f_train["a"], f_train["b"], sym["a"], sym["b"], maxlag=cfg.maxlag)
    decision = eg_decision(
        results,
        alpha_level=cfg.alpha_level,
        fdr_q=cfg.fdr_q,
        fdr_method=cfg.fdr_method,
        override=cfg.direction,
        family_pvalues=family_pvalues,
    )
    out.eg, out.eg_decision = results, decision
    out.eg_p_adjusted = dict(decision.p_adjusted)
    out.chosen_direction = decision.direction
    out.direction_was_auto = decision.direction_was_auto
    evidence_cause = _evidence_cause(decision, results)
    if evidence_cause.status != PairStatus.EVIDENCE:
        causes.append(evidence_cause)
    for res in results.values():
        if res.warning:
            warnings.append(res.warning)

    dep, indep = ("a", "b") if decision.direction == Direction.A_ON_B else ("b", "a")
    chosen = results[decision.direction]
    out.stability = rolling_eg_stability(
        f_train[dep],
        f_train[indep],
        direction=decision.direction,
        dependent=chosen.dependent,
        independent=chosen.independent,
        window=cfg.stability_window,
        step=cfg.stability_step,
        alpha_level=cfg.alpha_level,
        max_fail_frac=cfg.max_fail_frac,
        maxlag=cfg.maxlag,
    )
    causes.extend(_stability_causes(out.stability))

    # Spread con α y β de formación, aplicado a toda la muestra (sección 5).
    hedge = fit_hedge_ratio(
        train[dep], train[indep], cfg.spread_type, chosen.dependent, chosen.independent
    )
    out.hedge = hedge
    warnings.extend(hedge.warnings)
    spread = compute_spread(prices[dep], prices[indep], hedge)
    out.spread = spread
    spread_train = segment_slice(spread, out.split, Segment.TRAIN)

    # OU con formación (sección 6).
    assert out.calendar is not None
    time_unit = time_unit_label(out.calendar)
    ou = fit_ou(
        spread_train,
        time_unit=time_unit,
        half_life_max=cfg.half_life_max,
        half_life_min=cfg.half_life_min,
    )
    out.ou = ou
    warnings.extend(ou.warnings)
    discards.extend(_ou_discards(ou, cfg))
    if ou.valid:
        out.zscore = zscore(spread, ou)
        out.forecasts = forecast(
            ou, float(spread_train.iloc[-1]), cfg.forecast_horizons, cfg.forecast_level
        )
        out.residuals = residual_diagnostics(spread_train, ou)
        if out.residuals.autocorrelation_detected:
            warnings.append(out.residuals.message)
        out.ou_stability = rolling_ou_stability(
            spread_train, cfg.stability_window, cfg.stability_step, time_unit=time_unit
        )
        warnings.extend([PARAMETER_UNCERTAINTY_NOTE, MU_REFERENCE_NOTE])

    # Filtro exploratorio de correlación (opcional; nunca es una prueba de cointegración).
    out.correlation_train = _train_correlation(train)
    if cfg.correlation_filter and (
        out.correlation_train is None or out.correlation_train < cfg.correlation_min
    ):
        corr_txt = (
            "no se pudo calcular (hay precios ≤ 0)"
            if out.correlation_train is None
            else f"es {_fmt(out.correlation_train, 2)}"
        )
        discards.append(
            Cause(
                code="FILTRO_CORRELACION",
                status=None,
                message=(
                    f"La correlación de los rendimientos logarítmicos en formación {corr_txt}, por "
                    f"debajo del mínimo de {_fmt(cfg.correlation_min, 2)} del filtro exploratorio "
                    "activado. La correlación no es una prueba de cointegración: este filtro solo "
                    "acota la búsqueda."
                ),
            )
        )


def _finish(
    out: PairAnalysis, causes: list[Cause], discards: list[Cause], warnings: list[str]
) -> PairAnalysis:
    status_causes = _by_precedence([c for c in causes if c.status is not None])
    out.status = resolve_status(status_causes)
    if out.status == PairStatus.EVIDENCE and out.eg_decision is not None:
        out.causes = [_evidence_cause(out.eg_decision, out.eg)]
    else:
        out.causes = status_causes
    out.discard_reasons = [
        *[c for c in out.causes if c.status != PairStatus.EVIDENCE],
        *discards,
    ]
    out.strategy_allowed = out.status != PairStatus.INSUFFICIENT_DATA and bool(
        out.ou is not None and out.ou.valid
    )
    out.illustrative_only = out.status != PairStatus.EVIDENCE
    if out.strategy_allowed and out.illustrative_only:
        warnings.append(ILLUSTRATIVE_BANNER)
    out.warnings = _dedupe(warnings)
    return out
