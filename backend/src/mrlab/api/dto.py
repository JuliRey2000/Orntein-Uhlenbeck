"""Serialización de los resultados del núcleo para la API (DTO reutilizables por varios routers).

- ``pair_analysis_out(analysis, show_test_period)`` → ``PairAnalysisOut``.
- ``backtest_segment_out(bt)`` → ``BacktestSegmentOut``.
- ``analysis_out(result)`` → ``AnalysisOut``; ``backtest_out(result)`` → ``BacktestOut``.

Reglas:
- Nunca NaN/±inf: las series pasan por ``series_out``/``frame_columns`` y los diccionarios libres
  (``details`` de incidencias y eventos) por ``json_safe``. Además, FastAPI serializa con
  pydantic, que escribe ``null`` para cualquier float no finito que quede en un modelo.
- Con un OU NO válido la respuesta no tiene λ, μ, σ, σ_eq, vida media, bandas ni z: el OU se
  serializa como ``OUInvalidOut``, que no tiene esos campos (prohibición de la sección 17).
- Toda respuesta con resultados lleva ``disclaimer`` y ``synthetic`` (y ``data_origin``).
- Las series para gráficos (precios, spread, z) terminan al final de la VALIDACIÓN si la
  ejecución no incluye la prueba: el periodo de prueba no se muestra hasta ejecutarla en un
  estudio (se informa en ``chart_note``). Las estimaciones no cambian: siempre usan solo la
  formación.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mrlab.analysis.pair_pipeline import (
    STATUS_LABELS,
    STRATEGY_BLOCKING_CODES,
    Cause,
    PairAnalysis,
    PairStatus,
)
from mrlab.analysis.runner import RunResult
from mrlab.api.schemas import (
    ColumnsOut,
    Educational,
    SeriesOut,
    clean_float,
    frame_columns,
    series_out,
)
from mrlab.backtest.engine import DAILY_COLUMNS, EXECUTION_CONVENTION, BacktestResult, Event, Trade
from mrlab.backtest.metrics import METRIC_HELP, Metrics
from mrlab.backtest.split import SampleSplit
from mrlab.data.calendar import periods_per_year, time_unit_label
from mrlab.data.validation import ValidationIssue, ValidationReport
from mrlab.domain.models import AssetClass, CalendarType, DataOrigin, Segment, SpreadType
from mrlab.model.diagnostics import OUStability, ResidualDiagnostics
from mrlab.model.ou import BANDS_VS_INTERVALS_NOTE, OUFit, OUForecast, OUStatus
from mrlab.model.spread import HedgeFit
from mrlab.reporting.run_config import RunConfig
from mrlab.stats.cointegration import (
    DIRECTION_ORDER,
    Direction,
    EGDecision,
    EGResult,
    EGStability,
)
from mrlab.stats.integration import IntegrationResult

SEGMENT_LABELS = {
    Segment.TRAIN: "formación",
    Segment.VALIDATION: "validación",
    Segment.TEST: "prueba",
}
SEGMENT_ORDER = (Segment.TRAIN, Segment.VALIDATION, Segment.TEST)

IN_SAMPLE_NOTE = (
    "Formación: dentro de muestra (los parámetros se estimaron con estos mismos datos); el "
    "resultado es solo ilustrativo."
)
BLOCKED_PREFIX = "La simulación de la estrategia está bloqueada para este par."


# --------------------------------------------------------------------------- utilidades


def json_safe(value: Any) -> Any:
    """Convierte un valor arbitrario en algo JSON válido: NaN/±inf → None, numpy → Python,
    fechas → ISO, enums → su valor, modelos → dict."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return clean_float(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    return str(value)


def _plural(time_unit: str) -> str:
    return "días" + time_unit[3:] if time_unit.startswith("día") else time_unit


def _trim(obj: pd.Series | pd.DataFrame | None, end: date | None):
    if obj is None or end is None:
        return obj
    return obj.loc[: pd.Timestamp(end)]


def _safe_issue(issue: ValidationIssue) -> ValidationIssue:
    return issue.model_copy(update={"details": json_safe(issue.details)})


def _safe_report(report: ValidationReport) -> ValidationReport:
    return report.model_copy(
        update={
            "issues": [_safe_issue(i) for i in report.issues],
            "series": [
                s.model_copy(update={"issues": [_safe_issue(i) for i in s.issues]})
                for s in report.series
            ],
        }
    )


# --------------------------------------------------------------------------- modelos


class PricesOut(BaseModel):
    """Precios alineados de las dos piernas (formato columnar, mismas fechas)."""

    symbol_a: str = Field(description="Símbolo de la pierna A")
    symbol_b: str = Field(description="Símbolo de la pierna B")
    dates: list[str] = Field(description="Fechas ISO (AAAA-MM-DD)")
    a: list[float | None] = Field(description="Precios de A")
    b: list[float | None] = Field(description="Precios de B")


class BandsOut(BaseModel):
    """Bandas estacionarias del OU: μ ± 1 y ± 2 σ_eq (solo con un OU válido)."""

    mu: float = Field(description="μ: referencia estadística del modelo (nunca «valor justo»)")
    sigma_eq: float = Field(description="Desviación estacionaria σ_eq = σ/√(2λ)")
    upper1: float = Field(description="μ + 1·σ_eq")
    lower1: float = Field(description="μ − 1·σ_eq")
    upper2: float = Field(description="μ + 2·σ_eq")
    lower2: float = Field(description="μ − 2·σ_eq")
    note: str = Field(BANDS_VS_INTERVALS_NOTE, description="Bandas frente a intervalos predictivos")


class OUUnitsOut(BaseModel):
    """Unidades de los parámetros del OU (Δt = 1 observación)."""

    time_unit: str = Field(description="Unidad de tiempo, p. ej. «día de negociación»")
    lam: str = Field(description="Unidad de λ (velocidad de reversión)")
    half_life: str = Field(description="Unidad de la vida media")
    sigma: str = Field(description="Unidad de σ (volatilidad instantánea)")
    sigma_eq: str = Field(description="Unidad de σ_eq (desviación estacionaria)")
    mu: str = Field(description="Unidad de μ (nivel del spread)")


class _OUCommon(BaseModel):
    nobs: int = Field(description="Transiciones usadas en el AR(1) de formación")
    dt: float = Field(description="Δt en observaciones")
    time_unit: str = Field(description="Unidad de tiempo")
    a: float | None = Field(None, description="Constante del AR(1) X_{t+1} = a + b·X_t + ε")
    b: float | None = Field(None, description="Coeficiente del AR(1)")
    a_se: float | None = Field(None, description="Error estándar de a")
    b_se: float | None = Field(None, description="Error estándar de b")
    s2: float | None = Field(None, description="Varianza residual del AR(1)")
    message: str = Field(description="Interpretación en español")
    warnings: list[str] = Field(default_factory=list, description="Avisos del ajuste")
    df_stat: float | None = Field(None, description="Estadístico de Dickey–Fuller aumentado")
    df_crit: float | None = Field(None, description="Valor crítico de MacKinnon al nivel df_alpha")
    df_pvalue: float | None = Field(None, description="p-value de MacKinnon")
    df_lags: int | None = Field(None, description="Rezagos de las diferencias (AIC)")
    df_alpha: float = Field(description="Nivel de la prueba de raíz unitaria")
    df_estimated_residual: bool = Field(
        description="True: valores críticos de residuo estimado (como Engle–Granger)"
    )
    near_unit_root: bool = Field(description="0 < b < 1 pero b no se distingue de 1")


class OUValidOut(_OUCommon):
    """Ajuste OU válido: 0 < b < 1 y la prueba ADF rechaza la raíz unitaria (decisión O7)."""

    status: Literal["valido"] = Field(description="Estado del ajuste")
    valid: Literal[True] = True
    lam: float = Field(description="λ: velocidad de reversión")
    mu: float = Field(description="μ: referencia estadística del modelo")
    sigma: float = Field(description="σ: volatilidad instantánea")
    sigma_eq: float = Field(description="σ_eq = σ/√(2λ)")
    half_life: float = Field(description="Vida media = ln 2/λ")
    mu_se: float | None = Field(None, description="Error estándar asintótico de μ")
    units: OUUnitsOut = Field(description="Unidades de cada parámetro")


class OUInvalidOut(_OUCommon):
    """Ajuste OU NO válido: sin λ, μ, σ, σ_eq, vida media ni bandas (sección 17)."""

    status: Literal["no_estacionario", "b_no_positivo", "datos_insuficientes"] = Field(
        description="Motivo por el que no es un OU válido"
    )
    valid: Literal[False] = False


OUOut = Annotated[OUValidOut | OUInvalidOut, Field(discriminator="status")]


class PairAnalysisOut(BaseModel):
    """Análisis completo de un par: estado, causas, pruebas, spread y OU."""

    symbol_a: str = Field(description="Símbolo de la serie A analizada")
    symbol_b: str = Field(description="Símbolo de la serie B analizada")
    exec_symbol_a: str | None = Field(
        None, description="Instrumento ejecutable de A (proxy); null = se opera la propia serie"
    )
    exec_symbol_b: str | None = Field(
        None, description="Instrumento ejecutable de B (proxy); null = se opera la propia serie"
    )
    status: PairStatus = Field(description="Estado del par (uno de exactamente cinco)")
    status_label: str = Field(description="Etiqueta del estado en español, con su símbolo")
    causes: list[Cause] = Field(description="Todas las causas del estado, por precedencia")
    discard_reasons: list[Cause] = Field(description="Panel «¿Por qué se descartó este par?»")
    strategy_allowed: bool = Field(description="True si se puede simular la estrategia")
    illustrative_only: bool = Field(description="True si el resultado es solo ilustrativo")
    illustrative_banner: str | None = Field(None, description="Banner «solo ilustrativo»")
    illustrative_reasons: list[str] = Field(description="Motivos del banner")
    validation: ValidationReport = Field(description="Informe de validación de datos")
    calendar: CalendarType | None = Field(None, description="Calendario efectivo")
    time_unit: str | None = Field(None, description="Unidad de tiempo del OU")
    periods_per_year: int | None = Field(None, description="Periodos por año para anualizar")
    split: SampleSplit | None = Field(None, description="Fechas de corte congeladas 60/20/20")
    integration: dict[str, IntegrationResult] = Field(
        default_factory=dict, description="ADF/KPSS por símbolo (niveles y diferencias)"
    )
    eg: list[EGResult] = Field(default_factory=list, description="Engle–Granger por dirección")
    eg_p_adjusted: dict[str, float | None] = Field(
        default_factory=dict, description="p-values ajustados (FDR) por dirección"
    )
    eg_decision: EGDecision | None = Field(None, description="Decisión de Engle–Granger")
    chosen_direction: Direction | None = Field(None, description="Dirección elegida")
    direction_was_auto: bool = Field(description="True si la dirección fue la de menor p-value")
    dependent_symbol: str | None = Field(None, description="Variable dependiente del spread")
    stability: EGStability | None = Field(None, description="Estabilidad por ventanas")
    hedge: HedgeFit | None = Field(None, description="α, β por MCO (solo formación)")
    ou: OUOut | None = Field(None, description="Ajuste OU (formación)")
    forecasts: list[OUForecast] = Field(
        default_factory=list, description="Pronósticos a 1, 5 y 20 desde el fin de formación"
    )
    residuals: ResidualDiagnostics | None = Field(None, description="Diagnóstico de residuos")
    ou_stability: OUStability | None = Field(None, description="Estabilidad de λ, μ, σ")
    correlation_train: float | None = Field(
        None, description="Pearson de log-rendimientos en formación (solo exploratorio)"
    )
    warnings: list[str] = Field(default_factory=list, description="Avisos del análisis")
    prices: PricesOut | None = Field(None, description="Precios analizados, alineados")
    exec_prices: PricesOut | None = Field(
        None, description="Precios de los instrumentos ejecutables (si hay proxy)"
    )
    spread: SeriesOut | None = Field(None, description="Spread con α, β fijos de formación")
    spread_units: str | None = Field(None, description="Unidades del spread")
    zscore: SeriesOut | None = Field(
        None, description="z con μ y σ_eq fijos de formación; null si el OU no es válido"
    )
    bands: BandsOut | None = Field(None, description="Bandas del OU; null si no es válido")
    chart_end: str | None = Field(None, description="Última fecha de las series para gráficos")
    chart_note: str | None = Field(None, description="Por qué las series terminan en chart_end")


class AnalysisOut(Educational):
    """Respuesta de /pairs/analyze."""

    data_origin: DataOrigin = Field(description="HISTORICO o SINTETICO")
    config: RunConfig = Field(description="Configuración fijada (instantáneas y división)")
    analysis: PairAnalysisOut


class LegOut(BaseModel):
    symbol: str = Field(description="Instrumento operado")
    asset_class: AssetClass = Field(description="Clase de activo (decide el slippage)")


class BacktestSegmentOut(BaseModel):
    """Backtest de un segmento (formación, validación o prueba)."""

    segment: Segment = Field(description="formacion, validacion o prueba")
    label: str = Field(description="Nombre del segmento en español")
    in_sample: bool = Field(description="True para formación (dentro de muestra, ilustrativo)")
    start: str | None = Field(None, description="Primera fecha simulada")
    end: str | None = Field(None, description="Última fecha simulada")
    daily: ColumnsOut | None = Field(
        None, description=f"Serie diaria en columnas: {', '.join(DAILY_COLUMNS)}"
    )
    trades: list[Trade] = Field(default_factory=list, description="Operaciones")
    events: list[Event] = Field(default_factory=list, description="Eventos del motor")
    metrics: Metrics | None = Field(None, description="Métricas del segmento")
    metric_help: dict[str, str] = Field(
        default_factory=lambda: dict(METRIC_HELP), description="Ayuda de una línea por métrica"
    )
    warnings: list[str] = Field(default_factory=list, description="Avisos del backtest")
    max_holding_bars: int | None = Field(None, description="Tiempo máximo, en observaciones")
    legs: dict[str, LegOut] | None = Field(None, description="Piernas y (dependiente) y x")
    beta: float | None = Field(None, description="β del spread (fijo de formación)")
    spread_type: SpreadType | None = Field(None, description="log o precio")
    calendar: CalendarType | None = Field(None, description="Calendario efectivo")
    periods_per_year: int | None = Field(None, description="Periodos por año (252 o 365)")
    error: str | None = Field(None, description="Por qué no se pudo simular el segmento")


class BacktestOut(Educational):
    """Respuesta de /backtest (y del segmento de prueba de un estudio)."""

    data_origin: DataOrigin = Field(description="HISTORICO o SINTETICO")
    config: RunConfig = Field(description="Configuración fijada (instantáneas y división)")
    symbol_a: str
    symbol_b: str
    exec_symbol_a: str | None = None
    exec_symbol_b: str | None = None
    status: PairStatus = Field(description="Estado del par")
    status_label: str = Field(description="Etiqueta del estado en español")
    strategy_allowed: bool = Field(description="True si se puede simular la estrategia")
    illustrative_only: bool = Field(description="True si el resultado es solo ilustrativo")
    illustrative_banner: str | None = Field(None, description="Banner «solo ilustrativo»")
    illustrative_reasons: list[str] = Field(default_factory=list)
    discard_reasons: list[Cause] = Field(default_factory=list)
    segments: dict[str, BacktestSegmentOut] = Field(
        default_factory=dict, description="Backtest por segmento (formacion, validacion, prueba)"
    )
    blocked_reason: str | None = Field(
        None, description="Por qué no se simuló (si la estrategia está bloqueada)"
    )
    execution_convention: str = Field(EXECUTION_CONVENTION, description="Convención t → t+1")
    in_sample_note: str = Field(IN_SAMPLE_NOTE, description="Nota sobre la formación")
    test_included: bool = Field(description="True si se simuló el segmento de prueba")
    test_contaminated: bool | None = Field(
        None, description="Prueba contaminada (null si la prueba no se simuló)"
    )


# --------------------------------------------------------------------------- constructores


def blocked_reason(analysis: PairAnalysis) -> str | None:
    """Texto en español de por qué no se simula la estrategia (None si se permite)."""
    if analysis.strategy_allowed:
        return None
    if analysis.status == PairStatus.INSUFFICIENT_DATA:
        reasons = [c.message for c in analysis.causes]
    else:
        reasons = [d.message for d in analysis.discard_reasons if d.code in STRATEGY_BLOCKING_CODES]
    return " ".join([BLOCKED_PREFIX, *reasons])


def _prices_out(frame: pd.DataFrame | None, sym_a: str, sym_b: str) -> PricesOut | None:
    if frame is None or frame.empty:
        return None
    return PricesOut(
        symbol_a=sym_a,
        symbol_b=sym_b,
        dates=[pd.Timestamp(t).date().isoformat() for t in frame.index],
        a=[clean_float(v) for v in frame["a"].to_numpy()],
        b=[clean_float(v) for v in frame["b"].to_numpy()],
    )


def _ou_out(ou: OUFit, spread_units: str | None) -> OUValidOut | OUInvalidOut:
    common = {
        "nobs": ou.nobs,
        "dt": ou.dt,
        "time_unit": ou.time_unit,
        "a": clean_float(ou.a),
        "b": clean_float(ou.b),
        "a_se": clean_float(ou.a_se),
        "b_se": clean_float(ou.b_se),
        "s2": clean_float(ou.s2),
        "message": ou.message,
        "warnings": list(ou.warnings),
        "df_stat": clean_float(ou.df_stat),
        "df_crit": clean_float(ou.df_crit),
        "df_pvalue": clean_float(ou.df_pvalue),
        "df_lags": ou.df_lags,
        "df_alpha": ou.df_alpha,
        "df_estimated_residual": ou.df_estimated_residual,
        "near_unit_root": ou.near_unit_root,
    }
    params = (ou.lam, ou.mu, ou.sigma, ou.sigma_eq, ou.half_life)
    if not (ou.valid and all(p is not None and math.isfinite(p) for p in params)):
        status = ou.status.value if ou.status != OUStatus.VALID else "datos_insuficientes"
        return OUInvalidOut(status=status, **common)
    units = spread_units or "unidades del spread"
    tu = ou.time_unit
    return OUValidOut(
        status="valido",
        lam=ou.lam,
        mu=ou.mu,
        sigma=ou.sigma,
        sigma_eq=ou.sigma_eq,
        half_life=ou.half_life,
        mu_se=clean_float(ou.mu_se),
        units=OUUnitsOut(
            time_unit=tu,
            lam=f"1/{tu}",
            half_life=_plural(tu),
            sigma=f"{units} por raíz de {tu}",
            sigma_eq=units,
            mu=units,
        ),
        **common,
    )


def _chart_end(analysis: PairAnalysis, show_test_period: bool) -> date | None:
    split = analysis.split
    if show_test_period or split is None or split.n_test == 0 or split.test_start is None:
        return None
    return split.validation_end or split.train_end


def pair_analysis_out(analysis: PairAnalysis, show_test_period: bool = False) -> PairAnalysisOut:
    """PairAnalysis → PairAnalysisOut. Sin el periodo de prueba en los gráficos salvo que la
    ejecución lo incluya (``show_test_period``)."""
    an = analysis
    ou_valid = an.ou is not None and an.ou.valid
    spread_units = an.hedge.units if an.hedge else None
    end = _chart_end(an, show_test_period)
    chart_note = None
    if end is not None and an.split is not None:
        chart_note = (
            f"Las series de los gráficos terminan el {end.isoformat()}, al final de la validación: "
            f"el periodo de prueba ({an.split.test_start} a {an.split.test_end}) solo se muestra "
            "al ejecutar la prueba dentro de un estudio. Las estimaciones usan solo la formación."
        )
    aligned = an.aligned.prices if an.aligned is not None else None
    exec_frame = an.exec_prices if (an.exec_symbol_a or an.exec_symbol_b) else None
    bands = None
    zscore = None
    ou_out = _ou_out(an.ou, spread_units) if an.ou is not None else None
    if ou_valid and isinstance(ou_out, OUValidOut):
        mu, sd = ou_out.mu, ou_out.sigma_eq
        bands = BandsOut(
            mu=mu,
            sigma_eq=sd,
            upper1=mu + sd,
            lower1=mu - sd,
            upper2=mu + 2 * sd,
            lower2=mu - 2 * sd,
        )
        zscore = series_out(_trim(an.zscore, end))
    calendar = an.calendar
    return PairAnalysisOut(
        symbol_a=an.symbol_a,
        symbol_b=an.symbol_b,
        exec_symbol_a=an.exec_symbol_a,
        exec_symbol_b=an.exec_symbol_b,
        status=an.status,
        status_label=an.status_label,
        causes=list(an.causes),
        discard_reasons=list(an.discard_reasons),
        strategy_allowed=an.strategy_allowed,
        illustrative_only=an.illustrative_only,
        illustrative_banner=an.illustrative_banner,
        illustrative_reasons=list(an.illustrative_reasons),
        validation=_safe_report(an.validation),
        calendar=calendar,
        time_unit=time_unit_label(calendar) if calendar else None,
        periods_per_year=periods_per_year(calendar) if calendar else None,
        split=an.split,
        integration=dict(an.integration),
        eg=[an.eg[d] for d in DIRECTION_ORDER if d in an.eg],
        eg_p_adjusted={d.value: clean_float(p) for d, p in an.eg_p_adjusted.items()},
        eg_decision=an.eg_decision,
        chosen_direction=an.chosen_direction,
        direction_was_auto=an.direction_was_auto,
        dependent_symbol=an.dependent_symbol,
        stability=an.stability,
        hedge=an.hedge,
        ou=ou_out,
        forecasts=list(an.forecasts) if ou_valid else [],
        residuals=an.residuals if ou_valid else None,
        ou_stability=an.ou_stability if ou_valid else None,
        correlation_train=clean_float(an.correlation_train),
        warnings=list(an.warnings),
        prices=_prices_out(_trim(aligned, end), an.symbol_a, an.symbol_b),
        exec_prices=_prices_out(
            _trim(exec_frame, end),
            an.exec_symbol_a or an.symbol_a,
            an.exec_symbol_b or an.symbol_b,
        ),
        spread=series_out(_trim(an.spread, end)),
        spread_units=spread_units,
        zscore=zscore,
        bands=bands,
        chart_end=end.isoformat() if end else None,
        chart_note=chart_note,
    )


def backtest_segment_out(bt: BacktestResult) -> BacktestSegmentOut:
    daily = bt.daily
    dates = daily["date"] if "date" in daily.columns else pd.Series(daily.index)
    start = pd.Timestamp(dates.iloc[0]).date().isoformat() if len(dates) else None
    end = pd.Timestamp(dates.iloc[-1]).date().isoformat() if len(dates) else None
    columns = [c for c in DAILY_COLUMNS if c in daily.columns]
    columns += [c for c in daily.columns if c not in columns]
    return BacktestSegmentOut(
        segment=bt.segment,
        label=SEGMENT_LABELS[bt.segment],
        in_sample=bt.segment == Segment.TRAIN,
        start=start,
        end=end,
        daily=frame_columns(daily[columns]),
        trades=list(bt.trades),
        events=[e.model_copy(update={"details": json_safe(e.details)}) for e in bt.events],
        metrics=bt.metrics,
        warnings=list(bt.warnings),
        max_holding_bars=bt.max_holding_bars,
        legs={
            "y": LegOut(symbol=bt.leg_y.symbol, asset_class=bt.leg_y.asset_class),
            "x": LegOut(symbol=bt.leg_x.symbol, asset_class=bt.leg_x.asset_class),
        },
        beta=clean_float(bt.beta),
        spread_type=bt.spread_type,
        calendar=bt.calendar,
        periods_per_year=bt.metrics.periods_per_year,
    )


def _failed_segment(segment: Segment, message: str) -> BacktestSegmentOut:
    return BacktestSegmentOut(
        segment=segment,
        label=SEGMENT_LABELS[segment],
        in_sample=segment == Segment.TRAIN,
        error=message,
    )


def analysis_out(result: RunResult) -> AnalysisOut:
    return AnalysisOut(
        synthetic=result.is_synthetic,
        data_origin=result.data_origin,
        config=result.config,
        analysis=pair_analysis_out(result.analysis, show_test_period=result.test_included),
    )


def backtest_out(result: RunResult) -> BacktestOut:
    an = result.analysis
    segments: dict[str, BacktestSegmentOut] = {}
    for segment in SEGMENT_ORDER:
        if segment in result.backtests:
            segments[segment.value] = backtest_segment_out(result.backtests[segment])
        elif segment in result.segment_errors:
            segments[segment.value] = _failed_segment(segment, result.segment_errors[segment])
    return BacktestOut(
        synthetic=result.is_synthetic,
        data_origin=result.data_origin,
        config=result.config,
        symbol_a=an.symbol_a,
        symbol_b=an.symbol_b,
        exec_symbol_a=an.exec_symbol_a,
        exec_symbol_b=an.exec_symbol_b,
        status=an.status,
        status_label=STATUS_LABELS[an.status],
        strategy_allowed=an.strategy_allowed,
        illustrative_only=an.illustrative_only,
        illustrative_banner=an.illustrative_banner,
        illustrative_reasons=list(an.illustrative_reasons),
        discard_reasons=list(an.discard_reasons),
        segments=segments,
        blocked_reason=blocked_reason(an),
        test_included=Segment.TEST in result.backtests,
        test_contaminated=result.test_contaminated,
    )
