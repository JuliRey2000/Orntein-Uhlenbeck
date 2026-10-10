"""Orquestación del análisis de un par: validar → pruebas → spread → OU → estado y causas.

Estados posibles (4.5), EXACTAMENTE estos, con la causa explicada:
  ✅ Evidencia de cointegración en la ventana evaluada
  ⚠️ Evidencia insuficiente
  ⚠️ Supuestos dudosos
  ⚠️ Resultado inestable entre ventanas
  ⛔ Datos insuficientes   (también cubre bloqueos por datos no aptos: monedas
                            distintas o desconocidas, precios ≤ 0 con spread
                            logarítmico, el mismo activo dos veces)
Precedencia del estado principal: ⛔ > supuestos dudosos > evidencia
insuficiente > inestable > ✅. Se listan TODAS las causas activas.
Nunca usar "cointegración garantizada" ni "par rentable".

CONTRATO ``analyze_pair(a, b, config, exec_a=None, exec_b=None)``:
 1. ``validate_pair`` (tipo de spread, mínimo 500 en formación). Bloqueado (o el
    mismo activo dos veces) → PairAnalysis con status INSUFFICIENT_DATA,
    causas, y el resto en None.
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
    este par?"): p ajustado > α, inestabilidad, OU no válido, vida media
    fuera de rango (>126 o <1), datos insuficientes, supuestos dudosos,
    filtro de correlación (si se activó y no se cumple), y las razones que
    bloquean la simulación (proxy requerido, precios ejecutables ≤ 0, series
    colineales).
 7. ``strategy_allowed`` = no bloqueado, OU válido y ninguna razón de
    ``STRATEGY_BLOCKING_CODES``. ``illustrative_only`` = estado distinto de ✅
    o alguna razón de descarte (el backtest se permite con el aviso «solo
    ilustrativo», que cita los motivos: ``illustrative_reasons`` e
    ``illustrative_banner``). Las razones que se conocen después del análisis
    (las añade el runner) se registran con ``add_discard_reasons``, que
    mantiene esa regla.
 8. Si se pasan exec_a/exec_b (proxy ejecutable para índices), se alinean con
    las series de análisis; si no, se ejecuta sobre las mismas series.

Precisiones de la implementación:
- ``causes`` son las razones del estado principal, ordenadas por precedencia
  (todas las activas, no solo la que decide). Si no hay ninguna, el estado es ✅
  y ``causes`` contiene una única causa con la decisión de Engle–Granger.
- La validez del OU NO es un estado (la especificación fija cinco): un OU no
  válido bloquea la estrategia y aparece en ``discard_reasons`` con
  ``status=None``, igual que la vida media fuera de rango y el filtro de
  correlación. Decisión O7: el OU solo es válido si 0 < b < 1 y Dickey–Fuller
  rechaza la raíz unitaria al nivel ``alpha_level``, con los valores críticos
  del residuo de una regresión estimada (N = 2), también en la estabilidad
  del OU por ventanas. Si no la rechaza → ``OU_NO_ESTACIONARIO``.
- Dirección colineal (Engle–Granger con estadístico −∞): no se ajusta el OU
  (``INSUFFICIENT`` con el motivo) y se descarta con ``SERIES_COLINEALES``.
  Además ``fit_ou`` recibe como escala absoluta la desviación de f(y) en
  formación: un spread que es solo ruido de redondeo nunca es un OU válido.
- Índice o futuro sin instrumento ejecutable → ``PROXY_REQUERIDO`` (se sugiere
  el proxy, nunca se sustituye); el análisis estadístico se muestra igual.
- Precios EJECUTABLES (series o proxy) ≤ 0 en cualquier tramo →
  ``PRECIO_EJECUTABLE_NO_POSITIVO`` (la simulación necesita precios positivos
  para dimensionar y valorar las piernas). Esas fechas no se excluyen.
- Las incidencias de la limpieza de cada instrumento ejecutable se añaden al
  ``ValidationReport`` con el símbolo y la etiqueta «instrumento ejecutable».
- El calendario se determina con las clases de TODOS los instrumentos usados
  (análisis + ejecutables): cripto analizado + ejecutable de bolsa → mixto
  (252 periodos, día de negociación, aviso de desfase de cierre).
- ``zscore`` es siempre el z con μ y σ_eq FIJOS de formación. El z móvil
  (``StrategyParams.zscore_mode="movil"``) lo calcula el runner a partir de
  ``spread``: es solo una comparación pedagógica.
- Si con ``frozen_split`` (o al alinear el proxy ejecutable) la formación queda
  por debajo de ``min_train_obs``, el estado es ⛔ con los números.
- Los errores de datos durante los cálculos (p. ej. una serie constante en
  formación) se convierten en una causa ⛔ ``CALCULO_NO_POSIBLE`` en lugar de
  propagarse. Una ventana de estabilidad no evaluable NO es un error: cuenta
  como ventana fallida.
- ``AnalysisConfig`` valida la coherencia de sus parámetros (ConfigError en
  español), p. ej. ``stability_window ≥ 2·(maxlag + 1) + 30``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from mrlab.backtest.split import SampleSplit, SplitConfig, compute_split, segment_slice
from mrlab.data.calendar import CLOSE_TIME_MISMATCH_WARNING, calendar_type, time_unit_label
from mrlab.data.validation import (
    AlignedPair,
    Severity,
    ValidationIssue,
    ValidationReport,
    clean_series,
    format_dates,
    format_int,
    validate_pair,
)
from mrlab.domain.errors import AnalysisBlocked, ConfigError, DataError
from mrlab.domain.models import CalendarType, PriceSeries, Segment, SpreadType
from mrlab.model.diagnostics import (
    OUStability,
    ResidualDiagnostics,
    residual_diagnostics,
    rolling_ou_stability,
)
from mrlab.model.ou import (
    MIN_OBS_OU,
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

#: Banner permanente de la decisión O5 cuando el par no superó los filtros pero el OU es
#: válido. ``PairAnalysis.illustrative_banner`` lo completa con los motivos concretos.
ILLUSTRATIVE_BANNER = "El par no superó los filtros: resultado solo ilustrativo."

SYNTHETIC_WARNING = (
    "Datos SINTÉTICOS: el par fue generado por el laboratorio con parámetros conocidos; no son "
    "precios históricos."
)

STRATEGY_BLOCKED_NOTE = (
    "Sin un OU válido no hay μ ni σ_eq, así que no se puede calcular el z-score: la estrategia "
    "queda bloqueada para este par."
)

SIMULATION_BLOCKED_NOTE = (
    "La simulación de la estrategia queda bloqueada para este par; el análisis estadístico se "
    "muestra igualmente."
)

COLLINEAR_OU_MESSAGE = (
    "Las dos series son prácticamente idénticas salvo escala (la regresión de formación las "
    "explica casi por completo y Engle–Granger no es fiable): el spread es casi nulo y solo "
    "contiene ruido numérico, así que no se ajusta el OU. Revisa si se eligió dos veces el mismo "
    "activo o dos series duplicadas."
)

#: Razones de descarte que bloquean la simulación (además de las que hacen ⛔ el estado).
STRATEGY_BLOCKING_CODES = frozenset(
    {
        "OU_NO_ESTACIONARIO",
        "OU_B_NO_POSITIVO",
        "OU_DATOS_INSUFICIENTES",
        "SERIES_COLINEALES",
        "PROXY_REQUERIDO",
        "PROXY_NO_EJECUTABLE",
        "PRECIO_EJECUTABLE_NO_POSITIVO",
    }
)

#: Observaciones exigidas por encima del mínimo técnico de Engle–Granger en cada ventana.
STABILITY_WINDOW_MARGIN = 30

SEGMENT_NAMES = {
    Segment.TRAIN: "formación",
    Segment.VALIDATION: "validación",
    Segment.TEST: "prueba",
}


def min_stability_window(maxlag: int) -> int:
    """Ventana mínima de estabilidad: 2·(maxlag + 1) (mínimo de ``coint``) + 30 observaciones."""
    return 2 * (maxlag + 1) + STABILITY_WINDOW_MARGIN


def min_train_observations(maxlag: int) -> int:
    """Formación mínima con la que las pruebas pueden calcularse (ADF con constante sobre las
    diferencias y el OU)."""
    return max(MIN_OBS_OU, 2 * (maxlag + 2) + 1)


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

    @model_validator(mode="after")
    def _coherent(self) -> AnalysisConfig:
        """Parámetros coherentes entre sí; si no, ConfigError con la explicación en español."""
        problems: list[str] = []
        if self.fdr_method not in FDR_METHOD_LABELS:
            problems.append(
                f"el método de corrección debe ser «bh» (Benjamini–Hochberg) o «by» "
                f"(Benjamini–Yekutieli); se recibió «{self.fdr_method}»"
            )
        if self.maxlag < 0:
            problems.append(f"los rezagos máximos deben ser 0 o más (se recibió {self.maxlag})")
        else:
            min_window = min_stability_window(self.maxlag)
            if self.stability_window < min_window:
                problems.append(
                    f"la ventana de estabilidad debe tener al menos {min_window} observaciones "
                    f"con {self.maxlag} rezagos máximos: 2·(rezagos + 1) = "
                    f"{2 * (self.maxlag + 1)}, que es lo mínimo que exige Engle–Granger, más "
                    f"{STABILITY_WINDOW_MARGIN} para que la prueba tenga sentido (se recibió "
                    f"{self.stability_window})"
                )
            min_train = min_train_observations(self.maxlag)
            if self.min_train_obs < min_train:
                problems.append(
                    f"el mínimo de observaciones en formación debe ser al menos {min_train} con "
                    f"{self.maxlag} rezagos máximos (se recibió {self.min_train_obs})"
                )
        if self.stability_step < 1:
            problems.append(
                f"el paso entre ventanas debe ser al menos 1 (se recibió {self.stability_step})"
            )
        if not 0.0 <= self.max_fail_frac <= 1.0:
            problems.append(
                "la fracción máxima de ventanas fallidas debe estar entre 0 y 1 (se recibió "
                f"{self.max_fail_frac})"
            )
        if not (
            math.isfinite(self.half_life_min)
            and math.isfinite(self.half_life_max)
            and 0.0 <= self.half_life_min < self.half_life_max
        ):
            problems.append(
                "la vida media mínima debe ser 0 o más y menor que la máxima (se recibió "
                f"mínima {self.half_life_min} y máxima {self.half_life_max})"
            )
        if not -1.0 <= self.correlation_min <= 1.0:
            problems.append(
                "la correlación mínima del filtro debe estar entre −1 y 1 (se recibió "
                f"{self.correlation_min})"
            )
        if not self.forecast_horizons or any(h < 1 for h in self.forecast_horizons):
            problems.append(
                "los horizontes de pronóstico deben ser enteros mayores o iguales a 1 (se "
                f"recibió {list(self.forecast_horizons)})"
            )
        if not 0.0 < self.forecast_level < 1.0:
            problems.append(
                "el nivel de los intervalos de pronóstico debe estar entre 0 y 1, por ejemplo "
                f"0,95 (se recibió {self.forecast_level})"
            )
        if problems:
            raise ConfigError(
                "Configuración del análisis no válida: " + "; ".join(problems) + ".",
                code="CONFIGURACION_ANALISIS_INVALIDA",
                details={"problemas": problems},
            )
        return self


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
    calendar: CalendarType | None = None  # calendario efectivo (análisis + ejecutables)
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
    #: Por qué el resultado es solo ilustrativo (estado no ✅ y razones de descarte).
    illustrative_reasons: list[str] = field(default_factory=list)

    @property
    def dependent_symbol(self) -> str | None:
        return self.hedge.dependent if self.hedge else None

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def blocked(self) -> bool:
        return self.status == PairStatus.INSUFFICIENT_DATA

    @property
    def illustrative_banner(self) -> str | None:
        """Texto del banner «solo ilustrativo» con sus motivos (None si no corresponde).

        Empieza siempre por ``ILLUSTRATIVE_BANNER``; el runner lo usa en cada backtest.
        """
        if not self.illustrative_only:
            return None
        if not self.illustrative_reasons:
            return ILLUSTRATIVE_BANNER
        label = "Motivo" if len(self.illustrative_reasons) == 1 else "Motivos"
        return f"{ILLUSTRATIVE_BANNER} {label}: " + " ".join(self.illustrative_reasons)


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


def _currency(code: str | None) -> str | None:
    return code.strip() if code and code.strip() else None


# --------------------------------------------------------------------------- estado


def resolve_status(causes: list[Cause]) -> PairStatus:
    """Estado principal según la precedencia de la decisión E6 (sin causas → ✅)."""
    active = {c.status for c in causes if c.status is not None}
    return next((s for s in STATUS_PRECEDENCE if s in active), PairStatus.EVIDENCE)


def _by_precedence(causes: list[Cause]) -> list[Cause]:
    order = {s: i for i, s in enumerate(STATUS_PRECEDENCE)}
    return sorted(causes, key=lambda c: order.get(c.status, len(order)))


# --------------------------------------------------------------------------- piezas


EFFECTIVE_MIXED_CALENDAR_WARNING = (
    "Calendario efectivo mixto (cripto + bolsa): se combinan series de cripto con instrumentos "
    "ejecutables de bolsa (o al revés). El análisis y la simulación usan solo las fechas en que "
    "cotizan todos los instrumentos, el OU se mide en días de negociación y la anualización usa "
    "252 periodos."
)


@dataclass(frozen=True)
class _Execution:
    """Instrumentos ejecutables alineados con el análisis (ver ``_executable_prices``)."""

    aligned: AlignedPair
    prices: pd.DataFrame  # columnas "a", "b"
    symbols: tuple[str | None, str | None]  # proxy de cada pierna (None = la propia serie)
    issues: list[ValidationIssue]  # incidencias de la limpieza de los proxies
    calendar: CalendarType  # calendario de TODOS los instrumentos usados


def _same_asset_cause(a: PriceSeries, b: PriceSeries) -> Cause | None:
    """⛔ si las dos piernas son el mismo activo (mismo proveedor y símbolo)."""
    if a.meta.provider != b.meta.provider:
        return None
    if a.symbol.strip().upper() != b.symbol.strip().upper():
        return None
    return Cause(
        code="MISMO_ACTIVO",
        status=PairStatus.INSUFFICIENT_DATA,
        message=(
            f"Se eligió dos veces el mismo activo («{a.symbol}» del proveedor "
            f"«{a.meta.provider}»). Un par necesita dos series distintas: con la misma serie en "
            "las dos piernas el spread es cero salvo ruido numérico y ninguna prueba tiene "
            "sentido. Elige otro activo para una de las piernas."
        ),
    )


def _non_executable_kind(series: PriceSeries) -> str:
    kinds = {"indice": "un índice", "futuro": "un futuro continuo"}
    return kinds.get(series.meta.asset.asset_class.value, "un instrumento no ejecutable")


def _proxy_required_cause(series: PriceSeries) -> Cause:
    asset = series.meta.asset
    suggestion = (
        f" El proxy sugerido es «{asset.suggested_proxy}»: indícalo de forma explícita como "
        "instrumento ejecutable (nunca se sustituye en silencio)."
        if asset.suggested_proxy
        else " Elige un instrumento ejecutable que lo replique (por ejemplo, un ETF); nunca se "
        "sustituye en silencio."
    )
    return Cause(
        code="PROXY_REQUERIDO",
        status=None,
        message=(
            f"«{asset.symbol}» es {_non_executable_kind(series)}: no se puede comprar ni vender "
            f"directamente, así que la simulación necesita un instrumento ejecutable.{suggestion} "
            f"{SIMULATION_BLOCKED_NOTE}"
        ),
    )


def _exec_issue(issue: ValidationIssue, exec_symbol: str, analysis_symbol: str) -> ValidationIssue:
    """Incidencia de la limpieza de un proxy, rotulada como «instrumento ejecutable»."""
    body = issue.message.removeprefix(f"{exec_symbol}: ")
    if issue.code == "PRECIO_NO_POSITIVO":
        body = (
            f"{len(issue.dates)} precio(s) menores o iguales a cero "
            f"({format_dates(issue.dates)}). La simulación necesita precios positivos para "
            "dimensionar y valorar las piernas, así que la estrategia queda bloqueada."
        )
    return issue.model_copy(
        update={
            "symbol": exec_symbol,
            "message": (
                f"Instrumento ejecutable «{exec_symbol}» (se opera en lugar de "
                f"«{analysis_symbol}»): {body}"
            ),
        }
    )


def _executable_prices(
    aligned: AlignedPair,
    series: tuple[PriceSeries, PriceSeries],
    execs: tuple[PriceSeries | None, PriceSeries | None],
    warnings: list[str],
    causes: list[Cause],
    discards: list[Cause],
) -> _Execution:
    """Precios ejecutables (columnas "a", "b") en las mismas fechas que el análisis.

    Sin proxy, cada pierna se ejecuta sobre su propia serie (si es un índice o futuro, la
    simulación queda bloqueada con ``PROXY_REQUERIDO``). Con proxy, solo se usan las fechas
    en que también cotiza el proxy (nunca se rellena); sus precios ≤ 0 se conservan y
    bloquean la simulación más adelante.
    """
    prices = aligned.prices
    columns: dict[str, pd.Series] = {"a": prices["a"], "b": prices["b"]}
    index = prices.index
    exec_symbols: list[str | None] = []
    issues: list[ValidationIssue] = []
    classes = [s.meta.asset.asset_class for s in series]
    for leg, analysis_series, exec_series in zip(("a", "b"), series, execs, strict=True):
        asset = analysis_series.meta.asset
        if exec_series is None:
            exec_symbols.append(None)
            if not asset.executable:
                discards.append(_proxy_required_cause(analysis_series))
            continue
        exec_asset = exec_series.meta.asset
        exec_symbols.append(exec_series.symbol)
        classes.append(exec_asset.asset_class)
        clean, report = clean_series(exec_series)
        issues.extend(_exec_issue(i, exec_series.symbol, asset.symbol) for i in report.issues)
        if not exec_asset.executable:
            discards.append(
                Cause(
                    code="PROXY_NO_EJECUTABLE",
                    status=None,
                    message=(
                        f"El instrumento elegido para operar «{asset.symbol}», "
                        f"«{exec_series.symbol}», es {_non_executable_kind(exec_series)}: tampoco "
                        "se puede comprar ni vender directamente. Elige un instrumento ejecutable "
                        f"(por ejemplo, un ETF). {SIMULATION_BLOCKED_NOTE}"
                    ),
                )
            )
        cur, cur_exec = _currency(asset.currency), _currency(exec_asset.currency)
        if cur_exec is None:
            causes.append(
                Cause(
                    code="MONEDA_DESCONOCIDA",
                    status=PairStatus.INSUFFICIENT_DATA,
                    message=(
                        f"La moneda del instrumento ejecutable «{exec_series.symbol}» es "
                        f"desconocida («{asset.symbol}» cotiza en {cur}). Sin la moneda no se "
                        "puede comprobar que ambos coticen en la misma, así que el análisis "
                        "queda bloqueado. Si cargaste un CSV, declara su moneda; si no, prueba "
                        "con otro proveedor."
                    ),
                )
            )
        elif cur_exec != cur:
            causes.append(
                Cause(
                    code="MONEDA_PROXY_DISTINTA",
                    status=PairStatus.INSUFFICIENT_DATA,
                    message=(
                        f"«{asset.symbol}» cotiza en {cur} y su instrumento ejecutable "
                        f"«{exec_series.symbol}» en {cur_exec}: deben estar en la misma moneda."
                    ),
                )
            )
        warnings.append(
            f"Se analiza «{asset.symbol}» y se opera «{exec_series.symbol}»: la señal y el PnL "
            "provienen de instrumentos distintos (riesgo de base; por ejemplo, un índice de precios "
            "no incluye dividendos y un ETF ajustado sí)."
        )
        close = clean.data["close"].astype(float)
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
    calendar = calendar_type(classes)
    if dropped or calendar != aligned.calendar:
        aligned = dc_replace(aligned, prices=prices.loc[index], calendar=calendar)
    return _Execution(
        aligned=aligned,
        prices=exec_prices,
        symbols=(exec_symbols[0], exec_symbols[1]),
        issues=issues,
        calendar=calendar,
    )


def _report_with_execution(
    report: ValidationReport, execution: _Execution, warnings: list[str]
) -> ValidationReport:
    """Informe de validación con las incidencias de los proxies y el calendario efectivo."""
    added = list(execution.issues)
    if execution.calendar != report.calendar and execution.calendar == CalendarType.MIXED:
        added.append(
            ValidationIssue(
                code="DESFASE_CIERRE_CRIPTO",
                severity=Severity.WARNING,
                message=CLOSE_TIME_MISMATCH_WARNING,
            )
        )
        warnings.append(EFFECTIVE_MIXED_CALENDAR_WARNING)
    if not added and execution.calendar == report.calendar:
        return report
    n_exec = sum(i.severity != Severity.INFO for i in execution.issues)
    status = report.status_message
    if n_exec:
        status += (
            f" Además, los instrumentos ejecutables tienen {n_exec} incidencia(s) que conviene "
            "revisar."
        )
    return report.model_copy(
        update={
            "calendar": execution.calendar,
            "issues": [*report.issues, *added],
            "status_message": status,
        }
    )


def _non_positive_exec_discards(
    exec_prices: pd.DataFrame, split: SampleSplit, symbols: dict[str, str]
) -> list[Cause]:
    """Precios ejecutables ≤ 0 en cualquier tramo: la simulación no puede dimensionar ni
    valorar las piernas (cantidad = nocional / precio), así que se bloquea."""
    out: list[Cause] = []
    for leg in ("a", "b"):
        column = exec_prices[leg]
        bad = column[~(column > 0)]
        if bad.empty:
            continue
        segments = [SEGMENT_NAMES[s] for s in Segment if not segment_slice(bad, split, s).empty]
        if len(segments) == 1:
            where = f" en el tramo de {segments[0]}"
        elif segments:
            where = f" en los tramos de {', '.join(segments[:-1])} y {segments[-1]}"
        else:
            where = ""
        first = bad.index[0].date().isoformat()
        out.append(
            Cause(
                code="PRECIO_EJECUTABLE_NO_POSITIVO",
                status=None,
                message=(
                    f"El instrumento ejecutable «{symbols[leg]}» tiene {format_int(len(bad))} "
                    f"precio(s) menores o iguales a cero{where} (el primero el {first}). La "
                    "simulación necesita precios positivos para dimensionar las piernas "
                    "(cantidad = nocional / precio) y valorar la posición. "
                    f"{SIMULATION_BLOCKED_NOTE}"
                ),
            )
        )
    return out


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
        # Incluye b indistinguible de 1 (decisión O7): el mensaje del ajuste lo explica.
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
    assert ou.half_life is not None
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

    # 1. Bloqueos: el mismo activo dos veces y los de la validación (moneda, precios ≤ 0
    #    con log, < 500 en formación).
    blocking: list[Cause] = []
    same_asset = _same_asset_cause(a, b)
    if same_asset is not None:
        blocking.append(same_asset)
    if report.blocked or aligned is None:
        blocking += [
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
    if blocking or aligned is None:
        return _finish(out, blocking, [], warnings)

    # 2. Precios ejecutables, calendario efectivo y división de la muestra.
    causes: list[Cause] = []
    discards: list[Cause] = []
    execution = _executable_prices(aligned, (a, b), (exec_a, exec_b), warnings, causes, discards)
    aligned = execution.aligned
    out.validation = _report_with_execution(report, execution, warnings)
    out.calendar = execution.calendar
    out.aligned, out.exec_prices = aligned, execution.prices
    out.exec_symbol_a, out.exec_symbol_b = execution.symbols
    prices = aligned.prices
    if len(prices) == 0:
        causes.append(
            Cause(
                code="DATOS_INSUFICIENTES",
                status=PairStatus.INSUFFICIENT_DATA,
                message="No quedan fechas comunes entre las series de análisis y las ejecutables.",
            )
        )
        return _finish(out, causes, discards, warnings)
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
        return _finish(out, causes, discards, warnings)
    exec_names = {
        "a": execution.symbols[0] or sym_a,
        "b": execution.symbols[1] or sym_b,
    }
    discards.extend(_non_positive_exec_discards(execution.prices, split, exec_names))

    # 3–5. Pruebas, spread y OU (solo formación para estimar).
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
    collinear = not math.isfinite(chosen.statistic)
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

    # OU con formación (sección 6). El spread es el RESIDUO de la regresión de cointegración
    # estimada con estos datos: valores críticos N = 2 (decisión O7).
    assert out.calendar is not None
    time_unit = time_unit_label(out.calendar)
    if collinear:
        out.ou = OUFit(
            status=OUStatus.INSUFFICIENT,
            valid=False,
            nobs=max(len(spread_train) - 1, 0),
            time_unit=time_unit,
            df_alpha=cfg.alpha_level,
            df_estimated_residual=True,
            message=COLLINEAR_OU_MESSAGE,
        )
        discards.append(
            Cause(
                code="SERIES_COLINEALES",
                status=None,
                message=f"{COLLINEAR_OU_MESSAGE} {STRATEGY_BLOCKED_NOTE}",
            )
        )
    else:
        scale = float(np.std(f_train[dep].to_numpy(dtype=float)))
        scale_arg = scale if math.isfinite(scale) and scale > 0 else None
        ou = fit_ou(
            spread_train,
            time_unit=time_unit,
            half_life_max=cfg.half_life_max,
            half_life_min=cfg.half_life_min,
            unit_root_alpha=cfg.alpha_level,
            estimated_residual=True,
            scale=scale_arg,
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
                spread_train,
                cfg.stability_window,
                cfg.stability_step,
                time_unit=time_unit,
                unit_root_alpha=cfg.alpha_level,
                estimated_residual=True,
                scale=scale_arg,
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
    blocks_simulation = any(d.code in STRATEGY_BLOCKING_CODES for d in discards)
    out.strategy_allowed = (
        out.status != PairStatus.INSUFFICIENT_DATA
        and out.ou is not None
        and out.ou.valid
        and not blocks_simulation
    )
    # Solo ilustrativo si el par no es ✅ o si el panel «¿Por qué se descartó?» tiene alguna
    # razón (p. ej. vida media fuera de [1, 126]): el banner cita los motivos.
    reasons: list[str] = []
    if out.status != PairStatus.EVIDENCE:
        reasons.append(f"Estado del par: {STATUS_LABELS[out.status]}.")
    reasons.extend(d.message for d in discards)
    out.illustrative_reasons = _dedupe(reasons)
    out.illustrative_only = bool(out.illustrative_reasons)
    banner = out.illustrative_banner
    if out.strategy_allowed and banner is not None:
        warnings.append(banner)
    out.warnings = _dedupe(warnings)
    return out


def add_discard_reasons(analysis: PairAnalysis, causes: Iterable[Cause]) -> None:
    """Añade razones de descarte calculadas DESPUÉS de ``analyze_pair`` (p. ej. por el runner:
    costes que superan el PnL bruto en validación, un segmento que no se pudo simular).

    Mantiene la decisión O5: toda razón del panel «¿Por qué se descartó?» hace el resultado
    solo ilustrativo, así que se actualizan ``illustrative_reasons``, ``illustrative_only`` y
    el banner de ``warnings``. No cambia el estado ni ``strategy_allowed``.
    """
    new = list(causes)
    if not new:
        return
    old_banner = analysis.illustrative_banner
    analysis.discard_reasons.extend(new)
    analysis.illustrative_reasons = _dedupe(
        [*analysis.illustrative_reasons, *(c.message for c in new)]
    )
    analysis.illustrative_only = True
    warnings = [w for w in analysis.warnings if w != old_banner]
    banner = analysis.illustrative_banner
    if analysis.strategy_allowed and banner is not None:
        warnings.append(banner)
    analysis.warnings = _dedupe(warnings)
