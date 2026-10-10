"""Ejecución completa desde un RunConfig: datos (caché) → análisis → backtests.

CONTRATO:
- ``run_pair(config, data, include_test=False)``:
  carga las series con ``data.load_snapshot`` si el RunConfig trae snapshot_id
  (reproducción exacta, sin red) o con ``data.load`` si no; congela el split
  (si config.frozen_split es None se calcula y se guarda en el RunConfig
  devuelto); analiza; si ``strategy_allowed`` ejecuta el backtest de
  FORMACIÓN (marcado como dentro de muestra, solo ilustrativo) y VALIDACIÓN; la
  PRUEBA solo si ``include_test`` (se ejecuta una vez con la configuración
  congelada). Rellena ``config.series`` con los SeriesMeta usados y
  ``config.code_version``. Añade a ``discard_reasons`` "costes superan el PnL
  bruto" si ocurre en validación.
- El z de cada segmento sale de la serie z completa (parámetros de formación):
  no requiere calentamiento y no usa información futura.
- Anualización: ``mrlab.data.calendar.periods_per_year`` del calendario del par.

Precisiones de la implementación:
- El RunConfig devuelto es autosuficiente para reproducir la ejecución: cada
  ``SeriesRef`` queda con el ``snapshot_id`` de la instantánea usada (las
  fechas pedidas se conservan), ``frozen_split`` con las fechas de corte,
  ``data_origin`` según los datos (SINTETICO si alguna serie lo es, aunque el
  RunConfig dijera otra cosa), ``code_version`` y ``config_hash``. Guardarlo,
  cargarlo y volver a ejecutarlo da exactamente el mismo resultado sin usar el
  proveedor.
- Sin ``snapshot_id``, la referencia debe traer ``start`` y ``end``.
- Un snapshot cuyo proveedor o símbolo no coincide con la referencia →
  ConfigError ``RUN_CONFIG_INCONSISTENTE``.
- Los segmentos sin observaciones (p. ej. validación = 0 %) no se simulan.
- Con ``StrategyParams.zscore_mode="movil"`` el backtest usa el z móvil
  (``rolling_zscore`` sobre el spread completo, solo hacia atrás) y se avisa de
  que es solo una comparación pedagógica.
- Si el par no es ✅, cada backtest lleva el banner «solo ilustrativo» (O5).
- ``RunResult.test_contaminated`` lo fija la capa de estudios (decisión S10);
  la exportación lo informa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from mrlab.analysis.pair_pipeline import (
    ILLUSTRATIVE_BANNER,
    Cause,
    PairAnalysis,
    analyze_pair,
)
from mrlab.backtest.engine import BacktestInput, BacktestResult, LegSpec, run_backtest
from mrlab.backtest.split import segment_slice
from mrlab.data.service import DataService
from mrlab.domain.errors import ConfigError
from mrlab.domain.models import DataOrigin, PriceSeries, Segment
from mrlab.model.ou import rolling_zscore
from mrlab.reporting.run_config import PairSpec, RunConfig, SeriesRef, config_hash, git_version
from mrlab.stats.cointegration import Direction

ROLLING_Z_WARNING = (
    "Este backtest usa el z MÓVIL (media y desviación de una ventana hacia atrás): es solo una "
    "comparación pedagógica. El z de referencia usa μ y σ_eq fijos de formación."
)

#: Orden de las series en ``RunConfig.series`` y en ``RunResult.series``.
SERIES_ROLES = ("a", "b", "exec_a", "exec_b")


@dataclass
class RunResult:
    config: RunConfig
    analysis: PairAnalysis
    backtests: dict[Segment, BacktestResult] = field(default_factory=dict)
    is_synthetic: bool = False
    series: dict[str, PriceSeries] = field(default_factory=dict)  # por rol: a, b, exec_a, exec_b
    test_included: bool = False
    test_contaminated: bool = False

    @property
    def data_origin(self) -> DataOrigin:
        return DataOrigin.SYNTHETIC if self.is_synthetic else DataOrigin.HISTORICAL


def _parse_date(value: str | None, where: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(
            f"Fecha inválida en {where}: «{value}». Usa el formato AAAA-MM-DD (p. ej. 2020-01-31).",
            code="FECHA_INVALIDA",
            details={"campo": where, "valor": value},
        ) from exc


def _load(ref: SeriesRef, data: DataService, role: str) -> PriceSeries:
    start = _parse_date(ref.start, f"{role}.start")
    end = _parse_date(ref.end, f"{role}.end")
    if ref.snapshot_id:
        series = data.load_snapshot(ref.snapshot_id, start, end)
        if (
            series.meta.provider != ref.provider
            or series.symbol.strip().upper() != ref.symbol.strip().upper()
        ):
            raise ConfigError(
                f"La instantánea «{ref.snapshot_id}» contiene «{series.symbol}» de "
                f"«{series.meta.provider}», pero la configuración pide «{ref.symbol}» de "
                f"«{ref.provider}». Revisa el run_config.json.",
                code="RUN_CONFIG_INCONSISTENTE",
                details={"rol": role, "snapshot_id": ref.snapshot_id},
            )
        return series
    if start is None or end is None:
        raise ConfigError(
            f"Para cargar «{ref.symbol}» del proveedor «{ref.provider}» hay que indicar la fecha "
            "de inicio y la de fin (start y end, en formato AAAA-MM-DD).",
            code="RANGO_FECHAS_FALTANTE",
            details={"rol": role},
        )
    return data.load(ref.provider, ref.symbol, start, end)


def _pinned(ref: SeriesRef, series: PriceSeries) -> SeriesRef:
    """La misma referencia, fijada a la instantánea realmente usada."""
    return ref.model_copy(update={"snapshot_id": series.meta.snapshot_id})


def _segment_input(
    analysis: PairAnalysis,
    z: pd.Series,
    segment: Segment,
    legs: tuple[PriceSeries, PriceSeries],
    origin: DataOrigin,
) -> BacktestInput | None:
    assert analysis.split is not None and analysis.exec_prices is not None
    assert analysis.hedge is not None and analysis.ou is not None and analysis.calendar
    prices = segment_slice(analysis.exec_prices, analysis.split, segment)
    if prices.empty:
        return None
    y_leg, x_leg = ("a", "b") if analysis.chosen_direction == Direction.A_ON_B else ("b", "a")
    leg_series = dict(zip(("a", "b"), legs, strict=True))
    frame = pd.DataFrame({"y": prices[y_leg], "x": prices[x_leg]}, index=prices.index)

    def spec(leg: str) -> LegSpec:
        s = leg_series[leg]
        return LegSpec(symbol=s.symbol, asset_class=s.meta.asset.asset_class)

    return BacktestInput(
        prices=frame,
        z=segment_slice(z, analysis.split, segment).reindex(prices.index),
        beta=analysis.hedge.beta,
        spread_type=analysis.hedge.spread_type,
        half_life=analysis.ou.half_life,
        leg_y=spec(y_leg),
        leg_x=spec(x_leg),
        calendar=analysis.calendar,
        segment=segment,
        origin=origin,
    )


def _costs_cause(bt: BacktestResult) -> Cause | None:
    m = bt.metrics
    if not (m.gross_pnl > 0 and m.total_costs > m.gross_pnl):
        return None

    def money(x: float) -> str:
        return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

    return Cause(
        code="COSTES_SUPERAN_BRUTO",
        status=None,
        message=(
            f"En validación, los costes ({money(m.total_costs)}) superan el PnL bruto "
            f"({money(m.gross_pnl)}): con estos costes la estrategia no conserva nada de lo que "
            "gana por el movimiento de precios."
        ),
    )


def run_pair(config: RunConfig, data: DataService, include_test: bool = False) -> RunResult:
    refs = {
        "a": config.pair.a,
        "b": config.pair.b,
        "exec_a": config.pair.exec_a,
        "exec_b": config.pair.exec_b,
    }
    loaded = {role: _load(ref, data, role) for role, ref in refs.items() if ref is not None}
    is_synthetic = any(s.is_synthetic for s in loaded.values())
    origin = DataOrigin.SYNTHETIC if is_synthetic else DataOrigin.HISTORICAL

    analysis = analyze_pair(
        loaded["a"],
        loaded["b"],
        config.analysis,
        exec_a=loaded.get("exec_a"),
        exec_b=loaded.get("exec_b"),
        frozen_split=config.frozen_split,
    )

    pair = PairSpec(
        **{
            role: _pinned(ref, loaded[role]) if ref is not None else None
            for role, ref in refs.items()
        }
    )
    final = config.model_copy(
        update={
            "pair": pair,
            "frozen_split": analysis.split or config.frozen_split,
            "data_origin": origin,
            "series": [loaded[r].meta for r in SERIES_ROLES if r in loaded],
            "code_version": git_version(),
            "config_hash": None,
        }
    )
    final = final.model_copy(update={"config_hash": config_hash(final)})

    result = RunResult(
        config=final,
        analysis=analysis,
        is_synthetic=is_synthetic,
        series=loaded,
        test_included=include_test,
    )
    if not analysis.strategy_allowed:
        return result

    assert analysis.spread is not None and analysis.zscore is not None
    strategy = final.strategy
    rolling = strategy.zscore_mode == "movil"
    z = rolling_zscore(analysis.spread, strategy.rolling_window) if rolling else analysis.zscore
    legs = (loaded.get("exec_a", loaded["a"]), loaded.get("exec_b", loaded["b"]))
    segments = [Segment.TRAIN, Segment.VALIDATION] + ([Segment.TEST] if include_test else [])
    for segment in segments:
        inp = _segment_input(analysis, z, segment, legs, origin)
        if inp is None:
            continue
        bt = run_backtest(inp, strategy, final.costs)
        extra = ([ILLUSTRATIVE_BANNER] if analysis.illustrative_only else []) + (
            [ROLLING_Z_WARNING] if rolling else []
        )
        bt.warnings = [*extra, *bt.warnings]
        result.backtests[segment] = bt

    validation = result.backtests.get(Segment.VALIDATION)
    cause = _costs_cause(validation) if validation is not None else None
    if cause is not None:
        analysis.discard_reasons.append(cause)
    return result
