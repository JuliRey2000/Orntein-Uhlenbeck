"""Ejecución completa desde un RunConfig: datos (caché) → análisis → backtests.

CONTRATO:
- ``run_pair(config, data, include_test=None)``:
  carga las series con ``data.load_snapshot`` si el RunConfig trae snapshot_id
  (reproducción exacta, sin red) o con ``data.load`` si no; congela el split
  (si config.frozen_split es None se calcula y se guarda en el RunConfig
  devuelto); analiza; si ``strategy_allowed`` ejecuta el backtest de
  FORMACIÓN (marcado como dentro de muestra, solo ilustrativo) y VALIDACIÓN; la
  PRUEBA solo si se pide (ver abajo; se ejecuta con la configuración
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
  RunConfig dijera otra cosa), ``include_test``, ``seeds``, ``test_history``,
  ``code_version`` y ``config_hash``. Guardarlo, cargarlo y volver a ejecutarlo
  da exactamente el mismo resultado (también con la prueba) sin usar el
  proveedor.
- Prueba: se ejecuta si ``config.include_test`` o si el argumento
  ``include_test`` es True (el argumento solo puede ACTIVARLA: con None o False
  manda la configuración, de modo que reproducir un run_config.json que
  incluía la prueba la vuelve a incluir). El valor efectivo queda en el
  RunConfig devuelto y forma parte del hash.
- Prueba contaminada (decisión S10, versión de la fase 1): cuando la prueba se
  simula, h = config_hash; si ``test_history`` contiene algún hash distinto de
  h, ``RunResult.test_contaminated = True`` (la prueba ya se vio con otra
  configuración; la marca es permanente mientras el historial la conserve);
  si no, False. h se añade al historial si no estaba. Si la prueba no se
  simula, ``test_contaminated = None`` («no aplica») y el historial se
  conserva sin cambios. El historial viaja en run_config.json; el registro
  persistente son los estudios de la API (``mrlab.studies``).
- ``seeds``: las del RunConfig más las de ``SeriesMeta.seed`` de cada serie
  generada (clave = símbolo).
- Sin ``snapshot_id``, la referencia debe traer ``start`` y ``end``.
- Un snapshot cuyo proveedor o símbolo no coincide con la referencia →
  ConfigError ``RUN_CONFIG_INCONSISTENTE``.
- Los segmentos sin observaciones (p. ej. validación = 0 %) no se simulan.
- Si el motor rechaza un segmento con un ``LabError`` (p. ej. precios
  ejecutables ≤ 0), ese segmento no se simula, la causa se añade a
  ``discard_reasons`` (``SIMULACION_NO_POSIBLE``) y a
  ``RunResult.segment_errors``, y el resto del análisis se conserva. Otros
  errores (defectos de programación) se propagan.
- Con ``StrategyParams.zscore_mode="movil"`` el backtest usa el z móvil
  (``rolling_zscore`` sobre el spread completo, solo hacia atrás) y se avisa de
  que es solo una comparación pedagógica.
- Si el resultado es solo ilustrativo (``analysis.illustrative_only``: par no
  ✅ o con razones de descarte), cada backtest lleva primero el banner con sus
  motivos (``analysis.illustrative_banner``; decisión O5). Las razones que el
  runner añade al simular (``SIMULACION_NO_POSIBLE``, ``COSTES_SUPERAN_BRUTO``)
  se registran con ``add_discard_reasons``: el resultado pasa a ser solo
  ilustrativo y el banner de cada backtest las cita.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from mrlab.analysis.pair_pipeline import (
    Cause,
    PairAnalysis,
    add_discard_reasons,
    analyze_pair,
)
from mrlab.backtest.engine import BacktestInput, BacktestResult, LegSpec, run_backtest
from mrlab.backtest.split import segment_slice
from mrlab.data.service import DataService
from mrlab.domain.errors import ConfigError, LabError
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

_SEGMENT_NAMES = {
    Segment.TRAIN: "formación",
    Segment.VALIDATION: "validación",
    Segment.TEST: "prueba",
}


@dataclass
class RunResult:
    config: RunConfig
    analysis: PairAnalysis
    backtests: dict[Segment, BacktestResult] = field(default_factory=dict)
    is_synthetic: bool = False
    series: dict[str, PriceSeries] = field(default_factory=dict)  # por rol: a, b, exec_a, exec_b
    test_included: bool = False  # ¿se pidió la prueba? (RunConfig.include_test efectivo)
    #: True/False si la prueba se simuló (según ``RunConfig.test_history``); None si no.
    test_contaminated: bool | None = None
    #: Segmentos que no se pudieron simular → mensaje del error (en español).
    segment_errors: dict[Segment, str] = field(default_factory=dict)
    #: Origen y motivo de ``test_contaminated`` cuando lo decide un registro externo al
    #: ``run_config.json`` (el registro persistente de los estudios, ``mrlab.studies``).
    #: None = la marca sale solo del historial del run_config.json (decisión S10a).
    test_contamination_note: str | None = None

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


def load_ref(ref: SeriesRef, data: DataService, role: str = "serie") -> PriceSeries:
    """Carga una referencia como lo hace ``run_pair``: la instantánea fijada si trae
    ``snapshot_id`` (sin red); si no, el proveedor (con caché) en [start, end]."""
    return _load(ref, data, role)


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


def _segment_failure(segment: Segment, exc: LabError) -> Cause:
    return Cause(
        code="SIMULACION_NO_POSIBLE",
        status=None,
        message=(
            f"No se pudo simular el segmento de {_SEGMENT_NAMES[segment]}: {exc.message} Ese "
            "segmento queda sin backtest; el resto del análisis se conserva."
        ),
    )


def _seeds(config: RunConfig, loaded: dict[str, PriceSeries]) -> dict[str, int]:
    """Semillas del RunConfig más las de las series generadas (clave = símbolo)."""
    seeds = dict(config.seeds)
    for role in SERIES_ROLES:
        series = loaded.get(role)
        seed = getattr(series.meta, "seed", None) if series is not None else None
        if seed is not None:
            seeds[series.symbol] = int(seed)
    return dict(sorted(seeds.items()))


def _record_test_run(result: RunResult) -> None:
    """Registra en el historial la ejecución de la prueba y decide si está contaminada."""
    cfg = result.config
    current = cfg.config_hash
    assert current is not None
    history = list(cfg.test_history)
    result.test_contaminated = any(h != current for h in history)
    if current not in history:
        history.append(current)
    # test_history no forma parte del hash: config_hash no cambia.
    result.config = cfg.model_copy(update={"test_history": history})


def run_pair(config: RunConfig, data: DataService, include_test: bool | None = None) -> RunResult:
    refs = {
        "a": config.pair.a,
        "b": config.pair.b,
        "exec_a": config.pair.exec_a,
        "exec_b": config.pair.exec_b,
    }
    loaded = {role: _load(ref, data, role) for role, ref in refs.items() if ref is not None}
    is_synthetic = any(s.is_synthetic for s in loaded.values())
    origin = DataOrigin.SYNTHETIC if is_synthetic else DataOrigin.HISTORICAL
    run_test = bool(config.include_test or include_test)

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
            "seeds": _seeds(config, loaded),
            "include_test": run_test,
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
        test_included=run_test,
    )
    if not analysis.strategy_allowed:
        return result

    assert analysis.spread is not None and analysis.zscore is not None
    strategy = final.strategy
    rolling = strategy.zscore_mode == "movil"
    z = rolling_zscore(analysis.spread, strategy.rolling_window) if rolling else analysis.zscore
    legs = (loaded.get("exec_a", loaded["a"]), loaded.get("exec_b", loaded["b"]))
    banner = analysis.illustrative_banner
    extra = ([banner] if banner else []) + ([ROLLING_Z_WARNING] if rolling else [])
    late: list[Cause] = []  # razones de descarte que solo se conocen al simular
    segments = [Segment.TRAIN, Segment.VALIDATION] + ([Segment.TEST] if run_test else [])
    for segment in segments:
        try:
            inp = _segment_input(analysis, z, segment, legs, origin)
            if inp is None:
                continue
            bt = run_backtest(inp, strategy, final.costs)
        except LabError as exc:
            late.append(_segment_failure(segment, exc))
            result.segment_errors[segment] = exc.message
            continue
        bt.warnings = [*extra, *bt.warnings]
        result.backtests[segment] = bt

    validation = result.backtests.get(Segment.VALIDATION)
    cause = _costs_cause(validation) if validation is not None else None
    if cause is not None:
        late.append(cause)
    if late:
        # Decisión O5: cualquier razón de descarte hace el resultado solo ilustrativo; el
        # banner de cada backtest se actualiza para citar también estas razones.
        add_discard_reasons(analysis, late)
        updated = analysis.illustrative_banner
        for bt in result.backtests.values():
            bt.warnings = [updated, *(w for w in bt.warnings if w != banner)]
    if Segment.TEST in result.backtests:
        _record_test_run(result)
    return result
