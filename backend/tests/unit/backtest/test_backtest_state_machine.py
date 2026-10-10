"""Máquina de estados dentro del motor (7.5, S1–S4)."""

from __future__ import annotations

import numpy as np
import pytest
from backtest_helpers import NO_COSTS, SimPair, make_input

from mrlab.backtest.engine import run_backtest
from mrlab.strategy.params import CostParams, StrategyParams
from mrlab.strategy.state_machine import ExitReason, PositionState, Side

FLAT, LONG, SHORT = PositionState.FLAT, PositionState.LONG, PositionState.SHORT
CLOSING, COOLDOWN = PositionState.CLOSING, PositionState.COOLDOWN


def run(z, params=None, py=100.0, px=50.0, costs=NO_COSTS):
    res = run_backtest(make_input(z, py, px), params or StrategyParams(), costs)
    return res, res.daily["date"].dt.date


# ----------------------------------------------------------------- prioridad de salidas
# Corto spread (β = 1): señal fila 1, entrada fila 2 con q_A = −50 (A a 100). Tiempo máximo 3 → en la
# fila 5 la operación lleva 3 observaciones. Si A sube a 112 en la fila 5 el PnL es −600 ≤ −500.
@pytest.mark.parametrize(
    ("z5", "py5", "expected_active", "applied"),
    [
        (
            3.6,
            112.0,
            [ExitReason.STOP_MONEY, ExitReason.STOP_Z, ExitReason.MAX_TIME],
            ExitReason.STOP_MONEY,
        ),
        (
            0.3,
            112.0,
            [ExitReason.STOP_MONEY, ExitReason.MAX_TIME, ExitReason.NORMAL],
            ExitReason.STOP_MONEY,
        ),
        (3.6, 100.0, [ExitReason.STOP_Z, ExitReason.MAX_TIME], ExitReason.STOP_Z),
        (0.3, 100.0, [ExitReason.MAX_TIME, ExitReason.NORMAL], ExitReason.MAX_TIME),
    ],
)
def test_exit_priority_records_all_active_conditions(z5, py5, expected_active, applied):
    z = [0.0, 2.5, 2.6, 2.7, 2.8, z5, 2.0, 2.0]
    py = [100.0] * 5 + [py5] * 3
    params = StrategyParams(max_holding_override=3, cooldown_bars=0, rearm=False)
    res, dates = run(z, params, py=py)
    t = res.trades[0]
    assert t.active_exit_conditions == expected_active
    assert t.exit_reason == applied
    assert (t.exit_signal_date, t.exit_date) == (dates[5], dates[6])
    assert t.exit_z == z5
    assert res.daily.loc[5, "state"] == CLOSING
    if len(expected_active) > 1:
        assert "condiciones activas" in res.daily.loc[5, "event"]


def test_money_stop_uses_net_pnl_including_costs():
    # Pérdida bruta 495 (< 500) + costes de entrada 10 → neto −505 → stop monetario.
    z = [0.0, 2.5, 2.5, 2.5, 2.5, 2.5]
    py = [100.0, 100.0, 100.0, 109.9, 109.9, 109.9]
    costs = CostParams(borrow_rate_annual=0, financing_rate_annual=0)
    res, dates = run(z, StrategyParams(rearm=False), py=py, costs=costs)
    t = res.trades[0]
    assert t.exit_reason == ExitReason.STOP_MONEY and t.exit_signal_date == dates[3]
    res_free, _ = run(z, StrategyParams(rearm=False), py=py)
    assert res_free.trades[0].exit_reason is None  # sin costes no se alcanza el stop


# ----------------------------------------------------------------- nunca dos operaciones
@pytest.mark.parametrize(
    "params",
    [
        StrategyParams(),
        StrategyParams(rearm=False, cooldown_bars=0),
        StrategyParams(rearm=False, cooldown_bars=3, max_holding_override=4),
        StrategyParams(entry_z=1.0, exit_z=0.2, stop_z=1.8, max_holding_override=6),
    ],
)
def test_never_two_open_trades_and_no_entries_while_blocked(params):
    rng = np.random.default_rng(3)
    z = np.zeros(1500)
    for i in range(1, len(z)):
        z[i] = 0.85 * z[i - 1] + rng.normal(0.0, 1.2)
    pair = SimPair(n=1500, seed=11)
    inp = make_input(z, pair.prices["y"].to_numpy(), pair.prices["x"].to_numpy())
    res = run_backtest(inp, params, CostParams())
    d = res.daily
    trades = res.trades
    assert len(trades) >= 20
    for prev, nxt in zip(trades, trades[1:], strict=False):
        assert prev.exit_date is not None
        assert nxt.entry_signal_date >= prev.exit_date  # decidida tras ejecutar la salida
        assert nxt.entry_date > prev.exit_date
    # en cada cierre hay a lo sumo una operación con posición
    row_trades = d.loc[d["q_y"] != 0, "trade_id"]
    for i, tid in row_trades.items():
        t = trades[tid - 1]
        assert t.entry_date <= d.loc[i, "date"].date()
        assert t.exit_date is None or d.loc[i, "date"].date() < t.exit_date
    # ninguna señal de entrada con posición, en CERRANDO o en ENFRIAMIENTO
    blocked = d["state"].isin([LONG.value, SHORT.value, CLOSING.value, COOLDOWN.value])
    assert not d.loc[blocked, "event"].str.contains("Señal de entrada").any()
    for t in trades:
        row = d.index[d["date"].dt.date == t.entry_signal_date][0]
        assert d.loc[row, "state"] == FLAT and d.loc[row, "q_y"] == 0 and d.loc[row, "q_x"] == 0
    assert (d["state"] == CLOSING.value).sum() == sum(not t.open_at_end for t in trades)


# ----------------------------------------------------------------- enfriamiento (S3)
def test_cooldown_after_stop_is_exact():
    # señal 1, entrada 2, stop z decidido en 3, salida ejecutada en e = 4; z sigue en 3,0
    z = [0.0, 2.5, 3.0, 3.6] + [3.0] * 16
    params = StrategyParams(cooldown_bars=5, rearm=False)
    res, dates = run(z, params)
    first, second = res.trades[:2]
    assert first.exit_reason == ExitReason.STOP_Z and first.exit_date == dates[4]
    e = 4
    assert list(res.daily.loc[e : e + 4, "state"]) == [COOLDOWN] * 5
    assert res.daily.loc[e + 5, "state"] == FLAT
    assert (second.entry_signal_date, second.entry_date) == (dates[e + 5], dates[e + 6])
    # 5 observaciones completas (e+1..e+5) sin posición
    assert (res.daily.loc[e : e + 5, "q_y"] == 0).all()
    ignored = [ev for ev in res.events if ev.kind == "senal_ignorada_enfriamiento"]
    assert len(ignored) == 1 and ignored[0].date == dates[e]


def test_money_stop_also_triggers_cooldown():
    z = [0.0, 2.5, 2.5, 2.5] + [2.5] * 12
    py = [100.0, 100.0, 100.0, 112.0] + [112.0] * 12
    res, dates = run(z, StrategyParams(rearm=False, cooldown_bars=5), py=py)
    first, second = res.trades[:2]
    assert first.exit_reason == ExitReason.STOP_MONEY and first.exit_date == dates[4]
    assert second.entry_signal_date == dates[9]


@pytest.mark.parametrize("exit_case", ["max_time", "normal"])
def test_no_cooldown_after_max_time_or_normal_exit(exit_case):
    if exit_case == "max_time":
        z = [0.0, 2.5] + [2.5] * 10
        params = StrategyParams(max_holding_override=3, rearm=False)
        e = 6  # entrada 2, tiempo máximo en 5, salida ejecutada en 6
    else:
        z = [0.0, 2.5, 2.5, 0.3, 2.5, 2.5, 2.5]
        params = StrategyParams(rearm=False)
        e = 4
    res, dates = run(z, params)
    assert res.daily.loc[e, "state"] == FLAT
    assert res.trades[1].entry_signal_date == dates[e]  # se puede decidir en el mismo cierre e
    assert res.trades[1].entry_date == dates[e + 1]


def test_zero_cooldown_allows_entry_decision_at_exit_close():
    z = [0.0, 2.5, 3.0, 3.6] + [3.0] * 4
    res, dates = run(z, StrategyParams(cooldown_bars=0, rearm=False))
    assert res.daily.loc[4, "state"] == FLAT
    assert res.trades[1].entry_signal_date == dates[4]


# ----------------------------------------------------------------- rearme (S2)
STOP_THEN = [0.0, 2.5, 3.0, 3.6]  # corto con stop z decidido en 3 y ejecutado en 4


def test_rearm_blocks_same_side_until_z_returns_inside_band():
    z = STOP_THEN + [3.0] * 12
    res, dates = run(z, StrategyParams())
    assert len(res.trades) == 1
    blocked = [ev for ev in res.events if ev.kind == "rearme_bloqueado"]
    assert len(blocked) == 1 and blocked[0].date == dates[9]  # primer cierre tras el enfriamiento
    assert "rearme" in res.daily.loc[12, "event"]


def test_rearm_released_after_return_to_band_then_same_side_reenters():
    z = STOP_THEN + [3.0] * 8 + [1.5, 2.5, 2.5, 2.5]  # vuelve a 1,5 en la fila 12
    res, dates = run(z, StrategyParams())
    assert "Rearme liberado" in res.daily.loc[12, "event"]
    second = res.trades[1]
    assert second.side == Side.SHORT
    assert (second.entry_signal_date, second.entry_date) == (dates[13], dates[14])


def test_rearm_does_not_block_opposite_side():
    z = STOP_THEN + [3.0] * 8 + [-2.5, -2.5, -2.5]  # salta al lado contrario en la fila 12
    res, dates = run(z, StrategyParams())
    second = res.trades[1]
    assert second.side == Side.LONG and second.entry_signal_date == dates[12]


def test_rearm_also_applies_after_max_time_without_cooldown():
    z = [0.0, 2.5] + [2.5] * 10 + [1.0, 2.5, 2.5]
    params = StrategyParams(max_holding_override=3)
    res, dates = run(z, params)
    first = res.trades[0]
    assert first.exit_reason == ExitReason.MAX_TIME and first.exit_date == dates[6]
    assert res.daily.loc[6, "state"] == FLAT  # sin enfriamiento
    blocked = [ev for ev in res.events if ev.kind == "rearme_bloqueado"]
    assert blocked and blocked[0].date == dates[6]
    assert res.trades[1].entry_signal_date == dates[13]  # tras volver a 1,0 en la fila 12


def test_rearm_disabled_reenters_after_cooldown():
    z = STOP_THEN + [3.0] * 12
    res, dates = run(z, StrategyParams(rearm=False))
    assert res.trades[1].entry_signal_date == dates[9]


# ----------------------------------------------------------------- salidas direccionales (S1)
def test_short_exits_when_z_jumps_through_mean():
    z = [0.0, 2.5, 2.5, -1.0, -1.0, -1.0]
    res, dates = run(z)
    t = res.trades[0]
    assert t.exit_reason == ExitReason.NORMAL
    assert (t.exit_signal_date, t.exit_date) == (dates[3], dates[4])


def test_long_exits_when_z_jumps_through_mean():
    z = [0.0, -2.5, -2.5, 1.2, 1.2, 1.2]
    res, _ = run(z)
    assert res.trades[0].exit_reason == ExitReason.NORMAL
    assert not res.trades[0].open_at_end


def test_no_reversal_on_the_same_close():
    # el corto sale por salida normal cuando z cae a −2,5: la entrada larga se decide en el
    # cierre de la ejecución de la salida (no en el de la señal) y se ejecuta en el siguiente
    z = [0.0, 2.5, 2.5, -2.5, -2.5, -2.5, -2.5]
    res, dates = run(z)
    first, second = res.trades
    assert first.exit_signal_date == dates[3] and first.exit_date == dates[4]
    assert res.daily.loc[3, "state"] == CLOSING
    assert (second.entry_signal_date, second.entry_date) == (dates[4], dates[5])
