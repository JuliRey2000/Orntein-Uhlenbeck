"""Convención de ejecución t → t+1 (8.2), sin cortos (7.4, S9), β negativo y apalancamiento (S5, S7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from backtest_helpers import NO_COSTS, make_input

from mrlab.backtest.engine import DAILY_COLUMNS, EXECUTION_CONVENTION, run_backtest
from mrlab.domain.models import CalendarType
from mrlab.strategy.params import CostParams, StrategyParams
from mrlab.strategy.state_machine import ExitReason, PositionState, Side

P = StrategyParams()


def test_daily_columns_and_convention_text():
    res = run_backtest(make_input([0.0] * 5), P, NO_COSTS)
    assert list(res.daily.columns) == DAILY_COLUMNS
    assert res.execution_convention == EXECUTION_CONVENTION
    assert "t+1" in EXECUTION_CONVENTION
    assert res.daily["trade_id"].tolist() == [None] * 5
    assert res.daily["event"].tolist() == [""] * 5


def test_signal_at_t_executes_at_close_t_plus_1_with_that_price():
    z = [0.0, 0.5, 1.0, 2.4, 1.8, 1.2, 0.4, 0.1, 0.0]
    py = [100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0]
    px = [50.0, 50.5, 51.0, 51.5, 52.5, 52.0, 51.0, 50.0, 50.0]
    res = run_backtest(make_input(z, py, px), P, NO_COSTS)
    d, dates = res.daily, res.daily["date"].dt.date
    (t,) = res.trades
    assert t.side == Side.SHORT
    assert (t.entry_signal_date, t.entry_date) == (dates[3], dates[4])
    assert t.entry_z == 2.4
    assert (t.price_y_entry, t.price_x_entry) == (104.0, 52.5)
    # nocional decidido con el capital de t (10.000), cantidades con el precio de t+1
    assert t.q_y == pytest.approx(-5_000.0 / 104.0)
    assert t.q_x == pytest.approx(5_000.0 / 52.5)
    assert d.loc[3, "q_y"] == 0.0 and d.loc[3, "state"] == PositionState.FLAT
    assert d.loc[4, "gross_exposure"] == pytest.approx(10_000.0)
    assert d.loc[4, "pnl_gross"] == 0.0  # la posición nueva empieza a ganar/perder desde t+1
    # salida: z = 0,4 ≤ 0,5 en la fila 6 → se ejecuta al cierre de la fila 7 con su precio
    assert (t.exit_signal_date, t.exit_date) == (dates[6], dates[7])
    assert d.loc[6, "state"] == PositionState.CLOSING
    assert (t.price_y_exit, t.price_x_exit) == (101.0, 50.0)
    assert t.exit_reason == ExitReason.NORMAL and t.bars_held == 3
    expected = t.q_y * (101.0 - 104.0) + t.q_x * (50.0 - 52.5)
    assert t.gross_pnl == pytest.approx(expected)
    assert d["pnl_gross"].sum() == pytest.approx(expected)


def test_no_orders_on_last_close():
    z = [0.0, 0.0, 0.0, 0.0, 2.8]
    res = run_backtest(make_input(z), P, NO_COSTS)
    assert res.trades == [] and res.events == []
    assert res.daily["event"].iloc[-1] == ""
    assert (res.daily["q_y"] == 0).all()


def test_open_position_at_end_stays_unrealized():
    py = [100.0, 100.0, 100.0, 98.0, 97.0, 96.0]
    z = [0.0, 2.5, 2.2, 1.5, 1.0, 0.2]  # la salida (0,2) llega en el último cierre: no se ejecuta
    res = run_backtest(make_input(z, py), P, NO_COSTS)
    (t,) = res.trades
    assert t.open_at_end
    assert t.exit_signal_date is None and t.exit_date is None and t.exit_reason is None
    assert t.bars_held == 3
    last = res.daily.iloc[-1]
    assert last["state"] == PositionState.SHORT
    assert last["realized_cum"] == 0.0
    assert last["unrealized"] == pytest.approx(t.net_pnl)
    assert t.net_pnl == pytest.approx(t.q_y * (96.0 - 100.0))  # corto en A gana al bajar A
    assert res.metrics.n_trades == 0 and res.metrics.n_open_at_end == 1
    assert res.metrics.unrealized_pnl == pytest.approx(t.net_pnl)
    assert any("abierta al final" in w for w in res.warnings)


def test_entry_on_second_to_last_close_executes_on_last():
    res = run_backtest(make_input([0.0, 0.0, 0.0, -2.5, -2.5]), P, NO_COSTS)
    (t,) = res.trades
    assert t.entry_date == res.daily["date"].dt.date.iloc[-1]
    assert t.open_at_end and t.bars_held == 0


@pytest.mark.parametrize("beta", [1.0, 0.3])
def test_without_shorts_positive_beta_signals_are_logged_not_traded(beta):
    z = [0.0, 2.5, 2.6, 0.0, 0.0, -2.5, -2.7, 0.0, 0.0, 0.0]
    params = StrategyParams(allow_short=False)
    res = run_backtest(make_input(z, beta=beta), params, NO_COSTS)
    assert res.trades == []
    assert (res.daily["q_y"] == 0).all() and (res.daily["q_x"] == 0).all()
    kinds = [(e.kind, e.date) for e in res.events]
    dates = res.daily["date"].dt.date
    # un evento por episodio de señal (filas 1 y 5), anotado también en cada fila
    assert kinds == [("senal_no_ejecutable", dates[1]), ("senal_no_ejecutable", dates[5])]
    for e in res.events:
        assert "ventas en corto" in e.message and "ambas piernas" in e.message
    assert "AAA" in res.events[0].message  # corto spread vende A
    assert "BBB" in res.events[1].message  # largo spread con β > 0 vende B
    assert "no ejecutable" in res.daily.loc[2, "event"]
    assert any("no ejecutables" in w for w in res.warnings)


def test_without_shorts_negative_beta_buying_the_spread_executes():
    z = [0.0, -2.5, -1.5, -0.2, 0.0, 2.5, 2.5, 0.0]
    params = StrategyParams(allow_short=False)
    res = run_backtest(make_input(z, beta=-0.5), params, NO_COSTS)
    (t,) = res.trades
    assert t.side == Side.LONG and t.q_y > 0 and t.q_x > 0
    # vender el spread con β < 0 son dos cortos: no ejecutable
    assert [e.kind for e in res.events] == ["senal_no_ejecutable"]
    assert "AAA y BBB" in res.events[0].message


# --------------------------------------------------------------------------- β negativo
# Caso construido a mano: β = −0,5 (spread log), capital 30.000, exposición objetivo 0,5
# → G = 15.000, N = G/(1+|β|) = 10.000 → q_A = N/100 = 100 ; q_B = −β·N/50 = 100.
NEG_DATES = pd.DatetimeIndex(
    ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"], name="date"
)  # mié, jue, vie, lun, mar
NEG_PY = [100.0, 100.0, 100.0, 104.0, 106.0]
NEG_PX = [50.0, 50.0, 50.0, 47.0, 46.0]
NEG_PARAMS = StrategyParams(initial_capital=30_000.0, gross_exposure_target=0.5)


def _neg_beta_run(z, costs):
    return run_backtest(
        make_input(z, NEG_PY, NEG_PX, beta=-0.5, dates=NEG_DATES), NEG_PARAMS, costs
    )


def test_negative_beta_long_spread_buys_both_legs_hand_computed():
    z = [
        0.0,
        -2.5,
        -1.5,
        -0.3,
        0.0,
    ]  # señal fila 1, entrada fila 2, salida decidida fila 3, ejecutada fila 4
    res = _neg_beta_run(z, NO_COSTS)
    (t,) = res.trades
    assert t.side == Side.LONG
    assert (t.q_y, t.q_x) == pytest.approx((100.0, 100.0))  # ambas piernas LARGAS
    # PnL a mano: fila 3: 100·(104−100) + 100·(47−50) = 100 ; fila 4: 100·2 + 100·(−1) = 100
    assert res.daily["pnl_gross"].tolist() == pytest.approx([0, 0, 0, 100.0, 100.0])
    assert t.gross_pnl == pytest.approx(200.0)
    assert t.net_pnl == pytest.approx(200.0)
    assert res.daily["equity"].iloc[-1] == pytest.approx(30_200.0)

    costs = (
        CostParams()
    )  # 5 pb comisión + 5 pb slippage por pierna; sin cortos ni exceso de exposición
    res_c = _neg_beta_run(z, costs)
    (tc,) = res_c.trades
    entry_cost = (10_000 + 5_000) * 10e-4  # 15,0
    exit_cost = (100 * 106 + 100 * 46) * 10e-4  # 15,2
    assert tc.borrow == 0.0 and tc.financing == 0.0
    assert tc.total_costs == pytest.approx(entry_cost + exit_cost)
    assert tc.net_pnl == pytest.approx(200.0 - 30.2)
    assert res_c.daily["equity"].iloc[-1] == pytest.approx(30_000.0 + 169.8)


def test_negative_beta_short_spread_sells_both_legs_hand_computed():
    z = [0.0, 2.5, 1.5, 0.3, 0.0]
    res = _neg_beta_run(z, NO_COSTS)
    (t,) = res.trades
    assert t.side == Side.SHORT
    assert (t.q_y, t.q_x) == pytest.approx((-100.0, -100.0))  # ambas piernas CORTAS
    # fila 3: −100·4 − 100·(−3) = −100 ; fila 4: −100·2 − 100·(−1) = −100
    assert res.daily["pnl_gross"].tolist() == pytest.approx([0, 0, 0, -100.0, -100.0])
    assert t.net_pnl == pytest.approx(-200.0)

    res_c = _neg_beta_run(z, CostParams())
    (tc,) = res_c.trades
    borrow = 0.03 * (100 * 100 + 100 * 50) * 3 / 365 + 0.03 * (100 * 104 + 100 * 47) * 1 / 365
    assert tc.borrow == pytest.approx(borrow)  # vie → lun devenga 3 días naturales
    assert tc.financing == 0.0
    trans = (15_000 + 100 * 106 + 100 * 46) * 10e-4
    assert tc.commission + tc.slippage == pytest.approx(trans)
    assert tc.net_pnl == pytest.approx(-200.0 - trans - borrow)
    assert res_c.daily["equity"].iloc[-1] == pytest.approx(30_000.0 + tc.net_pnl)


# --------------------------------------------------------------------------- apalancamiento
def test_entry_size_reduced_to_max_leverage_is_logged():
    params = StrategyParams(gross_exposure_target=3.0, max_leverage=2.0)
    res = run_backtest(make_input([0.0, 2.5, 2.5, 2.5]), params, NO_COSTS)
    assert res.daily.loc[2, "gross_exposure"] == pytest.approx(20_000.0)
    (event,) = res.events
    assert event.kind == "tamano_reducido" and "apalancamiento máximo" in event.message


def test_leverage_above_max_during_trade_is_reduced_next_close_with_costs():
    params = StrategyParams(gross_exposure_target=1.9, max_leverage=2.0, stop_loss_pct=0.5)
    py = [100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0]
    z = [0.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    res = run_backtest(
        make_input(z, py), params, CostParams(borrow_rate_annual=0, financing_rate_annual=0)
    )
    d = res.daily
    q_y, q_x = -95.0, 190.0  # N = 9.500: corto 95 A a 100, largo 190 B a 50
    assert (d.loc[2, "q_y"], d.loc[2, "q_x"]) == pytest.approx((q_y, q_x))
    lev3 = d.loc[3, "leverage"]
    assert lev3 > 2.0 * 1.01
    reductions = [e for e in res.events if e.kind == "reduccion_apalancamiento"]
    assert len(reductions) == 1 and reductions[0].date == d.loc[3, "date"].date()
    factor = 2.0 * d.loc[3, "equity"] / d.loc[3, "gross_exposure"]
    assert (d.loc[4, "q_y"], d.loc[4, "q_x"]) == pytest.approx((q_y * factor, q_x * factor))
    traded = abs(q_y) * (1 - factor) * 110.0 + abs(q_x) * (1 - factor) * 50.0
    assert d.loc[4, "cost_commission"] == pytest.approx(traded * 5e-4)
    assert d.loc[4, "leverage"] == pytest.approx(2.0, rel=5e-3)
    assert "Reducción por apalancamiento" in d.loc[4, "event"]
    (t,) = res.trades
    assert (t.q_y, t.q_x) == pytest.approx(
        (q_y, q_x)
    )  # la operación conserva las cantidades de entrada


def test_crypto_pair_uses_365_periods():
    z = np.zeros(10)
    res = run_backtest(make_input(z, calendar=CalendarType.CRYPTO_7D), P, NO_COSTS)
    assert res.metrics.periods_per_year == 365
