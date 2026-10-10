"""Contabilidad del motor (sección 14, C3): bruto − costes = neto, capital, realizado + no realizado."""

from __future__ import annotations

import numpy as np
import pytest
from backtest_helpers import SimPair

from mrlab.backtest.engine import run_backtest
from mrlab.strategy.params import CostParams, StrategyParams

COST_COLUMNS = ["cost_commission", "cost_slippage", "cost_borrow", "cost_financing"]
TRADE_COST_FIELDS = {
    "cost_commission": "commission",
    "cost_slippage": "slippage",
    "cost_borrow": "borrow",
    "cost_financing": "financing",
}
ABS = 1e-7


@pytest.fixture(scope="module")
def result():
    pair = SimPair(n=900, seed=5, beta=0.8, lam=0.12, sigma=0.015)
    inp = pair.input_for()
    # Forzar una operación abierta al final: z vuelve a 0 (cierra lo que haya, deja pasar un
    # posible enfriamiento y libera el rearme) y en los 3 últimos cierres se queda en 2,6: señal en
    # n−3, entrada en n−2, y en n−1 (último cierre) ya no se crean órdenes.
    z = inp.z.copy()
    z.iloc[-12:-3] = 0.0
    z.iloc[-3:] = 2.6
    inp = pair.input_for(z=z)
    params = StrategyParams(gross_exposure_target=1.5)  # exposición > capital → financiación
    return run_backtest(inp, params, CostParams())


def test_scenario_is_rich_enough(result):
    closed = [t for t in result.trades if not t.open_at_end]
    assert len(closed) >= 10
    assert result.trades[-1].open_at_end
    assert sum(t.open_at_end for t in result.trades) == 1
    for col in COST_COLUMNS:
        assert result.daily[col].sum() > 0, col
    assert len({t.exit_reason for t in closed}) >= 2


def test_daily_identities(result):
    d = result.daily
    init = result.params.initial_capital
    np.testing.assert_allclose(d[COST_COLUMNS].sum(axis=1), d["cost_total"], rtol=0, atol=ABS)
    np.testing.assert_allclose(d["pnl_gross"] - d["cost_total"], d["pnl_net"], rtol=0, atol=ABS)
    assert d["pnl_gross"].sum() - d["cost_total"].sum() == pytest.approx(
        d["pnl_net"].sum(), abs=ABS
    )
    assert d["equity"].iloc[-1] == pytest.approx(init + d["pnl_net"].sum(), abs=ABS)
    np.testing.assert_allclose(d["equity"], init + d["pnl_net"].cumsum(), rtol=0, atol=ABS)
    # realizado + no realizado = PnL total en CADA fila
    np.testing.assert_allclose(
        d["realized_cum"] + d["unrealized"], d["equity"] - init, rtol=0, atol=ABS
    )
    # suma de costes por tipo = coste total
    assert sum(d[c].sum() for c in COST_COLUMNS) == pytest.approx(d["cost_total"].sum(), abs=ABS)


def test_trades_reconcile_with_daily(result):
    d = result.daily
    trades = result.trades
    closed = [t for t in trades if not t.open_at_end]
    (open_trade,) = [t for t in trades if t.open_at_end]
    realized, unrealized = d["realized_cum"].iloc[-1], d["unrealized"].iloc[-1]
    assert realized == pytest.approx(sum(t.net_pnl for t in closed), abs=ABS)
    assert unrealized == pytest.approx(open_trade.net_pnl, abs=ABS)
    assert sum(t.net_pnl for t in trades) == pytest.approx(realized + unrealized, abs=ABS)

    # todo PnL y todo coste se atribuye a una operación
    unattributed = d[d["trade_id"].isna()]
    assert (unattributed[["pnl_gross", "cost_total"]] == 0).all().all()
    by_trade = d.dropna(subset=["trade_id"]).groupby("trade_id")
    sums = by_trade[["pnl_gross", "cost_total", "pnl_net", *COST_COLUMNS]].sum()
    for t in trades:
        row = sums.loc[t.trade_id]
        assert t.gross_pnl - t.total_costs == pytest.approx(t.net_pnl, abs=ABS)
        assert t.commission + t.slippage + t.borrow + t.financing == pytest.approx(
            t.total_costs, abs=ABS
        )
        assert row["pnl_gross"] == pytest.approx(t.gross_pnl, abs=ABS)
        assert row["pnl_net"] == pytest.approx(t.net_pnl, abs=ABS)
        for col, attr in TRADE_COST_FIELDS.items():
            assert row[col] == pytest.approx(getattr(t, attr), abs=ABS)
        assert t.return_on_capital == pytest.approx(t.net_pnl / result.params.initial_capital)


def test_metrics_match_ledger(result):
    d, m = result.daily, result.metrics
    init = result.params.initial_capital
    assert m.final_equity == pytest.approx(d["equity"].iloc[-1])
    assert m.net_pnl == pytest.approx(m.final_equity - init, abs=ABS)
    assert m.gross_pnl - m.total_costs == pytest.approx(m.net_pnl, abs=ABS)
    assert sum(m.costs_by_type.values()) == pytest.approx(m.total_costs, abs=ABS)
    assert m.realized_pnl + m.unrealized_pnl == pytest.approx(m.net_pnl, abs=ABS)
    assert m.n_trades + m.n_open_at_end == len(result.trades)
