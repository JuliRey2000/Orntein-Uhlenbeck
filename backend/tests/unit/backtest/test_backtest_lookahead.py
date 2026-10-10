"""Anti look-ahead (sección 14): cambiar el futuro no cambia nada hasta t."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from backtest_helpers import SimPair

from mrlab.backtest.engine import BacktestResult, run_backtest
from mrlab.strategy.params import CostParams, StrategyParams

PAIR = SimPair(n=600, seed=21, lam=0.1, sigma=0.015)
PARAMS = StrategyParams()
COSTS = CostParams()

ENTRY_DECISION = ("trade_id", "side", "entry_signal_date", "entry_z", "entry_date")
ENTRY_EXECUTION = ("q_y", "q_x", "price_y_entry", "price_x_entry")
EXIT_DECISION = ("exit_signal_date", "exit_z", "exit_reason", "active_exit_conditions")


def run(prices: pd.DataFrame) -> BacktestResult:
    # z se recalcula con los MISMOS parámetros fijos: al tocar precios > t, también cambia z > t
    return run_backtest(PAIR.input_for(prices), PARAMS, COSTS)


def perturb(prices: pd.DataFrame, t: int, mode: str) -> pd.DataFrame:
    out = prices.copy()
    after = out.index > out.index[t]
    k = int(after.sum())
    if mode == "escala":
        out.loc[after, "y"] *= 1.15
        out.loc[after, "x"] *= 0.90
    else:
        rng = np.random.default_rng(t)
        out.loc[after, "y"] *= np.exp(rng.normal(0.0, 0.05, k))
        out.loc[after, "x"] *= np.exp(rng.normal(0.0, 0.05, k))
    return out


@pytest.fixture(scope="module")
def base() -> BacktestResult:
    return run(PAIR.prices)


def _cut_points(base: BacktestResult) -> list[int]:
    pos = {d: i for i, d in enumerate(base.daily["date"].dt.date)}
    closed = [t for t in base.trades if not t.open_at_end]
    points = {0, 57, 250, len(base.daily) - 2}
    for t in closed[:4]:
        points.add(pos[t.entry_signal_date])  # justo antes de una entrada
        points.add(pos[t.entry_date])
        points.add(pos[t.exit_signal_date])  # justo antes de una salida
    return sorted(points)


def assert_same_until(base: BacktestResult, other: BacktestResult, t: int) -> None:
    pd.testing.assert_frame_equal(
        base.daily.iloc[: t + 1], other.daily.iloc[: t + 1], check_exact=True
    )
    cutoff = base.daily["date"].iloc[t].date()

    def events(r):
        return [e.model_dump() for e in r.events if e.date <= cutoff]

    assert events(base) == events(other)

    started_b = [tr for tr in base.trades if tr.entry_signal_date <= cutoff]
    started_o = [tr for tr in other.trades if tr.entry_signal_date <= cutoff]
    assert len(started_b) == len(started_o)
    for tb, to in zip(started_b, started_o, strict=True):
        b, o = tb.model_dump(), to.model_dump()
        for f in ENTRY_DECISION:
            assert b[f] == o[f], f
        if tb.entry_date <= cutoff:
            for f in ENTRY_EXECUTION:
                assert b[f] == o[f], f
        if tb.exit_signal_date is not None and tb.exit_signal_date <= cutoff:
            for f in EXIT_DECISION:
                assert b[f] == o[f], f
        if tb.exit_date is not None and tb.exit_date <= cutoff:
            assert b == o


def test_base_scenario_has_trades(base):
    assert sum(not t.open_at_end for t in base.trades) >= 4


@pytest.mark.parametrize("mode", ["escala", "ruido"])
def test_future_prices_do_not_change_the_past(base, mode):
    for t in _cut_points(base):
        other = run(perturb(PAIR.prices, t, mode))
        assert_same_until(base, other, t)
        # control: la perturbación sí cambia el futuro (la prueba no es trivial)
        assert not base.daily.iloc[t + 1 :].equals(other.daily.iloc[t + 1 :])


def test_detects_a_leak(base):
    """Control negativo: si z(t) usara el precio de t+1, la comparación lo detectaría."""
    t = _cut_points(base)[2]
    leaky = PAIR.input_for()
    shifted = leaky.z.shift(-1).fillna(0.0)
    other_prices = perturb(PAIR.prices, t, "escala")
    other_z = PAIR.z_of(other_prices).shift(-1).fillna(0.0)
    r1 = run_backtest(PAIR.input_for(z=shifted), PARAMS, COSTS)
    r2 = run_backtest(PAIR.input_for(other_prices, z=other_z), PARAMS, COSTS)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            r1.daily.iloc[: t + 1], r2.daily.iloc[: t + 1], check_exact=True
        )
