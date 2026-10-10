"""Casos límite del motor: capital agotado, apalancamiento no reducible, entradas inválidas, z NaN."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from backtest_helpers import NO_COSTS, make_input

from mrlab.backtest.engine import run_backtest
from mrlab.domain.errors import ConfigError, DataError
from mrlab.strategy.params import StrategyParams
from mrlab.strategy.state_machine import ExitReason, PositionState


def test_capital_exhausted_forces_exit_and_blocks_new_entries():
    # largo spread 2×: +100 A a 100 y −200 B a 50; A se desploma y B sube → pérdida > capital
    params = StrategyParams(gross_exposure_target=2.0, stop_loss_pct=1.0, rearm=False)
    z = [0.0, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5]
    py = [100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    px = [50.0, 50.0, 50.0, 60.0, 60.0, 60.0, 60.0, 60.0]
    res = run_backtest(make_input(z, py, px), params, NO_COSTS)
    d = res.daily
    (t,) = res.trades
    assert d.loc[3, "equity"] == pytest.approx(10_000 - 100 * 99 - 200 * 10)
    assert math.isnan(d.loc[3, "leverage"])  # exposición con capital ≤ 0
    assert t.exit_reason == ExitReason.FORCED
    assert t.active_exit_conditions[0] == ExitReason.FORCED
    assert ExitReason.STOP_MONEY in t.active_exit_conditions
    kinds = [e.kind for e in res.events]
    assert kinds[0] == "capital_agotado"
    assert kinds.count("capital_agotado") == 2  # salida forzada + señal posterior ignorada
    assert (d.loc[4:, "q_y"] == 0).all()
    assert any("agotó" in w for w in res.warnings)


def test_integer_reduction_that_zeroes_a_leg_closes_the_position():
    params = StrategyParams(
        gross_exposure_target=1.9, integer_units=True, stop_loss_pct=0.5, rearm=False
    )
    z = [0.0, 2.5, 2.5, 2.5, 2.5, 2.5]
    py = [9_000.0, 9_000.0, 9_000.0, 10_000.0, 10_000.0, 10_000.0]
    res = run_backtest(make_input(z, py), params, NO_COSTS)
    (t,) = res.trades
    assert (t.q_y, t.q_x) == (-1.0, 190.0)
    assert (
        t.exit_reason == ExitReason.FORCED and t.exit_signal_date == res.daily.loc[3, "date"].date()
    )
    (event,) = [e for e in res.events if e.kind == "reduccion_apalancamiento"]
    assert "0 unidades" in event.message


def test_nan_z_produces_no_signals_and_only_money_or_time_exits():
    z = [math.nan, math.nan, 2.5, 2.4, math.nan, math.nan, math.nan, math.nan, math.nan]
    params = StrategyParams(max_holding_override=4)
    res = run_backtest(make_input(z), params, NO_COSTS)
    (t,) = res.trades
    assert t.entry_signal_date == res.daily.loc[2, "date"].date()
    assert t.exit_reason == ExitReason.MAX_TIME  # sin z no hay salida normal ni stop z
    assert t.exit_z is None or math.isnan(t.exit_z)
    assert res.daily.loc[7, "state"] == PositionState.CLOSING


def test_input_validation_errors_are_in_spanish():
    good = make_input([0.0] * 5)
    shifted_z = good.z.set_axis(good.z.index + pd.Timedelta(days=1))
    with pytest.raises(ConfigError, match="mismas fechas"):
        run_backtest(replace(good, z=shifted_z), StrategyParams(), NO_COSTS)
    with pytest.raises(DataError, match="precios faltantes o no positivos"):
        run_backtest(make_input([0.0] * 3, [100.0, np.nan, 100.0]), StrategyParams(), NO_COSTS)
    with pytest.raises(DataError, match="no tiene observaciones"):
        run_backtest(make_input([]), StrategyParams(), NO_COSTS)
    with pytest.raises(ConfigError, match="β"):
        run_backtest(make_input([0.0] * 3, beta=math.inf), StrategyParams(), NO_COSTS)


def test_capital_exhausted_at_the_last_close_is_reported():
    # Hallazgo: con equity ≤ 0 justo en el último cierre no había evento ni aviso.
    # Corto de spread (−50 A, +100 B) y A se dispara en el último cierre: capital −5.000.
    params = StrategyParams(stop_loss_pct=1.0, stop_z=50.0)
    z = [0.0, 2.5, 2.4, 2.6, 2.6]
    py = [100.0, 100.0, 100.0, 110.0, 400.0]
    res = run_backtest(make_input(z, py), params, NO_COSTS)
    d = res.daily
    assert d["equity"].iloc[-1] == pytest.approx(-5_000.0)
    assert any("agotó" in w for w in res.warnings)
    (event,) = [e for e in res.events if e.kind == "capital_agotado"]
    assert event.date == d["date"].iloc[-1].date()
    assert "último cierre" in event.message
    (t,) = res.trades
    assert t.open_at_end and t.exit_reason is None  # no queda cierre para ejecutar la salida


def test_capital_exhausted_warning_comes_from_the_daily_equity():
    # Cualquier cierre con capital ≤ 0 produce el aviso, aunque no haya evento registrado.
    from mrlab.backtest.engine import _result_warnings

    params = StrategyParams(stop_loss_pct=1.0, stop_z=50.0)
    z = [0.0, 2.5, 2.4, 2.6, 2.6]
    py = [100.0, 100.0, 100.0, 110.0, 400.0]
    res = run_backtest(make_input(z, py), params, NO_COSTS)
    inp = make_input([0.0] * 5)
    warnings = _result_warnings(inp, params, res.trades, [], res.daily)
    assert any("agotó" in w for w in warnings)
    healthy = run_backtest(make_input([0.0] * 5), params, NO_COSTS)
    assert not any("agotó" in w for w in healthy.warnings)
