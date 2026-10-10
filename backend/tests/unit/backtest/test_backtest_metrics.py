"""Métricas (sección 10, C4)."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mrlab.backtest.engine import DAILY_COLUMNS, Trade
from mrlab.backtest.metrics import (
    INSUFFICIENT_SAMPLE_WARNING,
    METRIC_HELP,
    Metrics,
    compute_metrics,
    max_drawdown,
)
from mrlab.domain.models import CalendarType
from mrlab.strategy.state_machine import ExitReason, Side

CAPITAL = 10_000.0
PROMOTIONAL = ("par rentable", "estrategia ganadora", "garantizad", "valor justo")


def daily_from_equity(equity, *, gross=None, costs=None) -> pd.DataFrame:
    equity = np.asarray(equity, dtype=float)
    n = len(equity)
    pnl_net = np.r_[0.0, np.diff(equity)]
    costs = np.zeros(n) if costs is None else np.asarray(costs, dtype=float)
    gross = pnl_net + costs if gross is None else np.asarray(gross, dtype=float)
    df = pd.DataFrame(0.0, index=range(n), columns=DAILY_COLUMNS)
    df["date"] = pd.date_range("2024-01-01", periods=n, freq="B")
    df["equity"] = equity
    df["pnl_net"] = pnl_net
    df["pnl_gross"] = gross
    df["cost_commission"] = costs
    df["cost_total"] = costs
    df["realized_cum"] = equity - equity[0]
    return df


def trade(
    net: float, bars: int = 5, open_at_end: bool = False, notional: float = 20_000.0
) -> Trade:
    return Trade(
        trade_id=1,
        side=Side.LONG,
        entry_signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_z=-2.5,
        exit_reason=None if open_at_end else ExitReason.NORMAL,
        bars_held=bars,
        q_y=1.0,
        q_x=-1.0,
        price_y_entry=100.0,
        price_x_entry=100.0,
        net_pnl=net,
        notional_traded=notional,
        open_at_end=open_at_end,
    )


def metrics(daily, trades=(), ppy=252, cal=CalendarType.EXCHANGE) -> Metrics:
    return compute_metrics(daily, list(trades), CAPITAL, ppy, cal)


def test_sharpe_with_known_returns_and_annualization():
    returns = np.array([0.01, -0.005, 0.002, 0.007, -0.003, 0.004, 0.0, -0.001])
    equity = CAPITAL * np.cumprod(np.r_[1.0, 1.0 + returns])
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    m = metrics(daily_from_equity(equity))
    assert m.sharpe == pytest.approx(expected, rel=1e-10)
    assert m.periods_per_year == 252
    assert "rf = 0" in m.sharpe_convention and "252" in m.sharpe_convention

    crypto = metrics(daily_from_equity(equity), ppy=365, cal=CalendarType.CRYPTO_7D)
    assert crypto.sharpe == pytest.approx(expected * math.sqrt(365 / 252), rel=1e-10)
    assert "365" in crypto.sharpe_convention and "cripto" in crypto.sharpe_convention

    total = equity[-1] / CAPITAL - 1
    assert m.total_return == pytest.approx(total)
    assert m.annualized_return == pytest.approx((1 + total) ** (252 / len(returns)) - 1)


def test_sharpe_undefined_without_variation():
    m = metrics(daily_from_equity([CAPITAL] * 20))
    assert m.sharpe is None
    assert m.max_drawdown == 0.0 and m.max_drawdown_duration == 0
    assert m.max_drawdown_start is None


def test_drawdown_not_recovered():
    path = [100, 110, 99, 105, 110, 120, 90, 95]
    dd, duration, start, end, recovered = max_drawdown(np.array(path, dtype=float))
    assert dd == pytest.approx(90 / 120 - 1)
    assert (start, end, duration, recovered) == (5, 7, 2, False)
    m = metrics(daily_from_equity(np.array(path) * 100.0))
    assert m.max_drawdown == pytest.approx(-0.25)
    assert m.max_drawdown_duration == 2
    assert m.max_drawdown_start == date(2024, 1, 8)  # fila 5
    assert m.max_drawdown_end == date(2024, 1, 10)  # último cierre
    assert m.max_drawdown_recovered is False


def test_drawdown_recovered_and_peak_is_last_high():
    path = [100, 120, 110, 120, 90, 100, 125, 110]
    dd, duration, start, end, recovered = max_drawdown(np.array(path, dtype=float))
    assert dd == pytest.approx(-0.25)
    # el episodio empieza en el último cierre en máximos (fila 3) y termina al superarlo (fila 6)
    assert (start, end, duration, recovered) == (3, 6, 3, True)


def test_trade_statistics_and_profit_factor():
    trades = [
        trade(300, 4),
        trade(-100, 6),
        trade(200, 8),
        trade(-50, 2),
        trade(999, 3, open_at_end=True),
    ]
    m = metrics(daily_from_equity([CAPITAL] * 10), trades)
    assert m.n_trades == 4 and m.n_open_at_end == 1
    assert m.win_rate == pytest.approx(0.5)
    assert m.avg_win == pytest.approx(250.0)
    assert m.avg_loss == pytest.approx(-75.0)
    assert m.profit_factor == pytest.approx(500 / 150)
    assert m.avg_holding_bars == pytest.approx(5.0)
    assert m.turnover == pytest.approx(5 * 20_000.0 / CAPITAL)  # incluye la abierta
    assert m.turnover_annual == pytest.approx(m.turnover * 252 / 9)


def test_profit_factor_undefined_without_losses():
    m = metrics(daily_from_equity([CAPITAL] * 5), [trade(10), trade(20)])
    assert m.profit_factor is None
    assert m.avg_loss is None and m.win_rate == 1.0
    empty = metrics(daily_from_equity([CAPITAL] * 5))
    assert empty.profit_factor is None and empty.win_rate is None and empty.avg_holding_bars is None


def test_costs_pct_of_gross():
    equity = [CAPITAL, CAPITAL + 80, CAPITAL + 160]
    m = metrics(daily_from_equity(equity, costs=[0, 20, 20]))
    assert m.gross_pnl == pytest.approx(200.0)
    assert m.total_costs == pytest.approx(40.0)
    assert m.costs_pct_of_gross == pytest.approx(0.2)
    assert m.costs_by_type["comision"] == pytest.approx(40.0)
    assert set(m.costs_by_type) == {"comision", "slippage", "prestamo_cortos", "financiacion"}

    losing = metrics(daily_from_equity([CAPITAL, CAPITAL - 50], costs=[0, 10]))
    assert losing.gross_pnl == pytest.approx(-40.0)
    assert losing.costs_pct_of_gross is None  # «no aplica» con bruto ≤ 0
    flat = metrics(daily_from_equity([CAPITAL, CAPITAL]))
    assert flat.costs_pct_of_gross is None


def test_costs_eating_the_result_warning():
    m = metrics(daily_from_equity([CAPITAL, CAPITAL - 20], gross=[0, 30], costs=[0, 50]))
    assert any("costes" in w and "PnL bruto" in w for w in m.warnings)


def test_insufficient_sample_warning():
    few = metrics(daily_from_equity([CAPITAL] * 5), [trade(10)] * 9)
    assert INSUFFICIENT_SAMPLE_WARNING in few.warnings
    assert (
        few.warnings[0]
        == "Muestra insuficiente para conclusiones estadísticas (menos de 10 operaciones)."
    )
    enough = metrics(daily_from_equity([CAPITAL] * 5), [trade(10)] * 10)
    assert INSUFFICIENT_SAMPLE_WARNING not in enough.warnings


def test_metric_help_covers_every_metric_with_one_line():
    assert set(METRIC_HELP) == set(Metrics.model_fields)
    for key, text in METRIC_HELP.items():
        assert text.strip(), key
        assert "\n" not in text, key
        assert len(text) <= 160, key
        assert not any(word in text.lower() for word in PROMOTIONAL), key


def test_exposure_and_leverage_metrics():
    df = daily_from_equity([CAPITAL] * 4)
    df["gross_exposure"] = [0.0, 10_000.0, 20_000.0, 10_000.0]
    df["leverage"] = [0.0, 1.0, 2.0, np.nan]
    m = metrics(df)
    assert m.avg_gross_exposure == pytest.approx(10_000.0 / CAPITAL)
    assert m.max_leverage_used == pytest.approx(2.0)
