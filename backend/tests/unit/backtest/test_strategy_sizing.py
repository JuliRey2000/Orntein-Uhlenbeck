"""Dimensionamiento de las piernas (7.3, S5, S7)."""

from __future__ import annotations

import pytest

from mrlab.domain.models import SpreadType
from mrlab.strategy.params import StrategyParams
from mrlab.strategy.sizing import requires_short, size_position
from mrlab.strategy.state_machine import Side

P = StrategyParams()


@pytest.mark.parametrize(
    ("side", "beta", "expected"),
    [
        (Side.LONG, 1.2, True),  # largo A, corto B
        (Side.SHORT, 1.2, True),  # corto A, largo B
        (Side.LONG, -0.5, False),  # β < 0: comprar el spread = dos largos
        (Side.SHORT, -0.5, True),  # β < 0: vender el spread = dos cortos
        (Side.LONG, 0.0, False),
    ],
)
def test_requires_short(side, beta, expected):
    assert requires_short(side, beta) is expected


def test_log_spread_formula_and_gross_equals_target():
    beta, py, px, equity = 1.5, 80.0, 40.0, 10_000.0
    res = size_position(Side.LONG, beta, SpreadType.LOG, py, px, equity, P)
    n = equity / (1 + beta)  # 4.000
    assert res.executable
    assert res.q_y == pytest.approx(n / py)
    assert res.q_x == pytest.approx(-beta * n / px)
    assert res.notional_y == pytest.approx(n)
    assert res.notional_x == pytest.approx(-beta * n)
    assert res.gross_exposure == pytest.approx(equity * P.gross_exposure_target)
    assert res.target_gross == pytest.approx(equity)
    assert res.leverage == pytest.approx(1.0)
    assert not res.reduced and res.reasons == []

    short = size_position(Side.SHORT, beta, SpreadType.LOG, py, px, equity, P)
    assert short.q_y == pytest.approx(-res.q_y)
    assert short.q_x == pytest.approx(-res.q_x)


def test_price_spread_formula():
    beta, py, px, equity = 0.8, 30.0, 25.0, 10_000.0
    res = size_position(Side.SHORT, beta, SpreadType.PRICE, py, px, equity, P)
    q_a = -equity / (py + beta * px)
    assert res.q_y == pytest.approx(q_a)
    assert res.q_x == pytest.approx(-beta * q_a)  # q_B = ∓β·q_A en unidades
    assert res.gross_exposure == pytest.approx(equity)


def test_negative_beta_both_legs_same_direction():
    res = size_position(Side.LONG, -0.5, SpreadType.LOG, 100.0, 50.0, 15_000.0, P)
    assert res.q_y > 0 and res.q_x > 0
    assert res.q_y == pytest.approx(100.0) and res.q_x == pytest.approx(100.0)
    res = size_position(Side.SHORT, -0.5, SpreadType.LOG, 100.0, 50.0, 15_000.0, P)
    assert res.q_y < 0 and res.q_x < 0


def test_max_leverage_reduces_and_records_reason():
    params = StrategyParams(gross_exposure_target=3.0, max_leverage=2.0)
    res = size_position(Side.LONG, 1.0, SpreadType.LOG, 100.0, 50.0, 10_000.0, params)
    assert res.executable and res.reduced
    assert res.target_gross == pytest.approx(30_000.0)
    assert res.gross_exposure == pytest.approx(20_000.0)
    assert res.leverage == pytest.approx(2.0)
    assert len(res.reasons) == 1 and "apalancamiento máximo" in res.reasons[0]


def test_integer_units_round_toward_zero():
    params = StrategyParams(integer_units=True)
    long_ = size_position(Side.LONG, 1.0, SpreadType.LOG, 30.0, 70.0, 10_000.0, params)
    # 5.000/30 = 166,67 → 166 ; −5.000/70 = −71,43 → −71 (hacia cero, no −72)
    assert (long_.q_y, long_.q_x) == (166.0, -71.0)
    assert long_.gross_exposure <= 10_000.0
    short = size_position(Side.SHORT, 1.0, SpreadType.LOG, 30.0, 70.0, 10_000.0, params)
    assert (short.q_y, short.q_x) == (-166.0, 71.0)


def test_leg_rounded_to_zero_is_not_executable():
    params = StrategyParams(integer_units=True)
    res = size_position(Side.LONG, 1.0, SpreadType.LOG, 100.0, 9_000.0, 10_000.0, params)
    assert not res.executable
    assert res.q_y == 0 and res.q_x == 0
    assert any("0 unidades" in r for r in res.reasons)


def test_not_executable_without_capital_or_short_permission():
    res = size_position(Side.LONG, 1.0, SpreadType.LOG, 100.0, 50.0, 0.0, P)
    assert not res.executable and "capital" in res.reasons[0]

    no_short = StrategyParams(allow_short=False)
    res = size_position(Side.LONG, 1.0, SpreadType.LOG, 100.0, 50.0, 10_000.0, no_short)
    assert not res.executable
    assert "ventas en corto" in res.reasons[0] and "ambas piernas" in res.reasons[0]
    assert "derivados" in res.reasons[0]

    ok = size_position(Side.LONG, -0.5, SpreadType.LOG, 100.0, 50.0, 10_000.0, no_short)
    assert ok.executable and ok.q_y > 0 and ok.q_x > 0


def test_zero_beta_is_not_a_pair():
    res = size_position(Side.LONG, 0.0, SpreadType.LOG, 100.0, 50.0, 10_000.0, P)
    assert not res.executable and "β = 0" in res.reasons[0]
