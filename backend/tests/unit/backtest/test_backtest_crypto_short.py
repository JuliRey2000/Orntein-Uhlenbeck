"""Corto de cripto spot simulado (sección 7.4): el resultado debe avisarlo."""

from __future__ import annotations

import pytest
from backtest_helpers import NO_COSTS, make_input

from mrlab.backtest.engine import run_backtest
from mrlab.domain.models import AssetClass
from mrlab.strategy.params import StrategyParams

# Corto de spread con z = +2,5 (vende y, compra x); largo de spread con z = −2,5 (compra y, vende x).
SHORT_SPREAD = [0.0, 2.5, 2.4, 2.4, 0.0, 0.0]
LONG_SPREAD = [0.0, -2.5, -2.4, -2.4, 0.0, 0.0]


def crypto_warnings(res) -> list[str]:
    return [w for w in res.warnings if "cripto spot" in w]


@pytest.mark.parametrize(
    ("z", "class_y", "class_x"),
    [
        (SHORT_SPREAD, AssetClass.CRYPTO, AssetClass.ETF),  # corto en y (cripto)
        (LONG_SPREAD, AssetClass.ETF, AssetClass.CRYPTO),  # corto en x (cripto)
        (SHORT_SPREAD, AssetClass.CRYPTO, AssetClass.CRYPTO),
    ],
)
def test_simulated_crypto_short_is_warned(z, class_y, class_x):
    res = run_backtest(make_input(z, class_y=class_y, class_x=class_x), StrategyParams(), NO_COSTS)
    assert len(res.trades) == 1
    (warning,) = crypto_warnings(res)
    assert "derivados" in warning and "perpetuos" in warning and "funding" in warning
    assert "aproximación" in warning
    short_symbol = "AAA" if res.trades[0].q_y < 0 else "BBB"
    assert short_symbol in warning


@pytest.mark.parametrize(
    ("z", "class_y", "class_x"),
    [
        (LONG_SPREAD, AssetClass.CRYPTO, AssetClass.ETF),  # la cripto solo se compra
        (SHORT_SPREAD, AssetClass.ETF, AssetClass.CRYPTO),
        (SHORT_SPREAD, AssetClass.EQUITY, AssetClass.EQUITY),  # sin cripto
        ([0.0] * 6, AssetClass.CRYPTO, AssetClass.CRYPTO),  # sin operaciones
    ],
)
def test_no_crypto_short_warning_when_crypto_is_not_shorted(z, class_y, class_x):
    res = run_backtest(make_input(z, class_y=class_y, class_x=class_x), StrategyParams(), NO_COSTS)
    assert crypto_warnings(res) == []


def test_negative_beta_long_spread_buys_both_legs_without_warning():
    # β < 0: «comprar el spread» compra ambas piernas, aunque sean cripto.
    inp = make_input(LONG_SPREAD, beta=-0.5, class_y=AssetClass.CRYPTO, class_x=AssetClass.CRYPTO)
    res = run_backtest(inp, StrategyParams(), NO_COSTS)
    (t,) = res.trades
    assert t.q_y > 0 and t.q_x > 0
    assert crypto_warnings(res) == []
