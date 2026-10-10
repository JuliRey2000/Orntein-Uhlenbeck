"""Costes de transacción y de mantenimiento (sección 9, C1, C2)."""

from __future__ import annotations

import numpy as np
import pytest
from backtest_helpers import NO_COSTS, make_input

from mrlab.backtest.costs import carry_costs, slippage_bps, transaction_costs
from mrlab.backtest.engine import run_backtest
from mrlab.domain.models import AssetClass
from mrlab.strategy.params import CostParams, StrategyParams

C = CostParams()


@pytest.mark.parametrize(
    ("asset_class", "expected"),
    [
        (AssetClass.CRYPTO, 10.0),
        (AssetClass.EQUITY, 5.0),
        (AssetClass.ETF, 5.0),
        (AssetClass.OTHER, 5.0),
    ],
)
def test_slippage_by_asset_class(asset_class, expected):
    assert slippage_bps(asset_class, C) == expected


def test_transaction_costs_bps_of_traded_notional():
    assert transaction_costs(10_000.0, AssetClass.EQUITY, C) == pytest.approx((5.0, 5.0))
    assert transaction_costs(10_000.0, AssetClass.CRYPTO, C) == pytest.approx((5.0, 10.0))
    assert transaction_costs(-2_000.0, AssetClass.CRYPTO, C) == pytest.approx((1.0, 2.0))


def test_carry_costs_calendar_days_and_excess_only():
    borrow1, fin1 = carry_costs(5_000.0, 10_000.0, 10_000.0, 1, C)
    assert borrow1 == pytest.approx(0.03 * 5_000.0 / 365)
    assert fin1 == 0.0  # exposición = capital: nada que financiar
    borrow3, _ = carry_costs(5_000.0, 10_000.0, 10_000.0, 3, C)
    assert borrow3 == pytest.approx(3 * borrow1)
    _, fin = carry_costs(0.0, 15_000.0, 10_000.0, 2, C)
    assert fin == pytest.approx(0.05 * 5_000.0 * 2 / 365)  # solo el exceso sobre el capital
    _, fin_under = carry_costs(0.0, 8_000.0, 10_000.0, 2, C)
    assert fin_under == 0.0
    assert carry_costs(5_000.0, 10_000.0, 10_000.0, 0, C) == (0.0, 0.0)
    assert carry_costs(0.0, 0.0, -100.0, 1, C) == (0.0, 0.0)  # sin posición no hay devengo


def test_monday_accrues_the_weekend():
    # Lunes 2024-01-01 … corto spread con β = 1: corto A (100) y largo B (50), precios constantes.
    z = [0.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    costs = CostParams(commission_bps=0, slippage_bps_equity=0, financing_rate_annual=0)
    res = run_backtest(make_input(z), StrategyParams(), costs)
    d = res.daily
    assert list(d["date"].dt.day_name()[:8]) == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Monday",
        "Tuesday",
        "Wednesday",
    ]
    # Señal el martes (fila 1), entrada el miércoles (fila 2): corto de 5.000 en A.
    daily_borrow = 0.03 * 5_000.0 / 365
    assert d.loc[2, "cost_borrow"] == 0.0
    assert d.loc[3, "cost_borrow"] == pytest.approx(daily_borrow)  # miércoles → jueves: 1 día
    assert d.loc[4, "cost_borrow"] == pytest.approx(daily_borrow)  # jueves → viernes: 1 día
    assert d.loc[5, "cost_borrow"] == pytest.approx(3 * daily_borrow)  # viernes → lunes: 3 días
    assert d.loc[6, "cost_borrow"] == pytest.approx(daily_borrow)


def test_crypto_leg_pays_10_bps_slippage_in_engine():
    z = [0.0, 2.5, 2.5, 2.5]
    costs = CostParams(borrow_rate_annual=0, financing_rate_annual=0)
    res = run_backtest(make_input(z, class_y=AssetClass.CRYPTO), StrategyParams(), costs)
    row = res.daily.loc[2]  # ejecución de la entrada: 5.000 por pierna
    assert row["cost_commission"] == pytest.approx(5_000 * 5e-4 + 5_000 * 5e-4)
    assert row["cost_slippage"] == pytest.approx(5_000 * 10e-4 + 5_000 * 5e-4)


def test_financing_only_on_exposure_above_capital():
    z = [0.0, -2.5, -2.5, -2.5, -2.5]
    fin_only = NO_COSTS.model_copy(update={"financing_rate_annual": 0.05})
    at_one = run_backtest(make_input(z), StrategyParams(), fin_only)
    assert at_one.daily["cost_financing"].sum() == 0.0  # exposición 1× = capital

    levered = run_backtest(make_input(z), StrategyParams(gross_exposure_target=1.5), fin_only)
    d = levered.daily
    assert d.loc[2, "gross_exposure"] == pytest.approx(15_000.0)
    expected = 0.05 * (15_000.0 - 10_000.0) * 1 / 365
    assert d.loc[3, "cost_financing"] == pytest.approx(expected)
    assert np.all(d["cost_borrow"] == 0.0)


# --------------------------------------------------------------------------- validación (CostParams)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("day_count", 0),  # antes: ZeroDivisionError en el motor
        ("day_count", -365),  # antes: préstamo y financiación como ingresos
        ("day_count", 3650),
        ("borrow_rate_annual", 3.0),  # 300 % anual: ¿se quiso escribir 0,03?
        ("borrow_rate_annual", -0.01),
        ("financing_rate_annual", 5.0),
        ("financing_rate_annual", float("nan")),
        ("commission_bps", -1.0),
        ("commission_bps", 5_000.0),
        ("slippage_bps_equity", float("inf")),
        ("slippage_bps_crypto", 2_000.0),
    ],
)
def test_cost_params_reject_unreasonable_values_in_spanish(field, value):
    from mrlab.domain.errors import ConfigError

    with pytest.raises(ConfigError) as info:
        CostParams(**{field: value})
    err = info.value
    assert err.code == "COSTES_INVALIDOS"
    assert f"costs.{field}" in err.message and "recibido" in err.message
    assert err.details["campo"] == f"costs.{field}"


def test_cost_params_hint_when_a_rate_looks_like_a_percentage():
    from mrlab.domain.errors import ConfigError

    with pytest.raises(ConfigError) as info:
        CostParams(borrow_rate_annual=3.0)
    assert "0,03" in info.value.message


@pytest.mark.parametrize(
    "updates",
    [
        {"day_count": 360},
        {"day_count": 365},
        {"borrow_rate_annual": 1.0, "financing_rate_annual": 0.0},
        {"commission_bps": 0.0, "slippage_bps_equity": 1_000.0, "slippage_bps_crypto": 150.0},
    ],
)
def test_cost_params_accept_reasonable_values(updates):
    costs = CostParams(**updates)
    assert all(getattr(costs, k) == v for k, v in updates.items())


def test_day_count_is_used_for_accruals():
    actual_360 = CostParams(day_count=360)
    borrow, _ = carry_costs(5_000.0, 10_000.0, 10_000.0, 1, actual_360)
    assert borrow == pytest.approx(0.03 * 5_000.0 / 360)
