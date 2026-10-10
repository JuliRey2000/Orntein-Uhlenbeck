"""Pruebas del spread (sección 5): MCO solo con los datos dados, interpretación y α, β fijos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import cointegrated_log_prices

from mrlab.domain.errors import AnalysisBlocked, DataError
from mrlab.domain.models import SpreadType
from mrlab.model.spread import (
    OLS_INFERENCE_WARNING,
    SPREAD_UNITS,
    HedgeFit,
    compute_spread,
    fit_hedge_ratio,
    transform,
)


def _prices(
    alpha: float = 0.3, beta: float = 1.0, n: int = 1000, seed: int = 11
) -> tuple[pd.Series, pd.Series]:
    log_a, log_b = cointegrated_log_prices(n=n, seed=seed, beta=beta, alpha=alpha)
    idx = pd.bdate_range("2018-01-01", periods=n, name="date")
    return pd.Series(np.exp(log_a), index=idx, name="AAA"), pd.Series(
        np.exp(log_b), index=idx, name="BBB"
    )


def _num(x: float) -> str:
    """Formato de la interfaz con 4 decimales (coma decimal)."""
    return f"{x:.4f}".replace(".", ",")


def test_fit_matches_ols_with_constant_on_given_sample():
    y, x = _prices(alpha=0.3, beta=0.8)
    train_y, train_x = y.iloc[:600], x.iloc[:600]
    fit = fit_hedge_ratio(train_y, train_x, SpreadType.LOG, "AAA", "BBB")

    beta_ref, alpha_ref = np.polyfit(np.log(train_x), np.log(train_y), 1)
    assert fit.beta == pytest.approx(beta_ref, abs=1e-10)
    assert fit.alpha == pytest.approx(alpha_ref, abs=1e-10)
    assert fit.beta == pytest.approx(0.8, abs=0.05)
    assert fit.nobs == 600
    assert fit.estimation_start == train_y.index[0].date()
    assert fit.estimation_end == train_y.index[-1].date()
    assert 0.0 <= fit.r_squared <= 1.0
    assert fit.alpha_se > 0 and fit.beta_se > 0
    assert fit.units == SPREAD_UNITS[SpreadType.LOG]
    assert "formación" in fit.method
    # Solo usa lo que recibe: con toda la muestra el ajuste cambia.
    full = fit_hedge_ratio(y, x, SpreadType.LOG, "AAA", "BBB")
    assert full.beta != fit.beta and full.nobs == 1000


def test_log_interpretation_is_elasticity_with_beta_number():
    y, x = _prices(alpha=0.3, beta=0.8)
    fit = fit_hedge_ratio(y, x, SpreadType.LOG, "AAA", "BBB")
    assert "elasticidad" in fit.interpretation
    assert f"un 1 % en BBB se asocia con un {_num(fit.beta)} % en AAA" in fit.interpretation
    assert "misma dirección" not in fit.interpretation


def test_price_interpretation_is_units_of_b_per_unit_of_a():
    y, x = _prices(alpha=0.3, beta=0.8)
    fit = fit_hedge_ratio(y, x, SpreadType.PRICE, "AAA", "BBB")
    assert f"{_num(fit.beta)} unidades de BBB por cada unidad de AAA" in fit.interpretation
    assert fit.units == SPREAD_UNITS[SpreadType.PRICE]
    beta_ref, alpha_ref = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    assert fit.beta == pytest.approx(beta_ref, rel=1e-9)
    assert fit.alpha == pytest.approx(alpha_ref, rel=1e-9)


@pytest.mark.parametrize("spread_type", [SpreadType.LOG, SpreadType.PRICE])
def test_negative_beta_explains_both_legs_same_direction(spread_type):
    y, x = _prices(alpha=8.0, beta=-0.5)
    fit = fit_hedge_ratio(y, x, spread_type, "AAA", "BBB")
    assert fit.beta < 0
    assert _num(fit.beta) in fit.interpretation
    assert "misma dirección" in fit.interpretation
    assert "comprar AAA y también comprar BBB" in fit.interpretation


@pytest.mark.parametrize("spread_type", [SpreadType.LOG, SpreadType.PRICE])
def test_ols_inference_warning_always_present(spread_type):
    y, x = _prices()
    fit = fit_hedge_ratio(y, x, spread_type, "AAA", "BBB")
    assert OLS_INFERENCE_WARNING in fit.warnings
    assert "superconsistencia" in OLS_INFERENCE_WARNING
    assert "R²" in OLS_INFERENCE_WARNING and "no son válidos" in OLS_INFERENCE_WARNING


def test_log_spread_with_non_positive_prices_is_blocked():
    y, x = _prices(n=100)
    bad = y.copy()
    bad.iloc[40] = 0.0
    bad.iloc[70] = -1.0

    with pytest.raises(AnalysisBlocked) as exc:
        transform(bad, SpreadType.LOG)
    assert exc.value.code == "SPREAD_LOG_BLOQUEADO"
    assert exc.value.details["n_no_positivos"] == 2
    assert exc.value.details["primera_fecha"] == bad.index[40].date().isoformat()
    assert "estrictamente positivos" in exc.value.message

    with pytest.raises(AnalysisBlocked):
        fit_hedge_ratio(bad, x, SpreadType.LOG, "AAA", "BBB")
    good_fit = fit_hedge_ratio(y, x, SpreadType.LOG, "AAA", "BBB")
    with pytest.raises(AnalysisBlocked):
        compute_spread(y, x.where(x.index != x.index[10], -5.0), good_fit)

    # En precios no hay logaritmos: los precios ≤ 0 no bloquean.
    price_fit = fit_hedge_ratio(bad, x, SpreadType.PRICE, "AAA", "BBB")
    assert np.isfinite(compute_spread(bad, x, price_fit)).all()


def test_compute_spread_applies_fixed_alpha_beta_to_new_data():
    y, x = _prices(alpha=0.3, beta=0.8)
    fit = fit_hedge_ratio(y.iloc[:600], x.iloc[:600], SpreadType.LOG, "AAA", "BBB")
    test_y, test_x = y.iloc[600:], x.iloc[600:]

    spread_test = compute_spread(test_y, test_x, fit)
    expected = np.log(test_y) - fit.alpha - fit.beta * np.log(test_x)
    assert spread_test.name == "spread"
    assert spread_test.index.equals(test_y.index)
    np.testing.assert_allclose(spread_test.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)
    # No se re-estima: el spread de prueba no queda centrado como lo haría un MCO nuevo.
    refit = fit_hedge_ratio(test_y, test_x, SpreadType.LOG, "AAA", "BBB")
    assert abs(refit.alpha - fit.alpha) > 1e-6
    assert abs(spread_test.mean()) > 1e-6

    manual = HedgeFit(**{**fit.model_dump(), "alpha": 1.0, "beta": 2.0})
    np.testing.assert_allclose(
        compute_spread(test_y, test_x, manual).to_numpy(),
        (np.log(test_y) - 1.0 - 2.0 * np.log(test_x)).to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_training_spread_has_zero_mean_with_constant():
    y, x = _prices(alpha=0.3)
    fit = fit_hedge_ratio(y.iloc[:600], x.iloc[:600], SpreadType.LOG, "AAA", "BBB")
    assert abs(compute_spread(y.iloc[:600], x.iloc[:600], fit).mean()) < 1e-10


def test_misaligned_or_dirty_inputs_raise_data_error():
    y, x = _prices(n=200)
    with pytest.raises(DataError):
        fit_hedge_ratio(y.iloc[1:], x.iloc[:-1], SpreadType.LOG, "AAA", "BBB")
    fit = fit_hedge_ratio(y, x, SpreadType.LOG, "AAA", "BBB")
    with pytest.raises(DataError):
        compute_spread(y.iloc[:100], x.iloc[50:150], fit)
    with_nan = y.copy()
    with_nan.iloc[5] = np.nan
    with pytest.raises(DataError):
        fit_hedge_ratio(with_nan, x, SpreadType.LOG, "AAA", "BBB")
    with pytest.raises(DataError):
        fit_hedge_ratio(y, pd.Series(100.0, index=x.index), SpreadType.LOG, "AAA", "BBB")
