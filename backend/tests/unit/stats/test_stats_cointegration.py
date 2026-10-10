"""Engle–Granger: coincidencia con statsmodels, dirección, BH y detección sintética (4.2, 4.3, 14)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import cointegrated_log_prices
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

from mrlab.domain.errors import ConfigError, DataError
from mrlab.stats.cointegration import (
    COLLINEAR_WARNING,
    Direction,
    EGResult,
    choose_direction,
    eg_decision,
    engle_granger,
    engle_granger_both,
)
from mrlab.stats.multitest import benjamini_hochberg, benjamini_yekutieli

A, B = Direction.A_ON_B, Direction.B_ON_A


def independent_walks(seed: int, n: int = 1000, vol: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = np.log(100.0) + np.cumsum(rng.normal(0.0, vol, n))
    b = np.log(100.0) + np.cumsum(rng.normal(0.0, vol, n))
    return a, b


def ma_residual_pair(seed: int = 11, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Spread AR(1) con innovaciones MA(1): AIC elige varios rezagos (prueba no trivial de rezagos)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 1)
    u = (e[1:] - 0.7 * e[:-1]) * 0.01
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + u[t]
    b = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    return b + x, b


PAIRS = {
    "cointegrado_fuerte": lambda: cointegrated_log_prices(n=1000, seed=3),
    "cointegrado_debil": lambda: cointegrated_log_prices(n=500, seed=4, lam=0.02),
    "cointegrado_muy_lento": lambda: cointegrated_log_prices(n=1000, seed=5, lam=0.01),
    "residuo_ma": ma_residual_pair,
    "paseos_independientes_7": lambda: independent_walks(7),
    "paseos_independientes_8": lambda: independent_walks(8),
}


def _eg(y: np.ndarray, x: np.ndarray, **kw) -> EGResult:
    return engle_granger(
        pd.Series(y), pd.Series(x), direction=A, dependent="Y", independent="X", **kw
    )


@pytest.mark.parametrize("name", list(PAIRS))
def test_engle_granger_is_exactly_statsmodels_coint(name: str) -> None:
    y, x = PAIRS[name]()
    ours = _eg(y, x)
    stat, pvalue, crit = coint(y, x, trend="c", maxlag=12, autolag="aic")

    assert ours.statistic == stat
    assert ours.pvalue == pvalue
    assert ours.critical_values == {"1%": crit[0], "5%": crit[1], "10%": crit[2]}
    assert ours.critical_values["1%"] < ours.critical_values["5%"] < ours.critical_values["10%"]
    assert (ours.trend, ours.maxlag, ours.autolag, ours.nobs) == ("c", 12, "aic", len(y))
    assert ours.warning is None


@pytest.mark.parametrize("name", list(PAIRS))
def test_lags_and_hedge_match_the_internal_step_of_coint(name: str) -> None:
    """Paso interno de coint: MCO y ~ const + x y ADF sin constante sobre los residuos."""
    y, x = PAIRS[name]()
    ours = _eg(y, x)
    ols = OLS(y, add_constant(x)).fit()
    stat, _, lags, *_ = adfuller(
        ols.resid, maxlag=12, autolag="aic", regression="n", result_object=False
    )

    assert np.isclose(stat, ours.statistic, rtol=1e-10, atol=0)  # mismo paso que coint
    assert ours.lags == lags
    assert np.isclose(ours.alpha_ols, ols.params[0], rtol=1e-10)
    assert np.isclose(ours.beta_ols, ols.params[1], rtol=1e-10)


def test_lag_selection_is_not_trivial_across_pairs() -> None:
    lags = {_eg(*PAIRS[name]()).lags for name in PAIRS}
    assert len(lags) > 1 and max(lags) >= 2


@pytest.mark.parametrize("name", list(PAIRS))
@pytest.mark.parametrize("regression", ["n", "c"])
def test_pvalue_is_not_an_ordinary_adf_on_residuals(name: str, regression: str) -> None:
    """Prohibido: el p de un ADF ordinario sobre residuos estimados es demasiado optimista."""
    y, x = PAIRS[name]()
    ours = _eg(y, x)
    resid = OLS(y, add_constant(x)).fit().resid
    naive_p = adfuller(resid, maxlag=12, autolag="aic", regression=regression, result_object=False)[
        1
    ]
    assert ours.pvalue != naive_p
    assert ours.pvalue > naive_p  # MacKinnon para residuos estimados es más exigente


def test_naive_adf_would_declare_spurious_evidence() -> None:
    """Dos paseos independientes: el ADF ordinario «encuentra» cointegración; Engle–Granger no."""
    y, x = independent_walks(7)
    resid = OLS(y, add_constant(x)).fit().resid
    naive_p = adfuller(resid, maxlag=12, autolag="aic", regression="n", result_object=False)[1]
    assert naive_p < 0.05 < _eg(y, x).pvalue


def test_naive_adf_false_positive_rate_is_much_higher() -> None:
    naive = eg = 0
    for seed in range(1000, 1100):
        y, x = independent_walks(seed)
        resid = OLS(y, add_constant(x)).fit().resid
        naive += (
            adfuller(resid, maxlag=12, autolag="aic", regression="n", result_object=False)[1]
            <= 0.05
        )
        eg += _eg(y, x).pvalue <= 0.05
    assert eg <= 10  # tamaño nominal ≈ 5 %
    assert naive >= 3 * max(eg, 1)


def test_engle_granger_both_directions() -> None:
    fa, fb = cointegrated_log_prices(n=800, seed=1, beta=0.8)
    idx = pd.date_range("2016-01-01", periods=800, freq="B", name="date")
    sa, sb = pd.Series(fa, idx), pd.Series(fb, idx)
    res = engle_granger_both(sa, sb, "KO", "PEP")

    assert set(res) == {A, B}
    assert (res[A].dependent, res[A].independent, res[A].direction) == ("KO", "PEP", A)
    assert (res[B].dependent, res[B].independent, res[B].direction) == ("PEP", "KO", B)
    assert res[A].pvalue == coint(fa, fb, trend="c", maxlag=12, autolag="aic")[1]
    assert res[B].pvalue == coint(fb, fa, trend="c", maxlag=12, autolag="aic")[1]
    assert res[A].beta_ols == pytest.approx(0.8, abs=0.05)


def test_collinear_series_are_flagged_without_leaking_warnings() -> None:
    _, x = independent_walks(0)
    y = 1.0 + 2.0 * x
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _eg(y, x)
    assert caught == []
    assert res.statistic == -np.inf and res.pvalue == 0.0 and res.lags == 0
    assert res.warning is not None and "no es fiable" in res.warning
    assert res.beta_ols == pytest.approx(2.0) and res.alpha_ols == pytest.approx(1.0)


def test_invalid_inputs_raise_data_error() -> None:
    y, x = independent_walks(0, n=200)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    with pytest.raises(DataError, match="alineadas"):
        engle_granger(
            pd.Series(y, idx),
            pd.Series(x, idx + pd.Timedelta(days=1)),
            direction=A,
            dependent="Y",
            independent="X",
        )
    with pytest.raises(DataError, match="mismo número"):
        _eg(y, x[:-1])
    with pytest.raises(DataError, match="al menos 26"):
        _eg(y[:20], x[:20])
    y_nan = y.copy()
    y_nan[5] = np.nan
    with pytest.raises(DataError, match="faltantes"):
        _eg(y_nan, x)


# ---------------------------------------------------------------- dirección y decisión


def fake_result(direction: Direction, pvalue: float) -> EGResult:
    dep, ind = ("KO", "PEP") if direction == A else ("PEP", "KO")
    return EGResult(
        direction=direction,
        dependent=dep,
        independent=ind,
        statistic=-3.0,
        pvalue=pvalue,
        critical_values={"1%": -3.9, "5%": -3.34, "10%": -3.04},
        nobs=600,
        lags=1,
        maxlag=12,
        alpha_ols=0.1,
        beta_ols=0.9,
    )


def results(pa: float, pb: float) -> dict[Direction, EGResult]:
    return {A: fake_result(A, pa), B: fake_result(B, pb)}


def test_choose_direction() -> None:
    assert choose_direction(results(0.03, 0.01)) == B
    assert choose_direction(results(0.01, 0.03)) == A
    assert choose_direction(results(0.02, 0.02)) == A  # empate → A sobre B
    assert choose_direction(results(0.01, 0.03), override=B) == B
    with pytest.raises(ConfigError):
        choose_direction({})


def test_decision_default_family_is_the_two_directions() -> None:
    dec = eg_decision(results(0.04, 0.012))
    _, adj = benjamini_hochberg([0.04, 0.012])
    assert dec.direction == B and dec.direction_was_auto
    assert dec.p_raw == {A: 0.04, B: 0.012}
    assert dec.p_adjusted == {A: adj[0], B: adj[1]}
    assert dec.p_adjusted[B] == pytest.approx(0.024)
    assert dec.evidence and dec.family_size == 2
    msg = dec.message
    assert "PEP sobre KO" in msg and "automáticamente" in msg and "menor p-value" in msg
    assert "(0,0240) no supera el nivel 0,05" in msg and "hay evidencia de cointegración" in msg


def test_decision_uses_adjusted_pvalue_not_raw() -> None:
    dec = eg_decision(results(0.03, 0.04))  # p ajustados = 0,04 y 0,04
    assert dec.evidence
    dec = eg_decision(results(0.03, 0.9))  # p ajustado de A = 0,06 > 0,05 aunque p crudo = 0,03
    assert dec.direction == A and dec.p_adjusted[A] == pytest.approx(0.06)
    assert not dec.evidence
    assert "(0,0600) supera el nivel 0,05" in dec.message and "insuficiente" in dec.message


def test_decision_alpha_level_and_boundary() -> None:
    assert not eg_decision(results(0.03, 0.9)).evidence
    assert eg_decision(results(0.03, 0.9), alpha_level=0.10).evidence
    assert eg_decision(results(0.025, 0.9)).evidence  # p ajustado = 0,05 exacto → evidencia (≤)


def test_decision_override_is_reported() -> None:
    dec = eg_decision(results(0.01, 0.3), override=B)
    assert dec.direction == B and not dec.direction_was_auto
    assert not dec.evidence
    assert "fijada por el usuario" in dec.message and "PEP sobre KO" in dec.message


def test_decision_with_universe_family() -> None:
    rng = np.random.default_rng(0)
    others = list(rng.uniform(0, 1, 98))
    family = [0.0004, 0.002, *others]
    dec = eg_decision(results(0.0004, 0.002), family_pvalues=family)
    _, adj = benjamini_hochberg(family)
    assert dec.p_adjusted == {A: adj[0], B: adj[1]}
    assert dec.family_size == 100
    assert dec.direction == A and dec.evidence
    assert "100 pruebas del universo" in dec.message
    # Más pruebas en la familia → p ajustado mayor que con m = 2.
    assert dec.p_adjusted[A] > eg_decision(results(0.0004, 0.002)).p_adjusted[A]


def test_decision_universe_family_must_start_with_the_pair() -> None:
    with pytest.raises(ConfigError, match="family_pvalues"):
        eg_decision(results(0.01, 0.02), family_pvalues=[0.02, 0.01, 0.5])
    with pytest.raises(ConfigError, match="family_pvalues"):
        eg_decision(results(0.01, 0.02), family_pvalues=[0.01])


def test_decision_by_method() -> None:
    dec = eg_decision(results(0.01, 0.2), fdr_method="by")
    _, adj = benjamini_yekutieli([0.01, 0.2])
    assert dec.p_adjusted == {A: adj[0], B: adj[1]}
    assert "Benjamini–Yekutieli" in dec.message


def test_decision_validates_inputs() -> None:
    with pytest.raises(ConfigError):
        eg_decision({A: fake_result(A, 0.01)})
    with pytest.raises(ConfigError):
        eg_decision(results(0.01, 0.02), alpha_level=0.0)
    with pytest.raises(ConfigError):
        eg_decision(results(0.01, 0.02), fdr_method="bonferroni")


def test_collinear_pair_is_never_evidence() -> None:
    """p = 0 por colinealidad es un artefacto numérico, no evidencia de cointegración."""
    _, x = independent_walks(0)
    y = 1.0 + 2.0 * x
    dec = eg_decision(engle_granger_both(pd.Series(y), pd.Series(x), "SPY", "SPY2"))
    assert dec.p_adjusted[dec.direction] == 0.0
    assert not dec.evidence
    assert "pero no se considera evidencia" in dec.message
    assert COLLINEAR_WARNING in dec.message


def test_decision_message_has_no_promotional_language() -> None:
    for pa, pb in ((0.001, 0.002), (0.6, 0.7)):
        msg = eg_decision(results(pa, pb)).message.lower()
        for banned in ("rentable", "ganadora", "garantizad", "valor justo"):
            assert banned not in msg


# ---------------------------------------------------------------- detección sintética (sección 14)


def test_synthetic_cointegrated_pairs_detected_in_90_percent() -> None:
    hits = 0
    for seed in range(100):
        fa, fb = cointegrated_log_prices(n=1000, seed=seed, lam=0.1, sigma=0.01, beta=1.0)
        hits += eg_decision(engle_granger_both(pd.Series(fa), pd.Series(fb), "A", "B")).evidence
    assert hits >= 90, f"detectados {hits}/100"  # medido: 100/100


def test_independent_random_walks_rejected_in_90_percent() -> None:
    rejected = 0
    for seed in range(100):
        fa, fb = independent_walks(seed, n=1000, vol=0.01)
        rejected += not eg_decision(
            engle_granger_both(pd.Series(fa), pd.Series(fb), "A", "B")
        ).evidence
    assert rejected >= 90, f"rechazados {rejected}/100"  # medido: 93/100
