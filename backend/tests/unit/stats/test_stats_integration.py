"""ADF + KPSS en niveles y diferencias e interpretación combinada (4.1, decisiones E1 y E2)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller, kpss

from mrlab.domain.errors import DataError
from mrlab.stats.integration import adf_test, integration_tests, kpss_test


def random_walk(seed: int, n: int = 1000, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", name="date")
    return pd.Series(np.log(100.0) + np.cumsum(rng.normal(0.0, vol, n)), index=idx)


def ar1(seed: int, n: int = 1000, phi: float = 0.5) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return pd.Series(x)


def test_random_walk_is_compatible_with_i1() -> None:
    res = integration_tests(random_walk(0), "RW")
    assert res.classification == "compatible_I1"
    assert not res.level_adf.rejects_null and res.level_kpss.rejects_null
    assert res.diff_adf.rejects_null and not res.diff_kpss.rejects_null
    assert res.level_kpss.pvalue_bound == "menor_que"
    assert res.diff_kpss.pvalue_bound == "mayor_que"
    assert "Compatible con I(1)" in res.explanation
    assert "p < 0,01" in res.explanation and "p > 0,10" in res.explanation
    assert res.level_adf.series == "niveles" and res.diff_adf.series == "diferencias"
    assert res.diff_adf.nobs < res.level_adf.nobs


def test_stationary_ar1_is_possibly_stationary() -> None:
    res = integration_tests(ar1(0), "AR")
    assert res.classification == "posiblemente_estacionaria"
    assert res.level_adf.rejects_null and not res.level_kpss.rejects_null
    assert res.level_kpss.pvalue_bound == "mayor_que"
    assert "Posiblemente estacionaria" in res.explanation


def test_i2_series_is_ambiguous() -> None:
    rng = np.random.default_rng(0)
    i2 = pd.Series(np.cumsum(np.cumsum(rng.normal(0.0, 0.01, 1000))))
    res = integration_tests(i2, "I2")
    assert res.classification == "ambiguo"
    assert not res.diff_adf.rejects_null  # las diferencias siguen teniendo raíz unitaria
    assert "cautela" in res.explanation


def test_classification_rates_over_seeds() -> None:
    """No depende de una semilla afortunada: tasas sobre 50 semillas fijas."""
    seeds = range(50)
    rw_ok = sum(
        integration_tests(random_walk(s), "RW").classification == "compatible_I1" for s in seeds
    )
    ar_ok = sum(
        integration_tests(ar1(s), "AR").classification == "posiblemente_estacionaria" for s in seeds
    )
    assert rw_ok >= 38  # ≥ 76 % (con nivel 5 % en cuatro pruebas se esperan ~85-90 %)
    assert ar_ok >= 43  # ≥ 86 %


def test_no_warnings_reach_the_user() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        integration_tests(random_walk(1), "RW")  # KPSS en el borde inferior (p < 0,01)
        integration_tests(ar1(1), "AR")  # KPSS en el borde superior (p > 0,10)
    assert caught == []


def test_adf_matches_statsmodels_exactly() -> None:
    x = random_walk(3)
    ours = adf_test(x)
    stat, p, lags, nobs, crit, _ = adfuller(
        x.to_numpy(), maxlag=12, regression="c", autolag="AIC", result_object=False
    )
    assert ours.statistic == stat and ours.pvalue == p
    assert ours.lags == lags and ours.nobs == nobs
    assert ours.critical_values == crit
    assert ours.test == "ADF" and ours.regression == "c"


def test_kpss_matches_statsmodels_and_reports_bounds() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, bound in ((random_walk(4), "menor_que"), (ar1(4), "mayor_que")):
            ours = kpss_test(x)
            stat, p, lags, crit = kpss(
                x.to_numpy(), regression="c", nlags="auto", result_object=False
            )
            assert ours.statistic == stat and ours.pvalue == p and ours.lags == lags
            assert ours.critical_values == crit
            assert ours.pvalue_bound == bound


def test_kpss_exact_pvalue_inside_table() -> None:
    # AR(1) muy persistente pero estacionario: el estadístico cae dentro de la tabla para esta semilla.
    found = [kpss_test(ar1(s, phi=0.97)) for s in range(40)]
    exact = [t for t in found if t.pvalue_bound == "exacto"]
    assert exact, "se esperaba algún p-value de KPSS interpolado"
    for t in exact:
        assert 0.01 < t.pvalue < 0.10
        assert t.rejects_null == (t.pvalue <= 0.05)


def test_kpss_rejection_only_when_bound_confirms_it() -> None:
    x = random_walk(5)
    assert kpss_test(x, alpha=0.05).rejects_null  # p < 0,01 ≤ 0,05
    assert not kpss_test(x, alpha=0.005).rejects_null  # p < 0,01 no confirma p ≤ 0,005
    y = ar1(5)
    assert not kpss_test(y, alpha=0.05).rejects_null
    assert not kpss_test(y, alpha=0.20).rejects_null  # p > 0,10 no confirma p ≤ 0,20


def test_invalid_inputs_raise_data_error_in_spanish() -> None:
    with pytest.raises(DataError, match="al menos 28 observaciones"):
        adf_test(random_walk(0, n=20))
    with pytest.raises(DataError, match="faltantes"):
        integration_tests(pd.Series([1.0, np.nan] + [2.0] * 100), "X")
    with pytest.raises(DataError, match="constante"):
        integration_tests(pd.Series([1.0] * 200), "X")
