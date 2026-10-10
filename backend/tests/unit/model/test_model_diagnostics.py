"""Pruebas del diagnóstico del OU (6.2): residuos AR(1), ACF, Ljung–Box y estabilidad."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import cointegrated_log_prices
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from mrlab.domain.errors import ConfigError, DataError, ModelError
from mrlab.model.diagnostics import ar1_residuals, residual_diagnostics, rolling_ou_stability
from mrlab.model.ou import OUStatus, adf_unit_root_test, fit_ou, simulate_ou_exact


def _series(values: np.ndarray) -> pd.Series:
    idx = pd.bdate_range("2016-01-01", periods=len(values), name="date")
    return pd.Series(values, index=idx, name="spread")


def _ou_spread(n: int = 1000, seed: int = 21, lam: float = 0.1) -> pd.Series:
    return _series(simulate_ou_exact(lam, 0.0, 0.01, 0.0, n - 1, 1, seed)[0])


def _ar2_spread(n: int = 1000, seed: int = 5) -> pd.Series:
    e = np.random.default_rng(seed).standard_normal(n) * 0.01
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] + 0.3 * x[t - 2] + e[t]
    return _series(x)


def test_ar1_residuals_definition_and_index():
    spread = _ou_spread(300)
    fit = fit_ou(spread)
    resid = ar1_residuals(spread, fit)
    x = spread.to_numpy()

    assert resid.name == "residuo_ar1"
    assert resid.index.equals(spread.index[1:])  # fecha del valor X_{t+1}
    np.testing.assert_allclose(resid.to_numpy(), x[1:] - fit.a - fit.b * x[:-1])
    assert abs(resid.mean()) < 1e-12  # MCO con constante


def test_ar1_residuals_exist_for_invalid_fit_with_coefficients():
    e = np.random.default_rng(3).standard_normal(500)
    x = np.zeros(500)
    for t in range(1, 500):
        x[t] = -0.5 * x[t - 1] + e[t]
    spread = _series(x)
    fit = fit_ou(spread)
    assert fit.status == OUStatus.NON_POSITIVE_B
    assert len(ar1_residuals(spread, fit)) == 499

    insufficient = fit_ou(spread.iloc[:10])
    with pytest.raises(ModelError):
        ar1_residuals(spread.iloc[:10], insufficient)


def test_true_ou_residuals_show_no_autocorrelation():
    spread = _ou_spread(2000, seed=21)
    diag = residual_diagnostics(spread, fit_ou(spread))
    assert [lb.lag for lb in diag.ljung_box] == [10, 20]
    assert all(lb.pvalue > 0.05 for lb in diag.ljung_box)
    assert not diag.autocorrelation_detected
    assert "No se detecta autocorrelación" in diag.message


def test_ar2_spread_autocorrelation_is_detected_with_message():
    spread = _ar2_spread()
    fit = fit_ou(spread)
    assert fit.valid  # el AR(1) "funciona" en apariencia…
    diag = residual_diagnostics(spread, fit)
    assert diag.autocorrelation_detected  # …pero deja dinámica sin explicar
    assert diag.ljung_box[-1].pvalue < 0.05
    assert "no captura toda la dinámica" in diag.message


def test_acf_20_lags_band_and_ljung_box_match_statsmodels():
    spread = _ou_spread(1000)
    fit = fit_ou(spread)
    diag = residual_diagnostics(spread, fit, nlags=20)
    resid = ar1_residuals(spread, fit).to_numpy()

    assert diag.nlags == 20 and len(diag.acf) == 20
    np.testing.assert_allclose(diag.acf, acf(resid, nlags=20, fft=False)[1:])
    assert diag.conf_band == pytest.approx(1.96 / math.sqrt(resid.size))
    ref = acorr_ljungbox(resid, lags=[10, 20], model_df=1, return_df=True)
    np.testing.assert_allclose([lb.statistic for lb in diag.ljung_box], ref["lb_stat"])
    np.testing.assert_allclose([lb.pvalue for lb in diag.ljung_box], ref["lb_pvalue"])
    assert diag.residual_mean == pytest.approx(resid.mean())
    assert diag.residual_std == pytest.approx(resid.std(ddof=1))


def test_residual_diagnostics_input_checks():
    spread = _ou_spread(200)
    fit = fit_ou(spread)
    with pytest.raises(ConfigError):
        residual_diagnostics(spread, fit, lb_lags=(1, 10))
    with pytest.raises(DataError):  # 19 residuos no alcanzan para 20 rezagos
        residual_diagnostics(spread.iloc[:20], fit)


def test_rolling_ou_stability_windows_and_cv_lambda():
    spread = _ou_spread(1000)
    stab = rolling_ou_stability(spread, window=252, step=63)

    starts = list(range(0, 1000 - 252 + 1, 63))
    assert len(stab.windows) == len(starts) == 12
    for w, s in zip(stab.windows, starts, strict=True):
        assert w.start == spread.index[s].date()
        assert w.end == spread.index[s + 251].date()
        assert w.nobs == 252
        ref = fit_ou(spread.iloc[s : s + 252], require_unit_root_rejection=False)
        assert w.estimable == ref.valid
        assert w.lam == ref.lam and w.half_life == ref.half_life and w.b == ref.b

    lams = np.array([w.lam for w in stab.windows if w.estimable])
    assert stab.n_invalid == sum(not w.estimable for w in stab.windows)
    assert stab.cv_lambda == pytest.approx(lams.std(ddof=1) / lams.mean())
    assert "12 ventanas" in stab.message and "α y β fijos" in stab.message


def test_rolling_ou_stability_counts_windows_without_parameters():
    calm = simulate_ou_exact(0.1, 0.0, 0.01, 0.0, 499, 1, seed=6)[0]
    e = np.random.default_rng(6).standard_normal(500) * 0.01
    choppy = np.zeros(500)
    for t in range(1, 500):
        choppy[t] = -0.5 * choppy[t - 1] + e[t]
    stab = rolling_ou_stability(_series(np.concatenate([calm, choppy])), window=252, step=63)

    assert len(stab.windows) == 12
    tail = [w for w, s in zip(stab.windows, range(0, 749, 63), strict=True) if s >= 500]
    assert len(tail) == 4 and not any(w.estimable for w in tail)
    assert all(w.lam is None and w.half_life is None and w.b < 0 for w in tail)
    assert stab.n_invalid == sum(not w.estimable for w in stab.windows) >= 4
    assert "no se pudieron estimar" in stab.message


def test_rolling_ou_stability_with_too_few_windows():
    short = rolling_ou_stability(_ou_spread(200), window=252, step=63)
    assert short.windows == [] and short.cv_lambda is None and short.n_invalid == 0
    assert "menos que una ventana" in short.message

    single = rolling_ou_stability(_ou_spread(300), window=252, step=63)
    assert len(single.windows) == 1 and single.cv_lambda is None
    assert "menos de 2 ventanas" in single.message

    with pytest.raises(DataError):
        rolling_ou_stability(pd.Series(np.zeros(300)), window=252, step=63)


def test_window_unit_root_test_is_reported_apart_and_does_not_filter_parameters():
    """Hallazgo de la verificación: exigir el rechazo de la raíz unitaria en cada ventana (con
    N = 2) dejaba sin parámetros a la mayoría de las ventanas de un OU verdadero. Ahora los
    parámetros no dependen del nivel de la prueba; el ADF de cada ventana (N = 1) va aparte."""
    spread = _ou_spread(1000, lam=0.05)
    lenient = rolling_ou_stability(spread, window=252, step=63, unit_root_alpha=0.10)
    strict = rolling_ou_stability(spread, window=252, step=63, unit_root_alpha=0.01, maxlag=6)
    for w, s in zip(strict.windows, range(0, 1000 - 252 + 1, 63), strict=True):
        ref = adf_unit_root_test(spread.iloc[s : s + 252], alpha=0.01, maxlag=6)
        assert ref is not None and not ref.estimated_residual  # N = 1: α y β no se re-estiman
        assert (w.df_stat, w.df_pvalue, w.df_lags) == (ref.stat, ref.pvalue, ref.lags)
        assert w.unit_root_rejected == ref.rejected == (ref.pvalue <= 0.01)
    params = ("estimable", "lam", "mu", "sigma", "half_life")
    lenient_params = [[getattr(w, k) for k in params] for w in lenient.windows]
    strict_params = [[getattr(w, k) for k in params] for w in strict.windows]
    assert lenient_params == strict_params and all(w.estimable for w in strict.windows)
    assert strict.cv_lambda == lenient.cv_lambda is not None
    assert strict.n_unit_root_rejected < lenient.n_unit_root_rejected
    assert strict.n_unit_root_tested == lenient.n_unit_root_tested == 12
    assert strict.unit_root_alpha == 0.01
    assert f"en {strict.n_unit_root_rejected} de 12 ventanas" in strict.message
    assert "poca potencia" in strict.message
    assert "no prueba que la dinámica sea inestable" in strict.message

    # ``scale`` también llega a cada ventana: ruido de redondeo → ninguna ventana con parámetros.
    tiny = _series(_ou_spread(600, lam=0.3).to_numpy() * 1e-14)
    assert rolling_ou_stability(tiny, window=252, step=63).n_invalid == 0
    scaled = rolling_ou_stability(tiny, window=252, step=63, scale=1.0)
    assert scaled.windows and scaled.n_invalid == len(scaled.windows)


def _training_residual(log_a: np.ndarray, log_b: np.ndarray, n_train: int = 900) -> pd.Series:
    """Spread de formación como en el análisis: residuo de la MCO de log A sobre log B."""
    y, x = log_a[:n_train], log_b[:n_train]
    design = np.column_stack([np.ones(n_train), x])
    return _series(y - design @ np.linalg.lstsq(design, y, rcond=None)[0])


def test_true_ou_windows_have_parameters_and_unbiased_half_life_range():
    """OU verdadero (λ = 0,05, vida media 13,9): antes el 83 % de las ventanas quedaba «no
    válido», el 48 % de los pares no tenía CV(λ) y la vida media de las ventanas que pasaban
    era ≈ 6,5 días (solo pasaban las de λ alto)."""
    lam = 0.05
    true_half_life = math.log(2) / lam
    with_cv, fractions, half_lives = 0, [], []
    for seed in range(50):
        stab = rolling_ou_stability(
            _training_residual(*cointegrated_log_prices(1500, seed, lam=lam))
        )
        with_cv += stab.cv_lambda is not None
        fractions.append(1 - stab.n_invalid / len(stab.windows))
        half_lives += [w.half_life for w in stab.windows if w.estimable]
    assert with_cv >= 45  # ≥ 90 % de las semillas
    assert np.mean(fractions) >= 0.9  # la mayoría de las ventanas tiene parámetros
    # Mediana cerca del valor verdadero (el AR(1) en 252 obs. sobrestima λ un poco).
    assert 0.7 * true_half_life <= np.median(half_lives) <= 1.3 * true_half_life


@pytest.mark.slow
def test_window_unit_root_rejections_are_near_alpha_for_independent_walks():
    """Paseos independientes (H0 cierta): cada ventana es un tramo de un spread con α y β fijos,
    así que corresponde la tabla N = 1. Con N = 2 solo rechazaba el 1,6 % (5 % nominal)."""
    rejected = tested = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        log_a = np.cumsum(rng.normal(0, 0.01, 900))
        log_b = np.cumsum(rng.normal(0, 0.01, 900))
        stab = rolling_ou_stability(_training_residual(log_a, log_b))
        rejected += stab.n_unit_root_rejected
        tested += stab.n_unit_root_tested
    assert tested == 400 * 11
    assert 0.03 <= rejected / tested <= 0.08
