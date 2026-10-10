"""Pruebas del OU (sección 6 y 14): recuperación, b fuera de rango, fórmulas, z y simulación."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp

from mrlab.domain.errors import ConfigError, ModelError
from mrlab.domain.models import SpreadType
from mrlab.model.ou import (
    MAX_SIM_PATHS,
    MIN_OBS_OU,
    NOT_OU_MESSAGE,
    OUFit,
    OUStatus,
    conditional_mean,
    conditional_variance,
    dickey_fuller_critical_value,
    fit_ou,
    forecast,
    rolling_zscore,
    simulate_ou_exact,
    simulate_paths_summary,
    zscore,
)
from mrlab.model.spread import compute_spread, fit_hedge_ratio

OU_PARAMS = ("lam", "mu", "sigma", "sigma_eq", "half_life", "mu_se")


def _ar1(coef: float, n: int, seed: int, c: float = 0.0) -> np.ndarray:
    """X_t = c + coef·X_{t−1} + e_t con e ~ N(0, 1)."""
    e = np.random.default_rng(seed).standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = c + coef * x[t - 1] + e[t]
    return x


def _valid_fit(lam: float = 0.1, mu: float = 0.5, sigma: float = 0.2) -> OUFit:
    """Ajuste válido con parámetros conocidos (para probar fórmulas sin ruido de estimación)."""
    return OUFit(
        status=OUStatus.VALID,
        valid=True,
        lam=lam,
        mu=mu,
        sigma=sigma,
        sigma_eq=sigma / math.sqrt(2 * lam),
        half_life=math.log(2) / lam,
    )


# --------------------------------------------------------------------------- recuperación


def test_recovers_parameters_seed_42():
    lam, mu, sigma = 0.05, 0.0, 0.02
    x = simulate_ou_exact(lam=lam, mu=mu, sigma=sigma, x0=0.0, n_steps=5000, n_paths=1, seed=42)[0]
    fit = fit_ou(x)

    assert fit.status == OUStatus.VALID and fit.valid
    assert abs(fit.lam - lam) / lam < 0.15
    assert abs(fit.sigma - sigma) / sigma < 0.10
    # Decisión O2: con μ = 0 el error relativo no está definido → |μ̂ − μ| ≤ 3·EE(μ̂).
    assert abs(fit.mu - mu) <= 3 * fit.mu_se
    sigma_eq = sigma / math.sqrt(2 * lam)
    assert fit.mu_se == pytest.approx(0.089 * sigma_eq, rel=0.15)
    assert fit.nobs == 5000
    assert fit.warnings == [] and not fit.near_unit_root
    # Decisión O7: la validez exige que Dickey–Fuller rechace la raíz unitaria (t ≈ −11).
    assert fit.df_stat < -10 and fit.df_stat <= fit.df_crit
    assert "rechaza la raíz unitaria" in fit.message


def test_mu_interval_coverage_and_lambda_bias_over_many_seeds():
    lam, mu, sigma = 0.05, 0.0, 0.02
    inside, lams = 0, []
    for seed in range(200):
        x = simulate_ou_exact(lam, mu, sigma, 0.0, 5000, 1, seed)[0]
        fit = fit_ou(x)
        assert fit.valid
        inside += abs(fit.mu - mu) <= 1.96 * fit.mu_se
        lams.append(fit.lam)
    coverage = inside / 200
    assert 0.90 <= coverage <= 0.99
    assert abs(np.mean(lams) / lam - 1) < 0.05


def test_estimates_follow_exact_discretization_formulas():
    x = simulate_ou_exact(0.1, 0.3, 0.05, 0.3, 2000, 1, seed=3)[0]
    fit = fit_ou(x)
    ols = OLS(x[1:], np.column_stack([np.ones(x.size - 1), x[:-1]])).fit()
    a, b = ols.params
    n = x.size - 1

    assert fit.a == pytest.approx(a, rel=1e-9)
    assert fit.b == pytest.approx(b, rel=1e-9)
    assert fit.a_se == pytest.approx(ols.bse[0], rel=1e-9)
    assert fit.b_se == pytest.approx(ols.bse[1], rel=1e-9)
    assert fit.s2 == pytest.approx(ols.ssr / (n - 2), rel=1e-9)
    assert fit.lam == pytest.approx(-math.log(b), rel=1e-12)
    assert fit.mu == pytest.approx(a / (1 - b), rel=1e-9)
    assert fit.sigma == pytest.approx(math.sqrt(2 * fit.lam * fit.s2 / (1 - b * b)), rel=1e-12)
    assert fit.sigma_eq == pytest.approx(fit.sigma / math.sqrt(2 * fit.lam), rel=1e-12)
    assert fit.half_life == pytest.approx(math.log(2) / fit.lam, rel=1e-12)
    assert fit.mu_se == pytest.approx(fit.sigma_eq * math.sqrt((1 + b) / ((1 - b) * n)), rel=1e-12)
    assert fit.df_stat == pytest.approx((b - 1) / ols.bse[1], rel=1e-9)


def test_dt_scales_lambda_and_half_life():
    x = simulate_ou_exact(0.1, 0.0, 0.05, 0.0, 1000, 1, seed=8)[0]
    daily, weekly = fit_ou(x), fit_ou(x, dt=5.0, time_unit="semana")
    assert weekly.b == daily.b
    assert weekly.lam == pytest.approx(daily.lam / 5)
    assert weekly.half_life == pytest.approx(daily.half_life * 5)
    assert weekly.sigma_eq == pytest.approx(daily.sigma_eq)
    assert weekly.time_unit == "semana"


# --------------------------------------------------------------------------- b fuera de rango


def _assert_no_ou_numbers(fit: OUFit) -> None:
    assert not fit.valid
    for name in OU_PARAMS:
        assert getattr(fit, name) is None, name


def _walk(seed: int, n: int = 1000) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _ols_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residuo de la MCO y = α + β·x (como el spread de formación del pipeline)."""
    design = np.column_stack([np.ones(x.size), x])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def test_random_walk_is_non_stationary_without_half_life():
    # Semilla 42: b̂ < 1 (sesgo de Dickey–Fuller). Con la regla anterior (solo 0 < b < 1)
    # este paseo aleatorio salía «válido», con vida media; la decisión O7 exige además que
    # Dickey–Fuller rechace la raíz unitaria.
    fit = fit_ou(_walk(42))
    assert 0 < fit.b < 1

    assert fit.status == OUStatus.NON_STATIONARY
    _assert_no_ou_numbers(fit)
    assert fit.near_unit_root
    assert NOT_OU_MESSAGE.lower() in fit.message.lower()
    assert "no se puede distinguir el spread de un paseo aleatorio" in fit.message
    # La regresión y la prueba sí se informan: b, el estadístico y el valor crítico.
    assert fit.a is not None and fit.nobs == 999
    assert fit.df_stat > fit.df_crit
    assert fit.df_crit == pytest.approx(mackinnoncrit(N=1, regression="c", nobs=999)[1])
    for number in (fit.b, fit.df_stat, fit.df_crit):
        decimals = 4 if number == fit.b else 2
        assert f"{number:.{decimals}f}".replace(".", ",") in fit.message
    assert fit.warnings == []


def test_random_walk_with_b_at_least_one_is_non_stationary():
    fit = fit_ou(_walk(0))
    assert fit.b >= 1
    assert fit.status == OUStatus.NON_STATIONARY
    _assert_no_ou_numbers(fit)
    assert not fit.near_unit_root  # la causa es b ≥ 1, no la prueba de raíz unitaria
    assert "≥ 1" in fit.message


def test_random_walks_are_non_stationary_in_90_percent_of_seeds():
    """Sección 14 («random walk → no estacionario y sin vida media»), robusta a la semilla."""
    non_stationary = 0
    for seed in range(100):
        fit = fit_ou(_walk(seed))
        if fit.status == OUStatus.NON_STATIONARY:
            non_stationary += 1
            _assert_no_ou_numbers(fit)
        if fit.near_unit_root:  # «b no se distingue de 1» implica ajuste no válido
            assert not fit.valid and fit.status == OUStatus.NON_STATIONARY
        assert fit.valid == (0 < fit.b < 1 and fit.df_stat <= fit.df_crit)
    assert non_stationary >= 90


def test_unit_root_test_uses_residual_critical_values_for_estimated_spreads():
    """Con el residuo de una MCO estimada hay que usar los valores críticos N = 2 (Engle–Granger):
    con N = 1, dos paseos aleatorios independientes parecen «revertir» con demasiada frecuencia."""
    valid_n1 = valid_n2 = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        y, x = np.cumsum(rng.standard_normal(1000)), np.cumsum(rng.standard_normal(1000))
        spread = _ols_residual(y, x)
        as_observed = fit_ou(spread)
        as_residual = fit_ou(spread, estimated_residual=True)
        valid_n1 += as_observed.valid
        valid_n2 += as_residual.valid
        assert as_residual.df_estimated_residual and not as_observed.df_estimated_residual
        assert as_residual.df_stat == as_observed.df_stat
        assert as_residual.df_crit < as_observed.df_crit  # N = 2 es más exigente
    assert valid_n2 / 200 <= 0.10  # tamaño nominal del 5 %
    assert valid_n1 / 200 >= 0.15  # el valor crítico de una serie observada no sirve aquí


def test_seed_where_only_residual_critical_value_detects_the_random_walk():
    rng = np.random.default_rng(1007)
    y, x = np.cumsum(rng.normal(0, 0.01, 1500)), np.cumsum(rng.normal(0, 0.01, 1500))
    spread = _ols_residual(y[:900], x[:900])
    observed, residual = fit_ou(spread), fit_ou(spread, estimated_residual=True)
    assert residual.df_crit < residual.df_stat < observed.df_crit
    assert observed.valid
    assert residual.status == OUStatus.NON_STATIONARY and residual.near_unit_root
    _assert_no_ou_numbers(residual)
    assert "Engle–Granger" in residual.message and "residuo" in residual.message


def test_dickey_fuller_critical_values():
    for estimated, n_vars in ((False, 1), (True, 2)):
        table = mackinnoncrit(N=n_vars, regression="c", nobs=899)
        for level, crit in zip((0.01, 0.05, 0.10), table, strict=True):
            assert dickey_fuller_critical_value(level, 899, estimated_residual=estimated) == crit
        # Otros niveles: valor asintótico coherente con los p-values de MacKinnon.
        c25 = dickey_fuller_critical_value(0.025, 899, estimated_residual=estimated)
        assert table[0] < c25 < table[1]
        assert mackinnonp(c25, regression="c", N=n_vars) == pytest.approx(0.025, abs=1e-6)
    for bad in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ConfigError):
            dickey_fuller_critical_value(bad, 899)
        with pytest.raises(ConfigError):
            fit_ou(_walk(1), unit_root_alpha=bad)


def test_unit_root_alpha_moves_the_decision():
    # AR(1) con b cercano a 1: rechaza al 10 % pero no al 1 %.
    x = _ar1(0.985, 1000, seed=5)
    loose, strict = fit_ou(x, unit_root_alpha=0.10), fit_ou(x, unit_root_alpha=0.01)
    assert loose.df_stat == strict.df_stat
    assert strict.df_crit < loose.df_stat <= loose.df_crit
    assert loose.valid and loose.df_alpha == 0.10
    assert strict.status == OUStatus.NON_STATIONARY and strict.near_unit_root
    assert "al 1 %" in strict.message


def test_scale_detects_rounding_noise_of_identical_legs():
    """Misma serie en las dos piernas: el spread es ruido de coma flotante (~1e-16)."""
    log_p = pd.Series(
        np.log(50.0) + np.cumsum(np.random.default_rng(5).normal(0, 0.012, 900)),
        index=pd.bdate_range("2015-01-01", periods=900, name="date"),
    )
    prices = np.exp(log_p)
    hedge = fit_hedge_ratio(prices, prices, SpreadType.LOG, "KO", "KO")
    spread = compute_spread(prices, prices, hedge)
    assert float(spread.std()) < 1e-12

    unscaled = fit_ou(spread, estimated_residual=True)
    assert unscaled.valid  # sin escala absoluta no se distingue del ruido (el defecto original)
    scale = float(np.std(np.log(prices)))
    fit = fit_ou(spread, estimated_residual=True, scale=scale)
    assert fit.status == OUStatus.INSUFFICIENT
    _assert_no_ou_numbers(fit)
    assert "ruido de redondeo" in fit.message

    # Un spread real no se ve afectado por la escala.
    ou = simulate_ou_exact(0.1, 0.0, 0.01, 0.0, 999, 1, seed=3)[0]
    assert fit_ou(ou, scale=scale) == fit_ou(ou)
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ConfigError):
            fit_ou(ou, scale=bad)


def test_explosive_series_is_non_stationary():
    fit = fit_ou(_ar1(1.01, 500, seed=4))
    assert fit.status == OUStatus.NON_STATIONARY
    _assert_no_ou_numbers(fit)


def test_negative_b_is_non_positive_b():
    fit = fit_ou(_ar1(-0.5, 1000, seed=3))
    assert fit.b < 0
    assert fit.status == OUStatus.NON_POSITIVE_B
    _assert_no_ou_numbers(fit)
    assert NOT_OU_MESSAGE.lower() in fit.message.lower()


@pytest.mark.parametrize(
    "series",
    [np.cumsum(np.random.default_rng(0).standard_normal(1000)), _ar1(-0.5, 1000, seed=3)],
    ids=["no_estacionario", "b_no_positivo"],
)
def test_invalid_fit_blocks_forecast_and_zscore(series):
    fit = fit_ou(series)
    assert not fit.valid
    with pytest.raises(ModelError):
        forecast(fit, 0.0)
    with pytest.raises(ModelError):
        zscore(pd.Series(series), fit)
    with pytest.raises(ModelError):
        conditional_mean(0.0, fit, 5)
    with pytest.raises(ModelError):
        conditional_variance(fit, 5)


def test_insufficient_and_degenerate_series():
    x = simulate_ou_exact(0.1, 0.0, 0.05, 0.0, MIN_OBS_OU, 1, seed=1)[0]  # MIN_OBS_OU + 1 puntos
    assert fit_ou(x[: MIN_OBS_OU - 1]).status == OUStatus.INSUFFICIENT
    assert fit_ou(x[:MIN_OBS_OU]).status != OUStatus.INSUFFICIENT

    for degenerate in (np.full(100, 0.3), 0.9 ** np.arange(100)):
        fit = fit_ou(degenerate)
        assert fit.status == OUStatus.INSUFFICIENT
        _assert_no_ou_numbers(fit)
        assert fit.message


def test_nan_breaks_transitions_instead_of_bridging_them():
    x = simulate_ou_exact(0.1, 0.0, 0.05, 0.0, 499, 1, seed=2)[0]
    x[[100, 250]] = np.nan
    fit = fit_ou(pd.Series(x))
    assert fit.nobs == 499 - 4  # cada NaN elimina dos transiciones
    clean = np.concatenate([x[:100], x[101:250], x[251:]])
    assert fit_ou(clean).nobs == 499 - 2  # (concatenar sí crearía transiciones ficticias)


# --------------------------------------------------------------------------- advertencias


def test_half_life_above_126_warns_too_slow():
    x = simulate_ou_exact(math.log(2) / 300, 0.0, 0.01, 0.0, 20000, 1, seed=5)[0]
    fit = fit_ou(x)
    assert fit.valid and fit.half_life > 126
    assert any("demasiado lenta para operar con este horizonte" in w for w in fit.warnings)
    assert any("> 126" in w for w in fit.warnings)
    # El umbral es configurable.
    assert not any("demasiado lenta" in w for w in fit_ou(x, half_life_max=1000).warnings)


def test_half_life_below_one_warns_not_tradable():
    x = simulate_ou_exact(1.0, 0.0, 0.01, 0.0, 2000, 1, seed=5)[0]
    fit = fit_ou(x)
    assert fit.valid and fit.half_life < 1
    assert any("con datos diarios no es operable ni fiable" in w for w in fit.warnings)
    assert not any("demasiado lenta" in w for w in fit.warnings)


# --------------------------------------------------------------------------- fórmulas


def test_conditional_mean_and_variance_closed_form():
    fit = _valid_fit(lam=0.1, mu=0.5, sigma=0.2)
    x_t, h = 1.0, 5.0
    assert conditional_mean(x_t, fit, h) == pytest.approx(0.5 + 0.5 * math.exp(-0.5), rel=1e-12)
    assert conditional_variance(fit, h) == pytest.approx(
        fit.sigma_eq**2 * (1 - math.exp(-1.0)), rel=1e-12
    )
    assert conditional_mean(x_t, fit, 0) == x_t
    assert conditional_variance(fit, 0) == 0.0
    # h → ∞: la media tiende a μ y la varianza a σ_eq².
    assert conditional_mean(x_t, fit, 1e4) == pytest.approx(fit.mu, abs=1e-12)
    assert conditional_variance(fit, 1e4) == pytest.approx(fit.sigma_eq**2, rel=1e-12)
    with pytest.raises(ConfigError):
        conditional_mean(x_t, fit, -1)


def test_conditional_moments_match_exact_simulation():
    lam, mu, sigma, x0, h = 0.1, 0.5, 0.2, 1.0, 5
    fit = _valid_fit(lam, mu, sigma)
    end = simulate_ou_exact(lam, mu, sigma, x0, h, n_paths=20000, seed=9)[:, -1]
    assert end.mean() == pytest.approx(
        conditional_mean(x0, fit, h), abs=4 * end.std() / math.sqrt(end.size)
    )
    assert end.var() == pytest.approx(conditional_variance(fit, h), rel=0.04)


@pytest.mark.parametrize("level", [0.95, 0.68, 0.99])
def test_forecast_intervals_at_requested_level(level):
    lam, mu, sigma, x0 = 0.1, 0.5, 0.2, 1.0
    fit = _valid_fit(lam, mu, sigma)
    result = forecast(fit, x0, horizons=(1, 5, 20), level=level)
    q = norm.ppf(0.5 + level / 2)

    assert [f.horizon for f in result] == [1, 5, 20]
    for f in result:
        assert f.level == level
        assert f.mean == pytest.approx(conditional_mean(x0, fit, f.horizon))
        assert f.std == pytest.approx(math.sqrt(conditional_variance(fit, f.horizon)))
        assert f.upper - f.mean == pytest.approx(q * f.std)
        assert f.mean - f.lower == pytest.approx(q * f.std)
    # Cobertura empírica a h = 5 con trayectorias exactas.
    end = simulate_ou_exact(lam, mu, sigma, x0, 5, n_paths=20000, seed=10)[:, -1]
    f5 = result[1]
    assert np.mean((end >= f5.lower) & (end <= f5.upper)) == pytest.approx(level, abs=0.01)


def test_forecast_rejects_bad_level_or_horizon():
    fit = _valid_fit()
    with pytest.raises(ConfigError):
        forecast(fit, 0.0, level=1.5)
    with pytest.raises(ConfigError):
        forecast(fit, 0.0, horizons=(0, 5))


# --------------------------------------------------------------------------- z-score


def test_zscore_uses_fixed_training_parameters():
    idx = pd.bdate_range("2020-01-01", periods=5, name="date")
    spread = pd.Series([0.5, 1.0, 0.0, 0.5 + 2 * 0.4472135955, -1.0], index=idx)
    fit = _valid_fit(lam=0.1, mu=0.5, sigma=0.2)  # σ_eq = 0,2/√0,2 ≈ 0,4472
    z = zscore(spread, fit)
    assert z.name == "zscore"
    assert z.index.equals(idx)
    np.testing.assert_allclose(z.to_numpy(), ((spread - 0.5) / fit.sigma_eq).to_numpy())
    assert z.iloc[3] == pytest.approx(2.0)


def test_rolling_zscore_has_no_lookahead():
    rng = np.random.default_rng(12)
    idx = pd.bdate_range("2020-01-01", periods=300, name="date")
    spread = pd.Series(rng.standard_normal(300).cumsum() * 0.01, index=idx)
    window = 63
    z = rolling_zscore(spread, window)

    assert z.name == "zscore_movil"
    assert z.iloc[: window - 1].isna().all()
    assert z.iloc[window - 1 :].notna().all()
    t = 150
    last = spread.iloc[t - window + 1 : t + 1]
    assert z.iloc[t] == pytest.approx((spread.iloc[t] - last.mean()) / last.std(ddof=1))

    altered = spread.copy()
    altered.iloc[t + 1 :] = altered.iloc[t + 1 :] * 50 + 3
    z_altered = rolling_zscore(altered, window)
    pd.testing.assert_series_equal(z.iloc[: t + 1], z_altered.iloc[: t + 1])
    assert not np.allclose(z.iloc[t + 1 :], z_altered.iloc[t + 1 :])

    with pytest.raises(ConfigError):
        rolling_zscore(spread, 1)


# --------------------------------------------------------------------------- simulación


def test_simulation_shapes_label_and_bands():
    lam, mu, sigma, x0, horizon, n_paths = 0.05, 0.1, 0.02, 0.3, 60, 40
    sim = simulate_paths_summary(lam, mu, sigma, x0, horizon, n_paths, seed=1)

    assert sim.label == "SINTÉTICO"
    assert len(sim.paths) == n_paths and all(len(p) == horizon + 1 for p in sim.paths)
    assert all(p[0] == x0 for p in sim.paths)
    for series in (
        sim.time_grid,
        sim.mean_empirical,
        sim.mean_theoretical,
        sim.band_lower_theoretical,
        sim.band_upper_theoretical,
        sim.stationary_lower,
        sim.stationary_upper,
    ):
        assert len(series) == horizon + 1
    assert len(sim.crossing_times) == n_paths
    np.testing.assert_array_equal(
        np.array(sim.paths), simulate_ou_exact(lam, mu, sigma, x0, horizon, n_paths, 1)
    )

    t = np.arange(horizon + 1)
    sigma_eq = sigma / math.sqrt(2 * lam)
    mean_th = mu + (x0 - mu) * np.exp(-lam * t)
    sd = sigma_eq * np.sqrt(1 - np.exp(-2 * lam * t))
    assert sim.sigma_eq == pytest.approx(sigma_eq)
    assert sim.half_life_theoretical == pytest.approx(math.log(2) / lam)
    np.testing.assert_allclose(sim.mean_theoretical, mean_th)
    np.testing.assert_allclose(sim.band_upper_theoretical, mean_th + 2 * sd)
    np.testing.assert_allclose(sim.band_lower_theoretical, mean_th - 2 * sd)
    np.testing.assert_allclose(sim.stationary_upper, mu + 2 * sigma_eq)
    np.testing.assert_allclose(sim.stationary_lower, mu - 2 * sigma_eq)
    assert any("SINTÉTICO" in n for n in sim.notes)
    assert any("sesgados hacia arriba" in n for n in sim.notes)


def test_simulation_limits_and_invalid_inputs():
    simulate_paths_summary(0.05, 0.0, 0.02, 0.1, 5, MAX_SIM_PATHS, seed=1)  # el máximo se permite
    with pytest.raises(ConfigError) as exc:
        simulate_paths_summary(0.05, 0.0, 0.02, 0.1, 5, MAX_SIM_PATHS + 1, seed=1)
    assert exc.value.code == "DEMASIADAS_TRAYECTORIAS"
    with pytest.raises(ConfigError) as exc:
        simulate_paths_summary(0.05, 0.2, 0.02, 0.2, 50, 10, seed=1)
    assert exc.value.code == "X0_IGUAL_A_MU"
    for bad in (
        {"lam": 0.0},
        {"sigma": -1.0},
        {"horizon": 0},
        {"n_paths": 0},
        {"lam": float("nan")},
    ):
        kwargs = {
            "lam": 0.05,
            "mu": 0.0,
            "sigma": 0.02,
            "x0": 0.1,
            "horizon": 10,
            "n_paths": 5,
            "seed": 1,
        }
        with pytest.raises(ConfigError):
            simulate_paths_summary(**{**kwargs, **bad})


def test_simulation_is_deterministic_given_seed():
    a = simulate_paths_summary(0.05, 0.0, 0.02, 0.2, 50, 20, seed=7)
    b = simulate_paths_summary(0.05, 0.0, 0.02, 0.2, 50, 20, seed=7)
    c = simulate_paths_summary(0.05, 0.0, 0.02, 0.2, 50, 20, seed=8)
    assert a == b
    assert a.paths != c.paths


def test_empirical_half_life_close_to_theoretical_with_500_paths():
    lam, mu, sigma = 0.05, 0.0, 0.02
    x0 = 3 * sigma / math.sqrt(2 * lam)  # desviación inicial de 3·σ_eq
    sim = simulate_paths_summary(lam, mu, sigma, x0, horizon=120, n_paths=500, seed=2026)
    assert sim.half_life_empirical is not None
    assert sim.half_life_empirical == pytest.approx(sim.half_life_theoretical, rel=0.10)


@pytest.mark.parametrize("x0", [0.25, -0.25])
def test_crossing_times_follow_definition(x0):
    mu = 0.05
    sim = simulate_paths_summary(0.05, mu, 0.02, x0, horizon=80, n_paths=200, seed=3)
    side = math.copysign(1.0, x0 - mu)
    for path, crossing in zip(sim.paths, sim.crossing_times, strict=True):
        dev = (np.array(path[1:]) - mu) * side  # > 0: sigue del lado inicial
        if crossing is None:
            assert (dev > 0).all()
        else:
            k = int(crossing)
            assert crossing == k and k >= 1
            assert dev[k - 1] <= 0 and (dev[: k - 1] > 0).all()
    assert sim.n_not_crossed == sum(c is None for c in sim.crossing_times)
    crossed = [c if c is not None else math.inf for c in sim.crossing_times]
    expected_median = float(np.median(crossed))
    assert sim.crossing_time_median == (expected_median if math.isfinite(expected_median) else None)


def test_no_crossing_within_short_horizon_is_reported():
    sim = simulate_paths_summary(0.01, 0.0, 0.001, 1.0, horizon=10, n_paths=30, seed=4)
    assert sim.crossing_times == [None] * 30
    assert sim.n_not_crossed == 30
    assert sim.half_life_empirical is None
    assert sim.crossing_time_median is None
    assert any("no cruzaron" in n for n in sim.notes)
    assert any("amplía el horizonte" in n for n in sim.notes)
