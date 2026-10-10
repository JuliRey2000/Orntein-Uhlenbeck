"""Consistencia α–μ (secciones 5 y 14): el nivel del spread se resta UNA sola vez.

α queda dentro del spread (MCO con constante), así que el residuo de formación tiene
media ≈ 0; el μ del OU se estima sobre ese residuo y el z-score lo usa tal cual:
z = (X − μ̂)/σ_eq. Estas pruebas fallan si alguien no resta α, lo resta dos veces o
vuelve a centrar el spread o el z con su propia media.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import cointegrated_log_prices

from mrlab.domain.models import SpreadType
from mrlab.model.ou import fit_ou, zscore
from mrlab.model.spread import compute_spread, fit_hedge_ratio

N, N_TRAIN, SEED = 1000, 600, 11


def _prices(alpha: float) -> tuple[pd.Series, pd.Series]:
    log_a, log_b = cointegrated_log_prices(n=N, seed=SEED, beta=1.0, alpha=alpha)
    idx = pd.bdate_range("2018-01-01", periods=N, name="date")
    return pd.Series(np.exp(log_a), index=idx, name="AAA"), pd.Series(
        np.exp(log_b), index=idx, name="BBB"
    )


def _pipeline(alpha: float):
    y, x = _prices(alpha)
    hedge = fit_hedge_ratio(y.iloc[:N_TRAIN], x.iloc[:N_TRAIN], SpreadType.LOG, "AAA", "BBB")
    spread = compute_spread(y, x, hedge)  # toda la muestra, α y β fijos de formación
    ou = fit_ou(spread.iloc[:N_TRAIN])
    return y, x, hedge, spread, ou


def test_training_residual_mean_zero_and_ou_mu_close_to_zero():
    y, x, hedge, spread, ou = _pipeline(alpha=0.3)
    assert hedge.alpha == pytest.approx(0.3, abs=0.15)  # α claramente distinto de 0
    train = spread.iloc[:N_TRAIN]

    assert abs(train.mean()) < 1e-10
    assert ou.valid
    assert abs(ou.mu) <= 3 * ou.mu_se
    assert abs(ou.mu) < 0.05 * ou.sigma_eq

    # Control: si se olvida α, el nivel del "spread" (y su μ̂) es α, no 0.
    without_alpha = np.log(y.iloc[:N_TRAIN]) - hedge.beta * np.log(x.iloc[:N_TRAIN])
    assert fit_ou(without_alpha).mu == pytest.approx(hedge.alpha, abs=0.05 * ou.sigma_eq)


def test_zscore_uses_ou_mu_as_is():
    _, _, _, spread, ou = _pipeline(alpha=0.3)
    z = zscore(spread, ou)

    expected = (spread - ou.mu) / ou.sigma_eq
    np.testing.assert_allclose(z.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)
    # En formación z está centrado (μ̂ ≈ 0 y media del spread = 0). Restar α otra vez
    # desplazaría z en −α/σ_eq ≈ −13.
    assert abs(z.iloc[:N_TRAIN].mean()) < 0.1
    # z no se recentra con la media de la serie que recibe: desplazar el spread desplaza z.
    shift = 0.05
    np.testing.assert_allclose(
        (zscore(spread + shift, ou) - z).to_numpy(), shift / ou.sigma_eq, atol=1e-9
    )


def test_spread_ou_and_z_do_not_depend_on_alpha():
    _, _, h0, s0, o0 = _pipeline(alpha=0.0)
    _, _, h1, s1, o1 = _pipeline(alpha=0.3)

    assert h1.alpha - h0.alpha == pytest.approx(0.3, abs=1e-9)
    assert h1.beta == pytest.approx(h0.beta, abs=1e-9)
    np.testing.assert_allclose(s1.to_numpy(), s0.to_numpy(), rtol=0, atol=1e-9)
    assert o1.mu == pytest.approx(o0.mu, abs=1e-9)
    assert o1.sigma_eq == pytest.approx(o0.sigma_eq, rel=1e-9)
    np.testing.assert_allclose(
        zscore(s1, o1).to_numpy(), zscore(s0, o0).to_numpy(), rtol=0, atol=1e-6
    )
