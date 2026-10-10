"""Estabilidad de Engle–Granger por ventanas de 252 con paso 63 (4.4, decisión E5)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from conftest import cointegrated_log_prices

from mrlab.domain.errors import DataError
from mrlab.stats.cointegration import Direction, EGStability, engle_granger, rolling_eg_stability

N = 1000
IDX = pd.date_range("2015-01-01", periods=N, freq="B", name="date")


def stability(a: np.ndarray, b: np.ndarray, **kw) -> EGStability:
    idx = IDX[: len(a)]
    return rolling_eg_stability(
        pd.Series(a, idx),
        pd.Series(b, idx),
        direction=Direction.A_ON_B,
        dependent="A",
        independent="B",
        **kw,
    )


def has_number(text: str) -> bool:
    return re.search(r"\d", text) is not None


def test_stable_cointegrated_pair() -> None:
    a, b = cointegrated_log_prices(n=N, seed=1)
    res = stability(a, b)
    assert not res.unstable
    assert not res.beta_sign_change
    assert res.frac_fail is not None and res.frac_fail <= 0.40
    assert all(w.beta > 0 for w in res.windows)
    assert len(res.reasons) == 2 and all(has_number(r) for r in res.reasons)
    assert "de 12 ventanas" in res.reasons[0] and "mismo signo" in res.reasons[1]


def test_windows_are_252_with_step_63_inside_the_sample() -> None:
    a, b = cointegrated_log_prices(n=N, seed=2)
    res = stability(a, b)
    starts = list(range(0, N - 252 + 1, 63))  # 0, 63, …, 693 → 12 ventanas
    assert (res.window, res.step) == (252, 63)
    assert len(res.windows) == len(starts) == 12
    for w, s in zip(res.windows, starts, strict=True):
        assert w.nobs == 252
        assert w.start == IDX[s].date() and w.end == IDX[s + 251].date()
    # Cada ventana es exactamente Engle–Granger sobre su tramo, en la dirección fija.
    w = res.windows[3]
    s = starts[3]
    eg = engle_granger(
        pd.Series(a[s : s + 252]),
        pd.Series(b[s : s + 252]),
        direction=Direction.A_ON_B,
        dependent="A",
        independent="B",
    )
    assert (w.pvalue, w.statistic, w.beta, w.alpha) == (
        eg.pvalue,
        eg.statistic,
        eg.beta_ols,
        eg.alpha_ols,
    )
    assert w.passed == (w.pvalue <= 0.05)


def test_pair_cointegrated_only_in_first_part_is_unstable() -> None:
    a, b = cointegrated_log_prices(n=N, seed=2)
    rng = np.random.default_rng(502)
    broken = a.copy()
    broken[400:] = a[399] + np.cumsum(rng.normal(0.0, 0.01, N - 400))  # A se desacopla de B
    res = stability(broken, b)

    assert res.unstable
    assert not res.beta_sign_change  # aquí la causa es solo la fracción de ventanas fallidas
    assert res.frac_fail is not None and res.frac_fail > 0.40
    n_fail = sum(not w.passed for w in res.windows)
    reason = res.reasons[0]
    assert f"En {n_fail} de 12 ventanas" in reason
    assert f"({100 * res.frac_fail:.0f} %)" in reason
    assert "nivel 0,05" in reason and "40 %" in reason
    assert "poca potencia" in reason


def test_beta_sign_change_is_unstable() -> None:
    a, b = cointegrated_log_prices(n=N, seed=1)
    spread = a - b
    flipped = a.copy()
    flipped[500:] = 2 * b[500] - b[500:] + spread[500:]  # β pasa de +1 a −1 sin salto de nivel
    res = stability(flipped, b)

    assert res.beta_sign_change and res.unstable
    assert res.frac_fail is not None and res.frac_fail <= 0.40  # la causa es el signo de β
    betas = [w.beta for w in res.windows]
    assert min(betas) < -0.5 and max(betas) > 0.5
    reason = res.reasons[1]
    assert "cambió de signo" in reason and has_number(reason)
    assert f"{min(betas):.3f}".replace(".", ",") in reason


def test_too_short_sample_has_no_windows() -> None:
    res = stability(*cointegrated_log_prices(n=200, seed=0))
    assert res.windows == [] and res.frac_fail is None
    assert not res.unstable and not res.beta_sign_change
    assert (
        len(res.reasons) == 1 and "200 observaciones" in res.reasons[0] and "252" in res.reasons[0]
    )


def test_custom_window_step_and_threshold() -> None:
    a, b = cointegrated_log_prices(n=N, seed=1)
    res = stability(a, b, window=500, step=250, alpha_level=0.01, max_fail_frac=0.0)
    assert [w.nobs for w in res.windows] == [500, 500, 500]
    assert (res.alpha_level, res.max_fail_frac) == (0.01, 0.0)
    assert res.unstable == (res.frac_fail > 0.0)


def test_requires_aligned_dated_series() -> None:
    a, b = cointegrated_log_prices(n=300, seed=0)
    with pytest.raises(DataError, match="fecha"):
        rolling_eg_stability(
            pd.Series(a), pd.Series(b), direction=Direction.A_ON_B, dependent="A", independent="B"
        )
    with pytest.raises(DataError, match="alineadas"):
        rolling_eg_stability(
            pd.Series(a, IDX[:300]),
            pd.Series(b, IDX[1:301]),
            direction=Direction.A_ON_B,
            dependent="A",
            independent="B",
        )


def test_window_with_a_constant_series_is_not_evaluable_and_counts_as_failed() -> None:
    """Un activo sin cotizar (precio plano) durante más de una ventana no interrumpe el
    diagnóstico: esa ventana queda «no evaluable», cuenta como fallida y se nombran sus fechas."""
    a, b = cointegrated_log_prices(n=N, seed=1)
    a, b = a.copy(), b.copy()
    a[300:600], b[300:600] = a[299], b[299]
    res = stability(a, b)

    flat = [w for w in res.windows if not w.evaluable]
    assert len(flat) == 1 and flat[0].start == IDX[315].date()
    w = flat[0]
    assert not w.passed and w.pvalue is None and w.statistic is None and w.beta is None
    assert "constante" in w.reason
    n_fail = sum(not x.passed for x in res.windows)
    assert res.frac_fail == pytest.approx(n_fail / len(res.windows))
    reason = res.reasons[0]
    assert f"En {n_fail} de {len(res.windows)} ventanas" in reason
    assert "1 no se pudo evaluar y se cuenta como fallida" in reason
    assert f"del {w.start.isoformat()} al {w.end.isoformat()}" in reason
    assert "mismo signo" in res.reasons[1]  # β solo de las ventanas evaluables


def test_collinear_window_is_not_evaluable() -> None:
    _, b = cointegrated_log_prices(n=300, seed=0)
    res = stability(1.0 + 2.0 * b, b)
    assert len(res.windows) == 1
    w = res.windows[0]
    assert not w.evaluable and not w.passed and w.pvalue is None
    assert w.beta == pytest.approx(2.0)  # β sí está definido
    assert "colineales" in w.reason
    assert res.frac_fail == 1.0 and res.unstable


def test_no_evaluable_window_has_no_beta_comparison() -> None:
    res = stability(np.full(300, 1.0), np.linspace(0.0, 1.0, 300))
    assert len(res.windows) == 1 and not res.windows[0].evaluable
    assert not res.beta_sign_change and res.unstable
    assert "Ninguna ventana" in res.reasons[1]
