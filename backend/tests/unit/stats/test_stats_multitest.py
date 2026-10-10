"""Benjamini–Hochberg / Benjamini–Yekutieli frente a statsmodels (sección 14: FDR)."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from mrlab.stats.multitest import adjust_pvalues, benjamini_hochberg, benjamini_yekutieli

OURS = {"fdr_bh": benjamini_hochberg, "fdr_by": benjamini_yekutieli}


def _assert_matches_statsmodels(p: np.ndarray, method: str, q: float) -> None:
    reject, adjusted = OURS[method](p, q)
    sm_reject, sm_adjusted, _, _ = multipletests(p, alpha=q, method=method)
    np.testing.assert_array_equal(reject, sm_reject)
    np.testing.assert_allclose(adjusted, sm_adjusted, rtol=1e-12, atol=0)
    assert reject.dtype == bool
    assert adjusted.shape == p.shape


@pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
@pytest.mark.parametrize("q", [0.05, 0.10])
@pytest.mark.parametrize("m", [1, 2, 3, 10, 57, 200, 870])
def test_matches_statsmodels_random(method: str, q: float, m: int) -> None:
    rng = np.random.default_rng(1000 + m)
    # Mezcla de "señales" (p pequeños) y nulos uniformes para que haya rechazos y no rechazos.
    k = m // 4
    p = np.concatenate([rng.uniform(0, 0.005, k), rng.uniform(0, 1, m - k)])
    rng.shuffle(p)
    _assert_matches_statsmodels(p, method, q)


@pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
@pytest.mark.parametrize("seed", range(5))
def test_matches_statsmodels_with_ties_zeros_and_ones(method: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    support = np.array([0.0, 0.0001, 0.001, 0.01, 0.02, 0.04, 0.2, 0.7, 1.0])
    p = rng.choice(support, size=60)
    p[:3] = [0.0, 1.0, 0.0]  # siempre hay p = 0 y p = 1, con empates
    _assert_matches_statsmodels(p, method, 0.05)


@pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
def test_matches_statsmodels_edge_arrays(method: str) -> None:
    for p in ([0.0], [1.0], [0.0, 0.0, 0.0], [1.0, 1.0], [0.05, 0.05, 0.05, 0.05], [0.0, 1.0]):
        _assert_matches_statsmodels(np.asarray(p), method, 0.05)


def test_bh_hand_example_keeps_original_order() -> None:
    # Ordenados: 0,01 0,03 0,04 0,5 → m·p/k = 0,04 0,06 0,0533 0,5 → mín. hacia atrás.
    reject, adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.5], q=0.05)
    np.testing.assert_allclose(adjusted, [0.04, 0.16 / 3, 0.16 / 3, 0.5], rtol=1e-12)
    np.testing.assert_array_equal(reject, [True, False, False, False])


def test_by_is_more_conservative_than_bh() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 0.2, 40)
    _, bh = benjamini_hochberg(p)
    _, by = benjamini_yekutieli(p)
    assert np.all(by >= bh)
    assert np.all(bh >= p)


def test_accepts_lists_and_returns_arrays() -> None:
    reject, adjusted = benjamini_hochberg([0.01, 0.02])
    assert isinstance(reject, np.ndarray) and isinstance(adjusted, np.ndarray)


@pytest.mark.parametrize("func", [benjamini_hochberg, benjamini_yekutieli])
def test_empty_input_returns_empty_arrays(func) -> None:
    reject, adjusted = func([])
    assert reject.shape == (0,) and reject.dtype == bool
    assert adjusted.shape == (0,) and adjusted.dtype == float


@pytest.mark.parametrize("func", [benjamini_hochberg, benjamini_yekutieli])
def test_nan_raises_value_error_in_spanish(func) -> None:
    with pytest.raises(ValueError, match="faltantes"):
        func([0.01, np.nan, 0.2])


@pytest.mark.parametrize("bad", [[-0.1, 0.2], [0.5, 1.2]])
def test_out_of_range_raises(bad) -> None:
    with pytest.raises(ValueError, match="entre 0 y 1"):
        benjamini_hochberg(bad)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.05])
def test_invalid_q_raises(q: float) -> None:
    with pytest.raises(ValueError, match="q"):
        benjamini_hochberg([0.01], q=q)


def test_adjust_pvalues_dispatch() -> None:
    p = [0.001, 0.02, 0.03, 0.6]
    for method, func in (("bh", benjamini_hochberg), ("by", benjamini_yekutieli)):
        r1, a1 = adjust_pvalues(p, method=method)
        r2, a2 = func(p)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(a1, a2)
    with pytest.raises(ValueError, match="desconocido"):
        adjust_pvalues(p, method="bonferroni")  # type: ignore[arg-type]
