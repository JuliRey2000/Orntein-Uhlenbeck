"""Proveedor SINTÉTICO: determinismo, etiquetas y propiedades estadísticas por escenario."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from mrlab.data.providers.synthetic import (
    PRESET_SYMBOLS,
    SyntheticPairParams,
    SyntheticProvider,
    SyntheticScenario,
    effective_params,
    generate_pair,
    preset_params,
)
from mrlab.domain.errors import ConfigError, ProviderError
from mrlab.domain.models import AssetClass, DataOrigin, PriceType


def log_prices(params: SyntheticPairParams) -> tuple[np.ndarray, np.ndarray]:
    a, b = generate_pair(params)
    return np.log(a.close.to_numpy()), np.log(b.close.to_numpy())


def eg_pvalue(y: np.ndarray, x: np.ndarray) -> float:
    return float(coint(y, x, trend="c", maxlag=12, autolag="aic")[1])


def return_corr(la: np.ndarray, lb: np.ndarray) -> float:
    return float(np.corrcoef(np.diff(la), np.diff(lb))[0, 1])


def test_same_seed_same_data_different_seed_different_data():
    p = SyntheticPairParams(seed=11)
    a1, b1 = generate_pair(p)
    a2, b2 = generate_pair(SyntheticPairParams(seed=11))
    pd.testing.assert_frame_equal(a1.data, a2.data)
    pd.testing.assert_frame_equal(b1.data, b2.data)
    a3, _ = generate_pair(SyntheticPairParams(seed=12))
    assert not np.allclose(a1.close.to_numpy(), a3.close.to_numpy())


@pytest.mark.parametrize("scenario", list(SyntheticScenario))
def test_series_are_always_labelled_synthetic(scenario):
    a, b = generate_pair(SyntheticPairParams(scenario=scenario, n_obs=300))
    for s in (a, b):
        assert s.is_synthetic and s.meta.asset.origin == DataOrigin.SYNTHETIC
        assert s.meta.price_type == PriceType.SYNTHETIC
        assert s.meta.asset.name.startswith("SINTÉTICO — ")
        assert s.meta.source_note.startswith("SINTÉTICO")
        assert "semilla 7" in s.meta.source_note
        assert any("SINTÉTICOS" in n for n in s.meta.asset.notes)
        assert s.meta.retrieved_at is None
        assert s.meta.asset.currency == "USD" and s.meta.asset.asset_class == AssetClass.EQUITY
        assert s.meta.provider == "synthetic"
        idx = s.data.index
        assert len(idx) == 300 and (idx.dayofweek < 5).all() and idx.is_unique
        assert idx.name == "date" and (s.close > 0).all()


def test_cointegrated_identity_and_ou_spread():
    p = SyntheticPairParams(alpha=0.3, beta=1.5, x0=0.05, ou_mu=0.0)
    la, lb = log_prices(p)
    x = la - 0.3 - 1.5 * lb
    assert x[0] == pytest.approx(0.05)
    assert lb[0] == pytest.approx(np.log(100.0))
    # AR(1) del spread ≈ e^{−λ}: es un OU (transición exacta), no un paseo aleatorio
    b_hat = np.polyfit(x[:-1], x[1:], 1)[0]
    assert abs(b_hat - np.exp(-0.1)) < 0.04


def test_cointegrated_scenario_is_detected():
    la, lb = log_prices(SyntheticPairParams(scenario=SyntheticScenario.COINTEGRATED, n_obs=1500))
    assert eg_pvalue(la, lb) < 0.01


def test_correlated_not_cointegrated_scenario():
    la, lb = log_prices(
        SyntheticPairParams(scenario=SyntheticScenario.CORRELATED_NOT_COINTEGRATED, n_obs=1500)
    )
    assert return_corr(la, lb) > 0.8
    assert eg_pvalue(la, lb) > 0.05
    assert eg_pvalue(lb, la) > 0.05


def test_cointegrated_low_correlation_scenario():
    la, lb = log_prices(
        SyntheticPairParams(scenario=SyntheticScenario.COINTEGRATED_LOW_CORRELATION, n_obs=1500)
    )
    assert return_corr(la, lb) < 0.5
    assert eg_pvalue(la, lb) < 0.05


def test_independent_scenario_has_low_correlation():
    la, lb = log_prices(SyntheticPairParams(scenario=SyntheticScenario.INDEPENDENT, n_obs=1500))
    assert abs(return_corr(la, lb)) < 0.2


def test_scenario_defaults_only_fill_unset_fields():
    implicit = effective_params(
        SyntheticPairParams(scenario=SyntheticScenario.COINTEGRATED_LOW_CORRELATION)
    )
    assert (implicit.b_vol, implicit.ou_sigma, implicit.ou_lambda) == (0.004, 0.015, 0.1)
    explicit = effective_params(
        SyntheticPairParams(scenario=SyntheticScenario.COINTEGRATED_LOW_CORRELATION, b_vol=0.01)
    )
    assert explicit.b_vol == 0.01 and explicit.ou_sigma == 0.015
    assert effective_params(SyntheticPairParams()).b_vol == 0.01


@pytest.mark.parametrize(
    "params",
    [
        SyntheticPairParams(n_obs=1),
        SyntheticPairParams(ou_lambda=0.0),
        SyntheticPairParams(ou_sigma=-0.01),
        SyntheticPairParams(b_start_price=0.0),
        SyntheticPairParams(scenario=SyntheticScenario.CORRELATED_NOT_COINTEGRATED, rho=1.5),
        SyntheticPairParams(b_vol=float("nan")),
    ],
)
def test_invalid_parameters_are_rejected_in_spanish(params):
    with pytest.raises(ConfigError) as exc_info:
        generate_pair(params)
    assert exc_info.value.code == "PARAMETROS_SINTETICOS_INVALIDOS"
    assert "no válidos" in exc_info.value.message


# --------------------------------------------------------------------------- proveedor


def test_trimming_by_range_does_not_change_values():
    provider = SyntheticProvider()
    full = provider.get_history("SINT-COI-A", date(2000, 1, 1), date(2100, 1, 1))
    part = provider.get_history("SINT-COI-A", date(2019, 3, 1), date(2019, 6, 28))
    assert full.meta.n_obs == 1500
    assert part.data.index.min() >= pd.Timestamp("2019-03-01")
    assert part.data.index.max() <= pd.Timestamp("2019-06-28")
    pd.testing.assert_frame_equal(part.data, full.data.loc[part.data.index])
    assert part.meta.requested_start == date(2019, 3, 1)
    assert part.meta.coverage_start == part.data.index.min().date()
    assert part.meta.n_obs == len(part.data)


@pytest.mark.parametrize("prefix", list(PRESET_SYMBOLS))
def test_preset_symbols_are_fixed_with_seed_7(prefix):
    provider = SyntheticProvider()
    expected_a, expected_b = generate_pair(preset_params(prefix), symbol_prefix=prefix)
    a = provider.get_history(f"{prefix}-A", date(2000, 1, 1), date(2100, 1, 1))
    b = provider.get_history(f"{prefix.lower()}-b", date(2000, 1, 1), date(2100, 1, 1))
    assert preset_params(prefix).seed == 7
    assert a.symbol == f"{prefix}-A" and b.symbol == f"{prefix}-B"
    pd.testing.assert_frame_equal(a.data, expected_a.data)
    pd.testing.assert_frame_equal(b.data, expected_b.data)
    assert provider.get_metadata(f"{prefix}-A").name.startswith("SINTÉTICO — ")


def test_user_symbols_are_deterministic():
    params = SyntheticPairParams(
        scenario=SyntheticScenario.CORRELATED_NOT_COINTEGRATED, seed=3, rho=0.7
    )
    sym_a, sym_b = SyntheticProvider().register(params)
    again = SyntheticProvider().register(
        SyntheticPairParams(scenario="correlado_no_cointegrado", seed=3, rho=0.7)
    )
    assert (sym_a, sym_b) == again
    assert re.fullmatch(r"SINT-U[0-9A-F]{8}-A", sym_a) and sym_b == sym_a[:-1] + "B"
    canonical = json.dumps(
        effective_params(params).model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    assert sym_a[6:14] == hashlib.sha256(canonical.encode()).hexdigest()[:8].upper()
    other, _ = SyntheticProvider().register(SyntheticPairParams(seed=4))
    assert other != sym_a


def test_registered_pair_matches_generate_pair():
    provider = SyntheticProvider()
    params = SyntheticPairParams(seed=21, n_obs=400)
    sym_a, _ = provider.register(params)
    expected, _ = generate_pair(params)
    got = provider.get_history(sym_a, date(2000, 1, 1), date(2100, 1, 1))
    assert expected.symbol == sym_a
    pd.testing.assert_frame_equal(got.data, expected.data)
    assert "usuario" in got.meta.asset.name


def test_unknown_symbol_or_empty_range():
    provider = SyntheticProvider()
    for symbol in ("SINT-XYZ-A", "SINT-COI", "KO"):
        with pytest.raises(ProviderError) as exc_info:
            provider.get_history(symbol, date(2018, 1, 1), date(2020, 1, 1))
        assert exc_info.value.code == "SIMBOLO_SIN_DATOS"
        assert "SINT-COI-A/B" in exc_info.value.message
    with pytest.raises(ProviderError) as exc_info:
        provider.get_history("SINT-COI-A", date(1990, 1, 1), date(1990, 12, 31))
    assert exc_info.value.code == "SIMBOLO_SIN_DATOS"


def test_search_over_presets_and_registered():
    provider = SyntheticProvider()
    assert len(provider.search("")) == 2 * len(PRESET_SYMBOLS)
    assert [a.symbol for a in provider.search("sint-coi")] == ["SINT-COI-A", "SINT-COI-B"]
    assert {a.symbol for a in provider.search("baja correlación")} == {"SINT-CBC-A", "SINT-CBC-B"}
    sym_a, _ = provider.register(SyntheticPairParams(seed=99))
    assert [a.symbol for a in provider.search(sym_a[:10].lower())] == [sym_a, sym_a[:-1] + "B"]
