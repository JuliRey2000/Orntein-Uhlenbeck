"""Sin información futura de extremo a extremo (sección 14): datos → análisis → backtest.

Con el split congelado, modificar los precios POSTERIORES a una fecha t de la
validación no puede cambiar nada de la formación ni ninguna fila diaria u
operación de validación hasta t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import cointegrated_log_prices
from integration_helpers import bdates, csv_provider, pair_config, service

from mrlab.analysis.runner import run_pair
from mrlab.domain.models import Segment

N = 1500


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lookahead")
    log_a, log_b = cointegrated_log_prices(n=N, seed=11, lam=0.12, sigma=0.012)
    original = run_pair(
        pair_config("csv"), service(tmp / "cache", csv=csv_provider(tmp / "v1", log_a, log_b))
    )
    validation = original.backtests[Segment.VALIDATION].daily
    t = pd.Timestamp(validation["date"].iloc[150])

    # Precios posteriores a t muy distintos (otra trayectoria del spread y un salto en B).
    after = bdates(N) > t
    rng = np.random.default_rng(99)
    log_a2, log_b2 = log_a.copy(), log_b.copy()
    log_a2[after] += np.cumsum(rng.normal(0.0, 0.03, after.sum()))
    log_b2[after] -= 0.2
    frozen = pair_config("csv", frozen_split=original.config.frozen_split)
    modified = run_pair(
        frozen, service(tmp / "cache", csv=csv_provider(tmp / "v2", log_a2, log_b2))
    )
    return original, modified, t


def test_split_is_frozen(runs):
    original, modified, _ = runs
    assert modified.config.frozen_split == original.config.frozen_split
    assert modified.analysis.split == original.analysis.split
    # Los datos sí cambiaron (otra instantánea), así que la prueba tiene sentido.
    assert modified.config.pair.a.snapshot_id != original.config.pair.a.snapshot_id


def test_training_analysis_is_unchanged(runs):
    original, modified, _ = runs
    a, m = original.analysis, modified.analysis
    assert m.status == a.status and m.causes == a.causes
    assert m.integration == a.integration
    assert m.eg == a.eg and m.eg_p_adjusted == a.eg_p_adjusted
    assert m.eg_decision == a.eg_decision and m.stability == a.stability
    assert m.hedge == a.hedge and m.ou == a.ou
    assert m.forecasts == a.forecasts and m.residuals == a.residuals
    assert m.ou_stability == a.ou_stability


def test_zscore_until_t_is_unchanged(runs):
    original, modified, t = runs
    pd.testing.assert_series_equal(
        original.analysis.zscore.loc[:t], modified.analysis.zscore.loc[:t]
    )
    assert (
        not original.analysis.zscore.loc[t:]
        .iloc[1:]
        .equals(modified.analysis.zscore.loc[t:].iloc[1:])
    )


def test_training_backtest_is_unchanged(runs):
    original, modified, _ = runs
    before, after = original.backtests[Segment.TRAIN], modified.backtests[Segment.TRAIN]
    pd.testing.assert_frame_equal(before.daily, after.daily)
    assert before.trades == after.trades and before.events == after.events


def test_validation_rows_and_trades_until_t_are_unchanged(runs):
    original, modified, t = runs
    v1 = original.backtests[Segment.VALIDATION]
    v2 = modified.backtests[Segment.VALIDATION]
    upto1 = v1.daily[v1.daily["date"] <= t].reset_index(drop=True)
    upto2 = v2.daily[v2.daily["date"] <= t].reset_index(drop=True)
    assert len(upto1) == 151
    pd.testing.assert_frame_equal(upto1, upto2)

    day = t.date()
    closed1 = [x for x in v1.trades if x.exit_date is not None and x.exit_date <= day]
    closed2 = [x for x in v2.trades if x.exit_date is not None and x.exit_date <= day]
    assert closed1 == closed2
    entry_fields = ("trade_id", "side", "entry_signal_date", "entry_date", "entry_z", "q_y", "q_x")
    entered1 = [x for x in v1.trades if x.entry_date <= day]
    entered2 = [x for x in v2.trades if x.entry_date <= day]
    assert [x.model_dump(include=set(entry_fields)) for x in entered1] == [
        x.model_dump(include=set(entry_fields)) for x in entered2
    ]
    assert len(closed1) >= 2, "hacen falta operaciones cerradas antes de t para que la prueba sirva"
    assert [e for e in v1.events if e.date <= day] == [e for e in v2.events if e.date <= day]

    # Después de t los resultados sí cambian.
    later1 = v1.daily[v1.daily["date"] > t]["equity"].to_numpy()
    later2 = v2.daily[v2.daily["date"] > t]["equity"].to_numpy()
    assert not np.array_equal(later1, later2)
