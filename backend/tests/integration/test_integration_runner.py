"""``run_pair``: qué segmentos se simulan, avisos que viajan con cada backtest y descartes del runner."""

from __future__ import annotations

import numpy as np
import pytest
from integration_helpers import csv_provider, explosive_spread, pair_config, service

from mrlab.analysis.pair_pipeline import ILLUSTRATIVE_BANNER, AnalysisConfig, PairStatus
from mrlab.analysis.runner import ROLLING_Z_WARNING, run_pair
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.domain.models import DataOrigin, Segment
from mrlab.stats.cointegration import Direction
from mrlab.strategy.params import CostParams, StrategyParams


@pytest.fixture(autouse=True)
def fixed_code_version(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "prueba-runner")


@pytest.fixture
def synthetic(tmp_path):
    return service(tmp_path / "cache", synthetic=SyntheticProvider())


def synthetic_config(prefix: str, **updates):
    return pair_config("synthetic", (f"{prefix}-A", f"{prefix}-B"), **updates)


def test_evidence_pair_runs_train_and_validation_backtests(synthetic):
    result = run_pair(synthetic_config("SINT-COI"), synthetic)
    assert result.analysis.status == PairStatus.EVIDENCE
    assert result.is_synthetic and result.config.data_origin == DataOrigin.SYNTHETIC
    assert set(result.backtests) == {Segment.TRAIN, Segment.VALIDATION}
    train, validation = result.backtests[Segment.TRAIN], result.backtests[Segment.VALIDATION]
    assert len(train.daily) == result.analysis.split.n_train
    assert len(validation.daily) == result.analysis.split.n_validation
    assert any("DENTRO DE MUESTRA" in w for w in train.warnings)
    assert ILLUSTRATIVE_BANNER not in train.warnings + validation.warnings
    # La pierna «y» es la dependiente de la dirección elegida; β y vida media de formación.
    assert train.leg_y.symbol == result.analysis.hedge.dependent
    assert train.beta == result.analysis.hedge.beta
    assert train.max_holding_bars == StrategyParams().max_holding_bars(result.analysis.ou.half_life)


def test_pair_without_evidence_is_only_illustrative(synthetic):
    result = run_pair(synthetic_config("SINT-IND"), synthetic)
    assert result.analysis.status != PairStatus.EVIDENCE
    assert result.analysis.strategy_allowed  # OU válido (decisión O5)
    for bt in result.backtests.values():
        assert bt.warnings[0] == ILLUSTRATIVE_BANNER


def test_invalid_ou_runs_no_backtest(tmp_path):
    log_a, log_b = explosive_spread(seed=1)
    data = service(tmp_path / "cache", csv=csv_provider(tmp_path / "csv", log_a, log_b))
    cfg = pair_config("csv", analysis=AnalysisConfig(direction=Direction.A_ON_B))
    result = run_pair(cfg, data, include_test=True)
    assert not result.analysis.strategy_allowed
    assert result.backtests == {}
    assert result.config.frozen_split is not None  # el split se congela igualmente


def test_rolling_zscore_mode_is_flagged_as_pedagogical(synthetic):
    cfg = synthetic_config(
        "SINT-COI", strategy=StrategyParams(zscore_mode="movil", rolling_window=63)
    )
    result = run_pair(cfg, synthetic)
    train = result.backtests[Segment.TRAIN]
    assert ROLLING_Z_WARNING in train.warnings
    fixed = result.analysis.zscore.loc[train.daily["date"]].to_numpy()
    assert np.isnan(train.daily["z"].iloc[:62]).all()  # calentamiento de la ventana móvil
    assert not np.allclose(train.daily["z"].iloc[62:], fixed[62:])


def test_costs_above_gross_pnl_in_validation_is_a_discard_reason(synthetic):
    expensive = CostParams(commission_bps=150.0)
    result = run_pair(synthetic_config("SINT-COI", costs=expensive), synthetic)
    m = result.backtests[Segment.VALIDATION].metrics
    assert m.gross_pnl > 0 and m.total_costs > m.gross_pnl
    reason = next(c for c in result.analysis.discard_reasons if c.code == "COSTES_SUPERAN_BRUTO")
    assert reason.status is None and "validación" in reason.message
    assert result.analysis.status == PairStatus.EVIDENCE  # no cambia el estado estadístico
