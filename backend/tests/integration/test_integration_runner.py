"""``run_pair``: qué segmentos se simulan, avisos que viajan con cada backtest y descartes del runner."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import cointegrated_log_prices
from integration_helpers import (
    StubData,
    broken_relationship,
    csv_provider,
    explosive_spread,
    pair_config,
    series_pair,
    service,
    stub_config,
    with_symbol,
)

from mrlab.analysis import runner
from mrlab.analysis.pair_pipeline import (
    ILLUSTRATIVE_BANNER,
    STATUS_LABELS,
    AnalysisConfig,
    PairStatus,
)
from mrlab.analysis.runner import ROLLING_Z_WARNING, run_pair
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.domain.errors import DataError
from mrlab.domain.models import AssetClass, DataOrigin, Segment
from mrlab.reporting.export import export_files, readme_txt
from mrlab.reporting.run_config import config_hash, load_run_config, save_run_config
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


def test_pair_without_evidence_is_only_illustrative():
    # Relación que se rompe en formación: ⚠️ resultado inestable (no ✅), pero con OU válido
    # (decisión O5). Con la O7 refinada la validez del OU coincide con Engle–Granger SIN AJUSTAR
    # (p = 0,012 aquí); con break_end=700 ese p es 0,067 y el OU ya no es válido.
    a, b = series_pair(*broken_relationship(break_start=300, break_end=650))
    result = run_pair(stub_config(), StubData([a, b]))
    analysis = result.analysis
    assert analysis.status != PairStatus.EVIDENCE
    assert analysis.strategy_allowed and analysis.illustrative_only
    assert set(result.backtests) == {Segment.TRAIN, Segment.VALIDATION}
    for bt in result.backtests.values():
        # El banner encabeza los avisos y cita los motivos (el estado del par).
        assert bt.warnings[0].startswith(ILLUSTRATIVE_BANNER)
        assert bt.warnings[0] == analysis.illustrative_banner
        assert STATUS_LABELS[analysis.status] in bt.warnings[0]


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
    # Decisión O5: toda razón de descarte (también las que añade el runner al simular) hace
    # el resultado solo ilustrativo, y el banner de cada backtest la cita.
    analysis = result.analysis
    assert analysis.illustrative_only and analysis.strategy_allowed
    assert reason.message in analysis.illustrative_reasons
    for bt in result.backtests.values():
        assert bt.warnings[0] == analysis.illustrative_banner
        assert reason.message in bt.warnings[0]
        assert sum(w.startswith(ILLUSTRATIVE_BANNER) for w in bt.warnings) == 1
    assert analysis.warnings.count(analysis.illustrative_banner) == 1
    stats = export_files(result)["SINTETICO_pruebas_estadisticas.csv"].decode("utf-8")
    assert "estado,par,solo_ilustrativo,sí" in stats


# --------------------------------------------------------------------------- «solo ilustrativo» con motivos


def test_evidence_pair_with_too_short_half_life_is_illustrative_with_its_reason():
    # Hallazgo: un par ✅ con vida media < 1 se presentaba sin el banner «solo ilustrativo».
    a, b = series_pair(*cointegrated_log_prices(n=1500, seed=4, lam=1.2, sigma=0.03))
    result = run_pair(stub_config(), StubData([a, b]))
    analysis = result.analysis
    assert analysis.status == PairStatus.EVIDENCE and analysis.strategy_allowed
    short = next(c for c in analysis.discard_reasons if c.code == "VIDA_MEDIA_CORTA")
    assert result.backtests
    for bt in result.backtests.values():
        assert bt.warnings[0].startswith(ILLUSTRATIVE_BANNER)
        assert short.message in bt.warnings[0]  # el banner cita la razón concreta
    readme = readme_txt(result)
    assert readme.count(analysis.illustrative_banner) == 1  # sin duplicados en el LEEME


# --------------------------------------------------------------------------- cortos de cripto spot


def test_simulated_crypto_spot_short_is_warned_in_backtests_and_readme():
    la, lb = cointegrated_log_prices(n=1500, seed=11, lam=0.12, sigma=0.012)
    a, b = series_pair(la, lb)
    b = with_symbol(b, "BTC-USD", asset_class=AssetClass.CRYPTO)
    result = run_pair(stub_config(("AAA", "BTC-USD")), StubData([a, b]))
    shorted = [
        bt
        for bt in result.backtests.values()
        if any((t.q_x if bt.leg_x.symbol == "BTC-USD" else t.q_y) < 0 for t in bt.trades)
    ]
    assert shorted, "el escenario debe vender en corto la pierna cripto"
    for bt in shorted:
        assert any("derivados" in w and "BTC-USD" in w for w in bt.warnings)
    assert "perpetuos con funding" in readme_txt(result)


# --------------------------------------------------------------------------- semillas


def test_seeds_are_recorded_from_the_synthetic_provider(synthetic):
    result = run_pair(synthetic_config("SINT-COI"), synthetic)
    assert result.config.seeds == {"SINT-COI-A": 7, "SINT-COI-B": 7}
    assert result.config.config_hash == config_hash(result.config)


def test_seeds_come_from_series_metadata_and_keep_user_seeds():
    a, b = series_pair(*cointegrated_log_prices(n=1000, seed=3))
    a = a.with_meta(seed=123)  # serie generada con semilla conocida
    result = run_pair(stub_config(seeds={"bootstrap": 5}), StubData([a, b]))
    assert result.config.seeds == {"AAA": 123, "bootstrap": 5}


# --------------------------------------------------------------------------- un segmento que falla


def test_a_failing_segment_does_not_lose_the_analysis(monkeypatch):
    # Defensa en profundidad: si el motor rechaza un segmento (LabError), el resto se conserva.
    real = runner.run_backtest

    def flaky(inp, params, costs):
        if inp.segment == Segment.VALIDATION:
            raise DataError(
                "Hay precios faltantes o no positivos en el segmento (primera fecha afectada: "
                "2019-03-21).",
                code="PRECIOS_INVALIDOS",
            )
        return real(inp, params, costs)

    monkeypatch.setattr(runner, "run_backtest", flaky)
    a, b = series_pair(*cointegrated_log_prices(n=1200, seed=21, lam=0.1, sigma=0.012))
    result = run_pair(stub_config(), StubData([a, b]), include_test=True)
    assert set(result.backtests) == {Segment.TRAIN, Segment.TEST}
    (cause,) = [c for c in result.analysis.discard_reasons if c.code == "SIMULACION_NO_POSIBLE"]
    assert "validación" in cause.message and "2019-03-21" in cause.message
    assert Segment.VALIDATION in result.segment_errors
    assert result.analysis.illustrative_only  # decisión O5: es una razón de descarte
    for bt in result.backtests.values():
        assert bt.warnings[0] == result.analysis.illustrative_banner
        assert cause.message in bt.warnings[0]
    readme = export_files(result)["LEEME.txt"].decode("utf-8")
    assert "no se pudo simular" in readme and "2019-03-21" in readme


def test_errors_that_are_not_lab_errors_still_propagate(monkeypatch):
    def broken(inp, params, costs):
        raise ZeroDivisionError("defecto de programación")

    monkeypatch.setattr(runner, "run_backtest", broken)
    a, b = series_pair(*cointegrated_log_prices(n=1200, seed=21, lam=0.1, sigma=0.012))
    with pytest.raises(ZeroDivisionError):
        run_pair(stub_config(), StubData([a, b]))


# --------------------------------------------------------------------------- prueba contaminada


@pytest.fixture
def csv_data(tmp_path):
    log_a, log_b = cointegrated_log_prices(n=1200, seed=21, lam=0.1, sigma=0.012)
    return service(tmp_path / "cache", csv=csv_provider(tmp_path / "uploads", log_a, log_b))


def _readme(result) -> str:
    return export_files(result)["LEEME.txt"].decode("utf-8")


def test_first_test_run_is_recorded_and_not_contaminated(csv_data):
    result = run_pair(pair_config("csv"), csv_data, include_test=True)
    cfg = result.config
    assert cfg.include_test is True
    assert cfg.test_history == [cfg.config_hash]
    assert result.test_contaminated is False
    assert "Prueba contaminada: no (" in _readme(result)


def test_without_test_contamination_does_not_apply(csv_data):
    result = run_pair(pair_config("csv"), csv_data)
    assert result.config.include_test is False and result.config.test_history == []
    assert result.test_contaminated is None
    assert "Prueba contaminada: no aplica: la prueba no se ejecutó" in _readme(result)


def test_hand_edited_thresholds_rerun_on_the_test_are_contaminated(csv_data, tmp_path):
    # Hallazgo: se veía la prueba, se editaban los umbrales del run_config.json (quitando el
    # hash) y se volvía a ejecutar la prueba; la exportación decía «Prueba contaminada: no.».
    original = run_pair(pair_config("csv"), csv_data, include_test=True)
    path = save_run_config(original.config, tmp_path / "run_config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["strategy"].update(entry_z=1.5, exit_z=0.2)
    del raw["config_hash"]
    edited_path = tmp_path / "editado.json"
    edited_path.write_text(json.dumps(raw), encoding="utf-8")
    edited = load_run_config(edited_path)
    assert edited.config_hash != original.config.config_hash

    rerun = run_pair(edited, csv_data, include_test=True)
    assert Segment.TEST in rerun.backtests
    assert rerun.test_contaminated is True
    assert rerun.config.test_history == [original.config.config_hash, edited.config_hash]
    readme = _readme(rerun)
    assert "Prueba contaminada: SÍ" in readme and "Prueba contaminada: no" not in readme
    rows = export_files(rerun)["pruebas_estadisticas.csv"].decode("utf-8").splitlines()
    assert "HISTORICO,prueba,par,contaminada,sí" in rows

    # La marca es permanente: volver a la configuración original con el historial completo
    # sigue marcando la prueba como contaminada.
    back = original.config.model_copy(update={"test_history": rerun.config.test_history})
    again = run_pair(back, csv_data)
    assert again.test_contaminated is True
    assert again.config.test_history == rerun.config.test_history


def test_rerunning_the_same_config_is_not_contamination(csv_data):
    first = run_pair(pair_config("csv"), csv_data, include_test=True)
    replay = run_pair(first.config, csv_data)  # include_test viene de la configuración
    assert Segment.TEST in replay.backtests
    assert replay.test_contaminated is False
    assert replay.config == first.config


def test_running_without_test_keeps_the_history(csv_data):
    first = run_pair(pair_config("csv"), csv_data, include_test=True)
    no_test = first.config.model_copy(
        update={"include_test": False, "strategy": StrategyParams(entry_z=1.8)}
    )
    later = run_pair(no_test, csv_data)
    assert Segment.TEST not in later.backtests and later.test_contaminated is None
    assert later.config.test_history == first.config.test_history
