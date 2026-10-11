"""StudyStore: configuración congelada y registro persistente de la prueba (decisión S10 y su versión persistente de la fase 2)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from mrlab.analysis.pair_pipeline import AnalysisConfig
from mrlab.analysis.runner import run_pair
from mrlab.backtest.split import SplitConfig
from mrlab.data.cache import SnapshotCache
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.data.registry import ProviderRegistry
from mrlab.data.service import DataService
from mrlab.domain.errors import AnalysisBlocked, ConfigError, DataError
from mrlab.domain.models import Segment
from mrlab.reporting.export import export_files
from mrlab.reporting.run_config import RunConfig, SeriesRef, config_hash
from mrlab.strategy.params import CostParams, StrategyParams
from mrlab.studies import (
    Study,
    StudyStore,
    contamination_note,
    contamination_text,
    hash_with_test,
)

START, END = "2018-01-01", "2023-12-31"


def _config(prefix: str = "SINT-COI", **extra) -> RunConfig:
    ref = {"provider": "synthetic", "start": START, "end": END}
    return RunConfig(
        pair={
            "a": SeriesRef(symbol=f"{prefix}-A", **ref),
            "b": SeriesRef(symbol=f"{prefix}-B", **ref),
        },
        **extra,
    )


@pytest.fixture
def data(tmp_path: Path) -> DataService:
    registry = ProviderRegistry({"synthetic": SyntheticProvider()})
    return DataService(registry, SnapshotCache(tmp_path / "cache"))


@pytest.fixture
def store(tmp_path: Path) -> StudyStore:
    return StudyStore(tmp_path / "studies")


@pytest.fixture
def study(store: StudyStore, data: DataService) -> Study:
    return store.create(_config(), data, name="Par sintético")


# --------------------------------------------------------------------------- crear


def test_create_pins_snapshots_and_split(store: StudyStore, study: Study) -> None:
    cfg = study.config
    assert cfg.pair.a.snapshot_id and cfg.pair.b.snapshot_id
    assert cfg.frozen_split is not None and cfg.frozen_split.n_test > 0
    assert cfg.include_test is False and cfg.test_history == []
    assert cfg.config_hash == config_hash(cfg)
    assert study.test_runs == [] and study.changes == []
    assert not study.test_viewed and not study.contaminated
    assert len(study.id) == 32
    assert (store.directory / f"{study.id}.json").is_file()
    assert store.get(study.id) == study


def test_create_rejects_include_test(store: StudyStore, data: DataService) -> None:
    with pytest.raises(ConfigError) as err:
        store.create(_config(include_test=True), data)
    assert err.value.code == "PRUEBA_SOLO_EN_ESTUDIO"
    assert "una sola vez" in err.value.message
    assert store.list() == []


def test_default_name_uses_symbols(store: StudyStore, data: DataService) -> None:
    created = store.create(_config(), data)
    assert created.name == "SINT-COI-A / SINT-COI-B"


def test_create_imports_existing_test_history(store: StudyStore, data: DataService) -> None:
    """Un run_config.json que ya registra la prueba no se «lava» al crear un estudio."""
    base = store.create(_config(), data)
    seen_same = hash_with_test(base.config)
    same = store.create(_config(test_history=[seen_same]), data)
    assert same.test_viewed and not same.contaminated
    assert [r.source for r in same.test_runs] == ["run_config"]

    other = store.create(_config(test_history=["f" * 64]), data)
    assert other.test_viewed and other.contaminated
    assert "run_config.json" in (other.contamination_reason or "")


# --------------------------------------------------------------------------- cambios


def test_update_before_test_is_not_contamination(
    store: StudyStore, study: Study, data: DataService
) -> None:
    updated = store.update_config(study.id, strategy=StrategyParams(entry_z=2.5))
    assert not updated.contaminated and updated.contamination_reason is None
    assert updated.config.strategy.entry_z == 2.5
    assert updated.config.pair == study.config.pair  # los datos no cambian
    assert updated.config.frozen_split == study.config.frozen_split
    (change,) = updated.changes
    assert change.fields == ["strategy.entry_z"]
    assert change.after_test is False
    assert change.hash_before == study.config.config_hash
    assert change.hash_after == updated.config.config_hash != change.hash_before
    # Ejecutar la prueba después de cambiar ANTES de verla no está contaminado.
    _, result = store.run_test(study.id, data)
    assert result.test_contaminated is False


def test_update_without_differences_is_not_recorded(store: StudyStore, study: Study) -> None:
    same = store.update_config(study.id, strategy=study.config.strategy, costs=CostParams())
    assert same.changes == []


def test_update_cannot_change_frozen_split(store: StudyStore, study: Study) -> None:
    other = AnalysisConfig(split=SplitConfig(train=0.5, validation=0.25, test=0.25))
    with pytest.raises(ConfigError) as err:
        store.update_config(study.id, analysis=other)
    assert err.value.code == "DIVISION_CONGELADA"


def test_run_test_records_the_run(store: StudyStore, study: Study, data: DataService) -> None:
    updated, result = store.run_test(study.id, data)
    assert Segment.TEST in result.backtests
    assert result.test_included is True and result.test_contaminated is False
    (run,) = updated.test_runs
    assert run.source == "prueba"
    assert run.config_hash == result.config.config_hash == hash_with_test(study.config)
    assert updated.test_viewed and not updated.contaminated
    assert "1 vez" in (result.test_contamination_note or "")
    assert store.get(study.id).test_runs == updated.test_runs
    # Repetirla con la MISMA configuración no contamina.
    again, result2 = store.run_test(study.id, data)
    assert len(again.test_runs) == 2 and not again.contaminated
    assert result2.test_contaminated is False


def test_update_after_test_contaminates_permanently(
    store: StudyStore, study: Study, data: DataService
) -> None:
    store.run_test(study.id, data)
    changed = store.update_config(study.id, strategy=StrategyParams(entry_z=2.5))
    assert changed.contaminated
    reason = changed.contamination_reason or ""
    assert "strategy.entry_z" in reason and "después de ver la prueba" in reason
    assert changed.changes[-1].after_test is True
    # Volver a la configuración original NO quita la marca.
    reverted = store.update_config(study.id, strategy=StrategyParams())
    assert reverted.config.config_hash == study.config.config_hash
    assert reverted.contaminated and reverted.contamination_reason == reason
    _, result = store.run_test(study.id, data)
    assert result.test_contaminated is True
    assert "después de ver la prueba" in (result.test_contamination_note or "")
    # El historial exportado conserva la marca fuera del estudio.
    replay = run_pair(result.config, data)
    assert replay.test_contaminated is True
    assert "Prueba contaminada" in contamination_text(store.get(study.id))


def test_changing_analysis_after_test_also_counts(
    store: StudyStore, study: Study, data: DataService
) -> None:
    store.run_test(study.id, data)
    changed = store.update_config(study.id, analysis=AnalysisConfig(alpha_level=0.10))
    assert changed.contaminated
    assert "analysis.alpha_level" in (changed.contamination_reason or "")


def test_run_test_with_another_hash_contaminates(
    store: StudyStore, study: Study, data: DataService
) -> None:
    """Si la configuración cambia por fuera de ``update_config`` (p. ej. el archivo se edita),
    la segunda ejecución con otro hash marca la prueba como contaminada."""
    store.run_test(study.id, data)
    edited = store.get(study.id)
    new_cfg = edited.config.model_copy(update={"strategy": StrategyParams(exit_z=0.25)})
    edited.config = new_cfg.model_copy(update={"config_hash": config_hash(new_cfg)})
    store._write(edited)
    contaminated, result = store.run_test(study.id, data)
    assert contaminated.contaminated
    assert "se volvió a ejecutar" in (contaminated.contamination_reason or "")
    assert result.test_contaminated is True
    assert len({r.config_hash for r in contaminated.test_runs}) == 2


def test_result_requires_test_first(store: StudyStore, study: Study, data: DataService) -> None:
    with pytest.raises(ConfigError) as err:
        store.result(study.id, data, include_test=True)
    assert err.value.code == "PRUEBA_NO_EJECUTADA"
    assert "ejecuta primero la prueba" in err.value.message
    without = store.result(study.id, data)
    assert Segment.TEST not in without.backtests
    assert without.test_contaminated is None
    assert "aún no se ha ejecutado" in (without.test_contamination_note or "")


def test_result_is_deterministic_and_reflects_contamination(
    store: StudyStore, study: Study, data: DataService
) -> None:
    store.run_test(study.id, data)
    first = store.result(study.id, data, include_test=True)
    second = store.result(study.id, data, include_test=True)
    assert export_files(first) == export_files(second)
    assert first.test_contaminated is False
    assert len(store.get(study.id).test_runs) == 1  # mostrar no registra una ejecución nueva
    store.update_config(study.id, costs=CostParams(commission_bps=10.0))
    after = store.result(study.id, data, include_test=True)
    assert after.test_contaminated is True
    runs = store.get(study.id).test_runs
    assert [r.source for r in runs] == ["prueba", "exportacion"]  # configuración nueva: se registra
    readme = next(v for k, v in export_files(after).items() if k.endswith("LEEME.txt"))
    text = readme.decode("utf-8")
    assert "Prueba contaminada: SÍ — según el registro persistente del estudio" in text
    assert "costs.commission_bps" in text


def test_run_test_on_blocked_pair_records_nothing(store: StudyStore, data: DataService) -> None:
    blocked = store.create(_config("SINT-IND"), data)
    with pytest.raises(AnalysisBlocked) as err:
        store.run_test(blocked.id, data)
    assert err.value.code == "PRUEBA_NO_SIMULABLE"
    assert "bloqueada" in err.value.message
    assert store.get(blocked.id).test_runs == []


def test_run_test_without_test_period(store: StudyStore, data: DataService) -> None:
    no_test = AnalysisConfig(split=SplitConfig(train=0.7, validation=0.3, test=0.0))
    created = store.create(_config(analysis=no_test), data)
    with pytest.raises(ConfigError) as err:
        store.run_test(created.id, data)
    assert err.value.code == "SIN_PERIODO_PRUEBA"


# --------------------------------------------------------------------------- persistencia


def test_state_survives_restart(tmp_path: Path, data: DataService) -> None:
    first = StudyStore(tmp_path / "studies")
    created = first.create(_config(), data)
    first.run_test(created.id, data)
    first.update_config(created.id, strategy=StrategyParams(entry_z=2.25))

    restarted = StudyStore(tmp_path / "studies")  # nuevo proceso / reinicio
    loaded = restarted.get(created.id)
    assert loaded.contaminated and loaded.test_viewed
    assert loaded.config.strategy.entry_z == 2.25
    assert loaded.config.config_hash == config_hash(loaded.config)
    _, result = restarted.run_test(created.id, data)
    assert result.test_contaminated is True


def test_list_and_delete(store: StudyStore, data: DataService) -> None:
    a = store.create(_config(), data, name="primero")
    b = store.create(_config(), data, name="segundo")
    (store.directory / "notas.json").write_text("{}", encoding="utf-8")  # se ignora
    assert [s.id for s in store.list()] == [b.id, a.id]  # más recientes primero
    store.delete(a.id)
    assert [s.id for s in store.list()] == [b.id]
    with pytest.raises(DataError) as err:
        store.get(a.id)
    assert err.value.code == "ESTUDIO_NO_ENCONTRADO"
    with pytest.raises(DataError):
        store.delete(a.id)


@pytest.mark.parametrize(
    "bad_id",
    ["../x", "..", "", "abc", "A" * 32, "0" * 31 + "/", "../" + "0" * 29, "0" * 33, None],
)
def test_dangerous_ids_are_not_found(store: StudyStore, bad_id) -> None:
    for action in (store.get, store.delete):
        with pytest.raises(DataError) as err:
            action(bad_id)
        assert err.value.code == "ESTUDIO_NO_ENCONTRADO"
    with pytest.raises(DataError):
        store.update_config(bad_id, strategy=StrategyParams())


def test_corrupt_file_is_reported_in_spanish(store: StudyStore, study: Study) -> None:
    (store.directory / f"{study.id}.json").write_text("{no es json", encoding="utf-8")
    with pytest.raises(DataError) as err:
        store.get(study.id)
    assert err.value.code == "ESTUDIO_CORRUPTO"
    assert "dañado" in err.value.message
    assert store.list() == []  # se omite en el listado


def test_atomic_write_keeps_previous_file(
    store: StudyStore, study: Study, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.directory / f"{study.id}.json"
    before = path.read_bytes()

    def boom(src, dst):
        raise OSError("disco lleno (simulado)")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.update_config(study.id, strategy=StrategyParams(entry_z=2.5))
    assert path.read_bytes() == before
    assert sorted(p.name for p in store.directory.iterdir()) == [path.name]  # sin temporales
    json.loads(before)


def test_concurrent_updates_keep_every_change(store: StudyStore, study: Study) -> None:
    def change(z: float) -> None:
        store.update_config(study.id, strategy=StrategyParams(entry_z=z))

    threads = [threading.Thread(target=change, args=(2.0 + i / 10,)) for i in range(1, 7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.get(study.id).changes) == 6


def test_contamination_note_texts(store: StudyStore, study: Study, data: DataService) -> None:
    assert "aún no se ha ejecutado" in contamination_note(study, included=False)
    viewed, _ = store.run_test(study.id, data)
    assert "no se cambió después de verla" in contamination_note(viewed)
    assert "no incluye la prueba" in contamination_note(viewed, included=False)
    assert contamination_text(viewed).startswith("Prueba no contaminada")
