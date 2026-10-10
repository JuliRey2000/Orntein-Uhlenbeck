"""Reproducibilidad (secciones 14 y 15): misma configuración + mismos datos en caché = mismos bytes."""

from __future__ import annotations

import json

import pytest
from conftest import cointegrated_log_prices
from integration_helpers import csv_provider, pair_config, service

from mrlab.analysis.runner import run_pair
from mrlab.domain.errors import ConfigError, DataError
from mrlab.domain.models import DataOrigin, Segment
from mrlab.reporting.export import export_zip_bytes
from mrlab.reporting.run_config import config_hash, load_run_config, save_run_config


@pytest.fixture(autouse=True)
def fixed_code_version(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "prueba-reproducibilidad")


@pytest.fixture
def csv_service(tmp_path):
    log_a, log_b = cointegrated_log_prices(n=1200, seed=21, lam=0.1, sigma=0.012)
    provider = csv_provider(tmp_path / "uploads", log_a, log_b)
    return service(tmp_path / "cache", csv=provider)


def test_same_config_and_cached_data_give_identical_export(csv_service):
    first = run_pair(pair_config("csv"), csv_service, include_test=True)
    second = run_pair(pair_config("csv"), csv_service, include_test=True)
    assert first.config == second.config
    assert export_zip_bytes(first) == export_zip_bytes(second)
    assert export_zip_bytes(first, excel_es=True) == export_zip_bytes(second, excel_es=True)


def test_returned_config_is_complete_and_hashed(csv_service):
    result = run_pair(pair_config("csv"), csv_service)
    cfg = result.config
    assert cfg.pair.a.snapshot_id and cfg.pair.b.snapshot_id
    assert cfg.frozen_split == result.analysis.split
    assert [m.snapshot_id for m in cfg.series] == [cfg.pair.a.snapshot_id, cfg.pair.b.snapshot_id]
    assert cfg.data_origin == DataOrigin.HISTORICAL
    assert cfg.code_version == "prueba-reproducibilidad"
    assert cfg.config_hash == config_hash(cfg)


def test_saved_run_config_replays_from_cache_without_provider(csv_service, tmp_path):
    original = run_pair(pair_config("csv"), csv_service, include_test=True)
    path = save_run_config(original.config, tmp_path / "run_config.json")
    loaded = load_run_config(path)
    assert loaded == original.config

    offline = service(tmp_path / "cache")  # sin proveedores: solo la caché
    replay = run_pair(loaded, offline, include_test=True)
    assert replay.config == original.config
    assert export_zip_bytes(replay) == export_zip_bytes(original)
    assert set(replay.backtests) == {Segment.TRAIN, Segment.VALIDATION, Segment.TEST}


def test_replay_with_missing_snapshot_fails_clearly(csv_service, tmp_path):
    original = run_pair(pair_config("csv"), csv_service)
    path = save_run_config(original.config, tmp_path / "run_config.json")
    empty = service(tmp_path / "otra_cache")
    with pytest.raises(DataError) as info:
        run_pair(load_run_config(path), empty)
    err = info.value
    assert err.code == "SNAPSHOT_NO_ENCONTRADO"
    assert original.config.pair.a.snapshot_id in err.message
    assert "volver a descargarlos" in err.message


def test_edited_run_config_is_rejected(csv_service, tmp_path):
    original = run_pair(pair_config("csv"), csv_service)
    path = save_run_config(original.config, tmp_path / "run_config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["strategy"]["entry_z"] = 1.5  # «ajustar» un umbral a mano
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_run_config(path)
    assert info.value.code == "RUN_CONFIG_HASH_DISTINTO"


def test_snapshot_must_match_the_reference(csv_service, tmp_path):
    original = run_pair(pair_config("csv"), csv_service)
    cfg = original.config
    swapped = cfg.model_copy(
        update={
            "pair": cfg.pair.model_copy(
                update={"a": cfg.pair.a.model_copy(update={"symbol": "ZZZ"})}
            )
        }
    )
    with pytest.raises(ConfigError) as info:
        run_pair(swapped, csv_service)
    assert info.value.code == "RUN_CONFIG_INCONSISTENTE"


def test_reference_without_snapshot_needs_dates(csv_service):
    cfg = pair_config("csv")
    cfg = cfg.model_copy(
        update={
            "pair": cfg.pair.model_copy(update={"a": cfg.pair.a.model_copy(update={"start": None})})
        }
    )
    with pytest.raises(ConfigError) as info:
        run_pair(cfg, csv_service)
    assert info.value.code == "RANGO_FECHAS_FALTANTE"


def test_test_segment_runs_only_when_requested(csv_service):
    without = run_pair(pair_config("csv"), csv_service)
    with_test = run_pair(pair_config("csv"), csv_service, include_test=True)
    assert set(without.backtests) == {Segment.TRAIN, Segment.VALIDATION}
    assert Segment.TEST in with_test.backtests
    # Ejecutar la prueba no cambia formación ni validación.
    for segment in (Segment.TRAIN, Segment.VALIDATION):
        assert without.backtests[segment].trades == with_test.backtests[segment].trades
