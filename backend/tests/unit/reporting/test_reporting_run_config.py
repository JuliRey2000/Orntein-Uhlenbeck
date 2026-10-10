"""run_config.json: JSON canónico, hash de la configuración, versión del código, guardar y cargar."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, date, datetime

import pytest

from mrlab.analysis.pair_pipeline import AnalysisConfig
from mrlab.domain.errors import ConfigError
from mrlab.reporting import run_config as rc
from mrlab.reporting.run_config import (
    PairSpec,
    RunConfig,
    SeriesRef,
    canonical_json,
    config_hash,
    git_version,
    load_run_config,
    save_run_config,
)
from mrlab.strategy.params import StrategyParams


def make_config(**updates) -> RunConfig:
    pair = PairSpec(
        a=SeriesRef(provider="csv", symbol="KO", start="2015-01-02", end="2020-12-31"),
        b=SeriesRef(provider="csv", symbol="PEP", start="2015-01-02", end="2020-12-31"),
    )
    return RunConfig(pair=pair, **updates)


# --------------------------------------------------------------------------- JSON canónico


def test_canonical_json_sorts_keys_and_is_compact():
    assert canonical_json({"b": 1, "a": [1, 2], "c": {"z": 0, "y": None}}) == (
        '{"a":[1,2],"b":1,"c":{"y":null,"z":0}}'
    )


def test_canonical_json_keeps_unicode_and_uses_repr_for_floats():
    text = canonical_json({"nombre": "Señal", "x": 0.1, "y": 1e-10, "z": 1 / 3})
    assert "Señal" in text
    assert '"x":0.1' in text and '"y":1e-10' in text and f'"z":{1 / 3!r}' in text


def test_canonical_json_serializes_dates_and_models():
    stamp = datetime(2026, 10, 10, 11, 30, tzinfo=UTC)
    text = canonical_json({"d": date(2020, 1, 31), "t": stamp})
    assert '"d":"2020-01-31"' in text and '"t":"2026-10-10T11:30:00+00:00"' in text
    model = SeriesRef(provider="csv", symbol="KO")
    assert canonical_json(model) == canonical_json(model.model_dump(mode="json"))


def test_canonical_json_rejects_nan():
    with pytest.raises(ConfigError) as info:
        canonical_json({"x": float("nan")})
    assert info.value.code == "RUN_CONFIG_NO_SERIALIZABLE"


# --------------------------------------------------------------------------- hash


def test_config_hash_is_sha256_hex_and_stable():
    h = config_hash(make_config())
    assert len(h) == 64 and int(h, 16) >= 0
    assert config_hash(make_config()) == h


def test_config_hash_ignores_hash_and_code_version():
    base = make_config()
    other = base.model_copy(update={"config_hash": "x" * 64, "code_version": "abc-dirty"})
    assert config_hash(other) == config_hash(base)


def test_config_hash_changes_with_any_parameter():
    base = config_hash(make_config())
    assert config_hash(make_config(strategy=StrategyParams(entry_z=2.5))) != base
    assert config_hash(make_config(analysis=AnalysisConfig(alpha_level=0.01))) != base
    assert config_hash(make_config(seeds={"sintetico": 7})) != base


# --------------------------------------------------------------------------- versión del código


def test_git_version_prefers_environment(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "  abc123  ")
    assert git_version() == "abc123"


def test_git_version_marks_dirty_tree(monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    answers = {"rev-parse": "deadbeef", "status": " M archivo.py"}
    monkeypatch.setattr(rc, "_git", lambda *args: answers[args[0]])
    assert git_version() == "deadbeef-dirty"
    answers["status"] = ""
    assert git_version() == "deadbeef"


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, "git"),
        subprocess.TimeoutExpired("git", 5),
    ],
)
def test_git_version_never_raises(monkeypatch, exc):
    monkeypatch.delenv("GIT_COMMIT", raising=False)

    def broken(*args, **kwargs):
        raise exc

    monkeypatch.setattr(subprocess, "run", broken)
    assert git_version() == "desconocido"


def test_git_version_in_this_repository(monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    version = git_version()
    assert version == "desconocido" or re.fullmatch(r"[0-9a-f]{40,64}(-dirty)?", version)


# --------------------------------------------------------------------------- guardar y cargar


def test_save_writes_readable_sorted_json_with_hash(tmp_path):
    cfg = make_config(code_version="abc")
    path = save_run_config(cfg, tmp_path / "sub" / "run_config.json")
    raw_bytes = path.read_bytes()
    assert raw_bytes.endswith(b"\n") and b"\r\n" not in raw_bytes
    text = raw_bytes.decode("utf-8")
    assert text.startswith('{\n  "analysis": {')  # indentado y con claves ordenadas
    data = json.loads(text)
    assert list(data) == sorted(data)
    assert data["config_hash"] == config_hash(cfg)
    assert data["code_version"] == "abc"


def test_save_and_load_roundtrip(tmp_path):
    cfg = make_config(seeds={"sintetico": 7})
    loaded = load_run_config(save_run_config(cfg, tmp_path / "run_config.json"))
    assert loaded == cfg.model_copy(update={"config_hash": config_hash(cfg)})


def test_load_rejects_edited_file(tmp_path):
    path = save_run_config(make_config(), tmp_path / "run_config.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["costs"]["commission_bps"] = 0.0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_run_config(path)
    assert info.value.code == "RUN_CONFIG_HASH_DISTINTO"
    assert "se modificó" in info.value.message


def test_load_accepts_hand_written_config_without_hash(tmp_path):
    cfg = make_config()
    path = tmp_path / "run_config.json"
    path.write_text(cfg.model_dump_json(exclude={"config_hash"}), encoding="utf-8")
    loaded = load_run_config(path)
    assert loaded.config_hash == config_hash(cfg)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("{no es json", "RUN_CONFIG_INVALIDO"),
        ("[1, 2]", "RUN_CONFIG_INVALIDO"),
        ('{"pair": 3}', "RUN_CONFIG_INVALIDO"),
        ('{"schema_version": 99}', "RUN_CONFIG_VERSION"),
    ],
)
def test_load_reports_invalid_files(tmp_path, content, code):
    path = tmp_path / "run_config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_run_config(path)
    assert info.value.code == code


def test_load_missing_file(tmp_path):
    with pytest.raises(ConfigError) as info:
        load_run_config(tmp_path / "no_existe.json")
    assert info.value.code == "RUN_CONFIG_NO_ENCONTRADO"
