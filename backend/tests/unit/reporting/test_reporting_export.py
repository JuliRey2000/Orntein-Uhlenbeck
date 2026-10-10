"""Formato determinista de la exportación: celdas, CSV, opción Excel en español y ZIP."""

from __future__ import annotations

import io
import math
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mrlab.analysis.runner import run_pair
from mrlab.data.cache import SnapshotCache
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.data.registry import ProviderRegistry
from mrlab.data.service import DataService
from mrlab.domain.models import Segment
from mrlab.reporting import export
from mrlab.reporting.export import (
    DAILY_HEADERS,
    STATS_COLUMNS,
    TRADE_COLUMNS,
    UTF8_BOM,
    daily_csv,
    export_zip_bytes,
    stats_csv,
    trades_csv,
)
from mrlab.reporting.run_config import PairSpec, RunConfig, SeriesRef


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    cache = SnapshotCache(tmp_path_factory.mktemp("cache"))
    data = DataService(ProviderRegistry({"synthetic": SyntheticProvider()}), cache)
    pair = PairSpec(
        a=SeriesRef(
            provider="synthetic", symbol="SINT-COI-A", start="2018-01-01", end="2023-12-31"
        ),
        b=SeriesRef(
            provider="synthetic", symbol="SINT-COI-B", start="2018-01-01", end="2023-12-31"
        ),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_COMMIT", "prueba")
        return run_pair(RunConfig(pair=pair), data)


# --------------------------------------------------------------------------- celdas


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (math.nan, ""),
        (math.inf, ""),
        (np.float64(np.nan), ""),
        (0.1, "0.1"),
        (1 / 3, "0.333333333333"),
        (-0.0, "0"),
        (10000.0, "10000"),
        (1.5e-7, "1.5e-07"),
        (np.float64(2.5), "2.5"),
        (3, "3"),
        (np.int64(4), "4"),
        (True, "sí"),
        (np.bool_(False), "no"),
        (date(2020, 1, 31), "2020-01-31"),
        (pd.Timestamp("2020-01-31"), "2020-01-31"),
        (Segment.TRAIN, "formacion"),
        (["a", 1.5], "a | 1.5"),
        ("texto, con coma", "texto, con coma"),
    ],
)
def test_cell_format(value, expected):
    assert export._cell(value, excel_es=False) == expected


def test_cell_format_excel_uses_decimal_comma():
    assert export._cell(1234.5, excel_es=True) == "1234,5"
    assert export._cell(2, excel_es=True) == "2"


# --------------------------------------------------------------------------- CSV


def test_standard_csv_is_comma_lf_without_bom(result):
    for text in (trades_csv(result), daily_csv(result), stats_csv(result)):
        assert not text.startswith(UTF8_BOM)
        assert "\r" not in text and text.endswith("\n")
    assert trades_csv(result).splitlines()[0] == ",".join(TRADE_COLUMNS)
    header = daily_csv(result).splitlines()[0].split(",")
    assert header == ["tipo_datos", "segmento", *DAILY_HEADERS.values()]
    assert stats_csv(result).splitlines()[0] == ",".join(STATS_COLUMNS)


def test_excel_csv_uses_semicolon_decimal_comma_and_bom(result):
    text = daily_csv(result, excel_es=True)
    assert text.startswith(UTF8_BOM)
    lines = text.removeprefix(UTF8_BOM).splitlines()
    n_cols = len(lines[0].split(";"))
    assert n_cols == len(DAILY_HEADERS) + 2
    row = lines[1].split(";")
    assert len(row) == n_cols
    numeric = [cell for cell in row[3:] if cell and cell[0].isdigit()]
    assert numeric and not any("." in cell for cell in numeric)
    assert any("," in cell for line in lines[1:50] for cell in line.split(";")[3:])


def test_daily_rows_follow_segment_order(result):
    rows = daily_csv(result).splitlines()[1:]
    segments = [r.split(",")[1] for r in rows]
    n_train = len(result.backtests[Segment.TRAIN].daily)
    assert segments[:n_train] == ["formacion"] * n_train
    assert set(segments[n_train:]) == {"validacion"}


def test_zip_is_byte_stable_and_platform_independent(result):
    data = export_zip_bytes(result)
    assert export_zip_bytes(result) == data
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.external_attr == 0o644 << 16
