"""Caché de instantáneas inmutables en Parquet."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from conftest import make_series

from mrlab.data.cache import SnapshotCache, content_hash, safe_symbol
from mrlab.domain.errors import DataError
from mrlab.domain.models import PriceSeries

T1 = datetime(2026, 10, 9, 8, 0, tzinfo=UTC)
T2 = datetime(2026, 10, 10, 11, 30, tzinfo=UTC)


def series(symbol: str = "KO", values=(1.0, 2.0, 3.0), retrieved_at=T1, **kwargs) -> PriceSeries:
    s = make_series(list(values), symbol=symbol, start="2020-01-01", provider="yahoo", **kwargs)
    return s.with_meta(retrieved_at=retrieved_at)


def test_put_get_roundtrip_is_identical(tmp_path):
    cache = SnapshotCache(tmp_path)
    original = series(values=np.linspace(50.0, 60.0, 300))
    original = original.with_data(
        original.data.assign(volume=np.arange(300), open=original.data["close"] - 0.5)
    )

    stored = cache.put(original)
    loaded = cache.get(stored.meta.snapshot_id)

    pd.testing.assert_frame_equal(loaded.data, stored.data)
    assert loaded.meta == stored.meta
    assert list(loaded.data.columns) == ["close", "open", "volume"]
    assert (loaded.data.dtypes == "float64").all()
    pd.testing.assert_frame_equal(
        loaded.data, original.data[["close", "open", "volume"]].astype("float64"), check_freq=False
    )
    assert stored.meta.n_obs == 300
    assert stored.meta.coverage_start == original.data.index[0].date()
    assert stored.meta.retrieved_at == T1


def test_raw_rows_keep_order_duplicates_and_nan(tmp_path):
    cache = SnapshotCache(tmp_path)
    raw = make_series(
        [3.0, np.nan, 1.0, 1.5], dates=["2020-01-06", "2020-01-02", "2020-01-02", "2020-01-03"]
    )
    loaded = cache.get(cache.put(raw).meta.snapshot_id)
    assert [ts.date().isoformat() for ts in loaded.data.index] == [
        "2020-01-06",
        "2020-01-02",
        "2020-01-02",
        "2020-01-03",
    ]
    np.testing.assert_array_equal(loaded.close.to_numpy(), [3.0, np.nan, 1.0, 1.5])


def test_snapshot_id_is_content_addressed_and_idempotent(tmp_path):
    cache = SnapshotCache(tmp_path)
    first = cache.put(series())
    sid = first.meta.snapshot_id
    assert re.fullmatch(r"yahoo_KO_1d_[0-9a-f]{12}", sid)
    parquet = next(tmp_path.glob("yahoo/KO/*.parquet"))
    mtime = parquet.stat().st_mtime_ns

    again = cache.put(series(retrieved_at=T2))  # misma descarga, otro momento
    assert again.meta.snapshot_id == sid
    assert again.meta.retrieved_at == T1  # se conserva el original
    assert len(list(tmp_path.glob("yahoo/KO/*.parquet"))) == 1
    assert parquet.stat().st_mtime_ns == mtime  # no se reescribe

    changed = cache.put(series(values=(1.0, 2.0, 3.0000001)))
    assert changed.meta.snapshot_id != sid
    assert len(cache.list("yahoo", "KO")) == 2


def test_same_table_with_other_identity_gets_other_id(tmp_path):
    cache = SnapshotCache(tmp_path)
    usd = cache.put(series(currency="USD"))
    eur = cache.put(series(currency="EUR"))
    assert usd.meta.snapshot_id != eur.meta.snapshot_id
    assert cache.get(eur.meta.snapshot_id).meta.asset.currency == "EUR"


def test_content_hash_ignores_column_insertion_order():
    idx = pd.bdate_range("2020-01-01", periods=3, name="date")
    a = pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10.0, 20.0, 30.0]}, index=idx)
    b = pd.DataFrame({"volume": [10.0, 20.0, 30.0], "close": [1.0, 2.0, 3.0]}, index=idx)
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(a.assign(close=[1.0, 2.0, 3.5]))
    assert re.fullmatch(r"[0-9a-f]{64}", content_hash(a))


def test_index_json_is_written(tmp_path):
    cache = SnapshotCache(tmp_path)
    sid = cache.put(series()).meta.snapshot_id
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    entry = index["snapshots"][sid]
    assert entry["path"] == f"yahoo/KO/{sid}.parquet"
    assert entry["meta"]["asset"]["symbol"] == "KO"
    assert entry["meta"]["snapshot_id"] == sid
    assert not list(tmp_path.rglob("*.tmp"))  # escritura atómica: sin restos


def test_list_and_latest(tmp_path):
    cache = SnapshotCache(tmp_path)
    old = cache.put(series(values=(1.0, 2.0), retrieved_at=T1))
    new = cache.put(series(values=(1.0, 2.5), retrieved_at=T2))
    cache.put(series(symbol="PEP", retrieved_at=T2))

    assert [m.snapshot_id for m in cache.list("yahoo", "KO")] == [
        new.meta.snapshot_id,
        old.meta.snapshot_id,
    ]
    assert [m.snapshot_id for m in cache.list(symbol="ko")] == [
        new.meta.snapshot_id,
        old.meta.snapshot_id,
    ]
    assert len(cache.list()) == 3
    assert cache.list(provider="csv") == []
    assert cache.latest("yahoo", "KO").snapshot_id == new.meta.snapshot_id
    assert cache.latest("yahoo", "KO", interval="1wk") is None
    assert cache.latest("yahoo", "MSFT") is None
    assert SnapshotCache(tmp_path / "vacia").list() == []


def test_missing_snapshot_raises_spanish_error(tmp_path):
    cache = SnapshotCache(tmp_path)
    for bad in ("yahoo_KO_1d_000000000000", "../../etc/passwd", ""):
        assert not cache.has(bad)
        with pytest.raises(DataError) as exc_info:
            cache.get(bad)
        err = exc_info.value
        assert err.code == "SNAPSHOT_NO_ENCONTRADO"
        assert "descargar" in err.message and "distintos" in err.message


@pytest.mark.parametrize(
    ("symbol", "safe"),
    [
        ("^GSPC", "_GSPC"),
        ("GC=F", "GC_F"),
        ("BTC-USD", "BTC-USD"),
        ("BRK.B", "BRK.B"),
        ("PRN", "PRN_"),
        ("a/b:c", "a_b_c"),
    ],
)
def test_safe_symbol(symbol, safe):
    assert safe_symbol(symbol) == safe


@pytest.mark.parametrize("symbol", ["^GSPC", "GC=F"])
def test_unsafe_symbols_are_stored_with_safe_names(tmp_path, symbol):
    cache = SnapshotCache(tmp_path)
    stored = cache.put(series(symbol=symbol))
    sid = stored.meta.snapshot_id
    assert sid.startswith(f"yahoo_{safe_symbol(symbol)}_1d_")
    assert re.fullmatch(r"[A-Za-z0-9._-]+", sid)
    path = tmp_path / "yahoo" / safe_symbol(symbol) / f"{sid}.parquet"
    assert path.exists()
    assert cache.get(sid).meta.asset.symbol == symbol
    assert cache.latest("yahoo", symbol).snapshot_id == sid


def test_index_is_rebuilt_when_missing_or_corrupt(tmp_path):
    cache = SnapshotCache(tmp_path)
    sid = cache.put(series()).meta.snapshot_id

    (tmp_path / "index.json").unlink()
    assert [m.snapshot_id for m in SnapshotCache(tmp_path).list()] == [sid]
    assert (tmp_path / "index.json").exists()

    (tmp_path / "index.json").write_text("{ esto no es json", encoding="utf-8")
    assert SnapshotCache(tmp_path).latest("yahoo", "KO").snapshot_id == sid
    assert SnapshotCache(tmp_path).get(sid).meta.snapshot_id == sid


def test_get_falls_back_to_file_when_index_is_stale(tmp_path):
    cache = SnapshotCache(tmp_path)
    sid = cache.put(series()).meta.snapshot_id
    (tmp_path / "index.json").write_text(
        json.dumps({"version": 1, "snapshots": {}}), encoding="utf-8"
    )
    assert cache.has(sid)
    assert cache.get(sid).meta.snapshot_id == sid


def test_corrupt_parquet_raises_clear_error(tmp_path):
    cache = SnapshotCache(tmp_path)
    sid = cache.put(series()).meta.snapshot_id
    next(tmp_path.glob("yahoo/KO/*.parquet")).write_bytes(b"basura")
    with pytest.raises(DataError) as exc_info:
        cache.get(sid)
    assert exc_info.value.code == "SNAPSHOT_CORRUPTO"


def test_synthetic_series_without_retrieved_at(tmp_path):
    from mrlab.data.providers.synthetic import SyntheticPairParams, generate_pair

    a, _ = generate_pair(SyntheticPairParams(n_obs=200))
    cache = SnapshotCache(tmp_path)
    stored = cache.put(a)
    assert stored.meta.retrieved_at is None and stored.is_synthetic
    assert SnapshotCache(tmp_path / "otra").put(a).meta.snapshot_id == stored.meta.snapshot_id
    assert cache.latest("synthetic", a.symbol).snapshot_id == stored.meta.snapshot_id
    assert cache.get(stored.meta.snapshot_id).meta.coverage_end == date(2018, 10, 5)


# --------------------------------------------------------------------------- verificación
# Hallazgo de la revisión: una instantánea alterada se usaba en silencio.


def _rewrite_parquet(path, change_data=None, change_meta=None):
    """Altera un Parquet de la caché conservando (o cambiando) sus metadatos."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    df = table.to_pandas()
    if change_data is not None:
        df = change_data(df)
    if change_meta is not None:
        meta = json.loads(metadata[b"mrlab_meta"])
        change_meta(meta)
        metadata[b"mrlab_meta"] = json.dumps(meta).encode()
    new = pa.Table.from_pandas(df, preserve_index=True).replace_schema_metadata(metadata)
    pq.write_table(new, path)


def _tamper_close(df):
    df = df.copy()
    df.iloc[1:, df.columns.get_loc("close")] *= 1.25
    return df


def test_tampered_snapshot_is_rejected_when_read(tmp_path):
    cache = SnapshotCache(tmp_path)
    sid = cache.put(series()).meta.snapshot_id
    _rewrite_parquet(next(tmp_path.glob("yahoo/KO/*.parquet")), change_data=_tamper_close)
    with pytest.raises(DataError) as exc_info:
        cache.get(sid)
    err = exc_info.value
    assert err.code == "SNAPSHOT_CORRUPTO"
    assert "alterada" in err.message and "vuelve a cargar" in err.message
    assert err.details["id_recalculado"] != sid


def test_tampered_metadata_is_rejected_when_read(tmp_path):
    cache = SnapshotCache(tmp_path)
    sid = cache.put(series()).meta.snapshot_id

    def other_currency(meta):
        meta["asset"]["currency"] = "EUR"

    _rewrite_parquet(next(tmp_path.glob("yahoo/KO/*.parquet")), change_meta=other_currency)
    with pytest.raises(DataError) as exc_info:
        cache.get(sid)
    assert exc_info.value.code == "SNAPSHOT_CORRUPTO"


def test_snapshot_copied_under_another_id_is_rejected(tmp_path):
    cache = SnapshotCache(tmp_path)
    sid_a = cache.put(series(values=(1.0, 2.0, 3.0))).meta.snapshot_id
    sid_b = cache.put(series(values=(4.0, 5.0, 6.0))).meta.snapshot_id
    folder = tmp_path / "yahoo" / "KO"
    (folder / f"{sid_b}.parquet").write_bytes((folder / f"{sid_a}.parquet").read_bytes())
    with pytest.raises(DataError) as exc_info:
        cache.get(sid_b)
    assert exc_info.value.code == "SNAPSHOT_CORRUPTO"
    assert cache.get(sid_a).meta.snapshot_id == sid_a


def test_put_rewrites_a_tampered_or_damaged_snapshot(tmp_path):
    cache = SnapshotCache(tmp_path)
    original = series(values=np.linspace(10.0, 20.0, 50))
    stored = cache.put(original)
    path = next(tmp_path.glob("yahoo/KO/*.parquet"))

    _rewrite_parquet(path, change_data=_tamper_close)
    repaired = cache.put(original)  # volver a cargar los datos originales la repara
    assert repaired.meta.snapshot_id == stored.meta.snapshot_id
    pd.testing.assert_frame_equal(cache.get(stored.meta.snapshot_id).data, stored.data)

    path.write_bytes(b"basura")  # también un archivo ilegible
    cache.put(original)
    pd.testing.assert_frame_equal(cache.get(stored.meta.snapshot_id).data, stored.data)


def test_snapshot_written_before_a_new_model_field_still_verifies(tmp_path):
    """La verificación usa la identidad guardada: un campo nuevo no invalida la caché."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mrlab.data.cache import _id_from, _identity_of, canonical_frame

    data = canonical_frame(series().data)
    old_meta = json.loads(series().meta.model_dump_json())
    del old_meta["asset"]["timezone"]  # como si el modelo aún no tuviera ese campo
    del old_meta["seed"]
    sid = _id_from(_identity_of(old_meta), data)
    old_meta["snapshot_id"] = sid
    path = tmp_path / "yahoo" / "KO" / f"{sid}.parquet"
    path.parent.mkdir(parents=True)
    table = pa.Table.from_pandas(data, preserve_index=True)
    metadata = {**table.schema.metadata, b"mrlab_meta": json.dumps(old_meta).encode()}
    pq.write_table(table.replace_schema_metadata(metadata), path)

    loaded = SnapshotCache(tmp_path).get(sid)
    assert loaded.meta.snapshot_id == sid and loaded.meta.asset.timezone is None
