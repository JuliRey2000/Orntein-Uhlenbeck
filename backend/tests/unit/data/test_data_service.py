"""DataService: caché agresiva, refresco explícito y reproducción sin red."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from mrlab.data.cache import SnapshotCache
from mrlab.data.providers.csv_provider import CSVMetadata, CSVProvider
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.data.registry import ProviderRegistry
from mrlab.data.service import DataService
from mrlab.domain.errors import ConfigError, DataError
from mrlab.domain.models import (
    AssetClass,
    AssetInfo,
    PriceSeries,
    PriceType,
    SeriesMeta,
    coverage_of,
)


class CountingProvider:
    """Proveedor falso: precio determinista por fecha; ``version`` simula un reajuste de Yahoo."""

    name = "fake"
    description = "Proveedor de prueba"

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date]] = []
        self.version = 1

    def search(self, query: str) -> list[AssetInfo]:
        return []

    def get_metadata(self, symbol: str) -> AssetInfo:
        return AssetInfo(
            symbol=symbol,
            name=symbol,
            asset_class=AssetClass.EQUITY,
            currency="USD",
            provider="fake",
        )

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d") -> PriceSeries:
        self.calls.append((symbol, start, end))
        idx = pd.bdate_range(start, end, name="date")
        close = 100.0 + np.array([ts.toordinal() % 97 for ts in idx], dtype=float) * self.version
        data = pd.DataFrame({"close": close}, index=idx)
        cs, ce, n = coverage_of(data)
        meta = SeriesMeta(
            asset=self.get_metadata(symbol.upper()),
            price_type=PriceType.ADJUSTED_CLOSE,
            retrieved_at=datetime(2026, 10, len(self.calls), tzinfo=UTC),
            coverage_start=cs,
            coverage_end=ce,
            n_obs=n,
        )
        return PriceSeries(meta=meta, data=data)


class ExplodingProvider(CountingProvider):
    def get_history(self, *args, **kwargs):  # pragma: no cover - si se llama, la prueba falla
        raise AssertionError("load_snapshot no debe llamar al proveedor")


@pytest.fixture
def setup(tmp_path):
    provider = CountingProvider()
    registry = ProviderRegistry(
        {"fake": provider, "synthetic": SyntheticProvider()}, default_name="fake"
    )
    return DataService(registry, SnapshotCache(tmp_path / "cache")), provider


def test_cache_hit_does_not_call_provider(setup):
    service, provider = setup
    first = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31))
    again = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31))
    lower = service.load("fake", "ko", date(2020, 3, 1), date(2020, 6, 30))

    assert len(provider.calls) == 1
    assert again.meta.snapshot_id == first.meta.snapshot_id == lower.meta.snapshot_id
    pd.testing.assert_frame_equal(again.data, first.data)


def test_trim_keeps_snapshot_id_and_updates_coverage(setup):
    service, _ = setup
    full = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31))
    part = service.load("fake", "KO", date(2020, 3, 2), date(2020, 3, 31))

    assert part.meta.snapshot_id == full.meta.snapshot_id
    assert part.meta.coverage_start == date(2020, 3, 2) and part.meta.coverage_end == date(
        2020, 3, 31
    )
    assert part.meta.n_obs == len(part.data) == 22
    assert part.meta.requested_start == date(2020, 3, 2) and part.meta.requested_end == date(
        2020, 3, 31
    )
    pd.testing.assert_frame_equal(part.data, full.data.loc["2020-03-02":"2020-03-31"])


def test_range_not_covered_downloads_again(setup):
    service, provider = setup
    service.load("fake", "KO", date(2020, 1, 1), date(2020, 6, 30))
    wider = service.load("fake", "KO", date(2019, 1, 1), date(2020, 6, 30))
    assert len(provider.calls) == 2
    assert provider.calls[-1][1:] == (date(2019, 1, 1), date(2020, 6, 30))
    assert wider.meta.coverage_start == date(2019, 1, 1)
    # el rango anterior ahora está cubierto por la nueva instantánea (la más reciente)
    service.load("fake", "KO", date(2020, 1, 1), date(2020, 6, 30))
    assert len(provider.calls) == 2


def test_refresh_forces_download_and_new_snapshot(setup):
    service, provider = setup
    old = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31))
    provider.version = 2  # Yahoo reajustó el histórico
    new = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31), refresh=True)

    assert len(provider.calls) == 2
    assert new.meta.snapshot_id != old.meta.snapshot_id
    assert (
        service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31)).meta.snapshot_id
        == new.meta.snapshot_id
    )
    assert len(provider.calls) == 2
    # la instantánea antigua sigue intacta para reproducir análisis previos
    pd.testing.assert_frame_equal(service.load_snapshot(old.meta.snapshot_id).data, old.data)


def test_refresh_with_identical_content_reuses_snapshot(setup):
    service, provider = setup
    old = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31))
    new = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31), refresh=True)
    assert len(provider.calls) == 2
    assert new.meta.snapshot_id == old.meta.snapshot_id
    assert new.meta.retrieved_at == old.meta.retrieved_at


def test_load_snapshot_never_calls_provider(tmp_path):
    cache = SnapshotCache(tmp_path)
    service = DataService(ProviderRegistry({"fake": CountingProvider()}), cache)
    sid = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31)).meta.snapshot_id

    offline = DataService(ProviderRegistry({"fake": ExplodingProvider()}), cache)
    full = offline.load_snapshot(sid)
    part = offline.load_snapshot(sid, date(2020, 2, 3), date(2020, 2, 7))
    assert full.meta.snapshot_id == part.meta.snapshot_id == sid
    assert part.meta.n_obs == 5
    with pytest.raises(DataError) as exc_info:
        offline.load_snapshot("fake_KO_1d_ffffffffffff")
    assert exc_info.value.code == "SNAPSHOT_NO_ENCONTRADO"
    with pytest.raises(DataError) as exc_info:
        offline.load_snapshot(sid, date(2021, 1, 1), date(2021, 1, 31))
    assert exc_info.value.code == "SIN_DATOS_EN_RANGO"


def test_snapshot_plus_requested_range_reproduces_the_load(setup):
    service, _ = setup
    loaded = service.load("fake", "KO", date(2020, 1, 1), date(2020, 12, 31))
    loaded = service.load("fake", "KO", date(2020, 4, 1), date(2020, 4, 30))
    m = loaded.meta
    replay = service.load_snapshot(m.snapshot_id, m.requested_start, m.requested_end)
    pd.testing.assert_frame_equal(replay.data, loaded.data)
    assert replay.meta == loaded.meta


def test_synthetic_goes_through_cache_with_stable_id(tmp_path):
    ids = []
    for folder in ("c1", "c2"):
        service = DataService(
            ProviderRegistry({"synthetic": SyntheticProvider()}), SnapshotCache(tmp_path / folder)
        )
        s = service.load("synthetic", "SINT-COI-A", date(2018, 1, 1), date(2023, 12, 31))
        assert s.is_synthetic and s.meta.snapshot_id
        ids.append(s.meta.snapshot_id)
    assert ids[0] == ids[1]


def test_errors(setup):
    service, provider = setup
    with pytest.raises(ConfigError) as exc_info:
        service.load("tiingo", "KO", date(2020, 1, 1), date(2020, 12, 31))
    assert exc_info.value.code == "PROVEEDOR_DESCONOCIDO"
    with pytest.raises(DataError) as exc_info:
        service.load("fake", "KO", date(2020, 12, 31), date(2020, 1, 1))
    assert exc_info.value.code == "RANGO_FECHAS_INVALIDO"
    assert provider.calls == []


def test_reuploaded_csv_is_not_served_from_a_stale_snapshot(tmp_path):
    csv = CSVProvider(tmp_path / "uploads")
    service = DataService(ProviderRegistry({"csv": csv}), SnapshotCache(tmp_path / "cache"))
    meta = CSVMetadata(symbol="MIO", name="Mi serie", currency="USD", asset_class=AssetClass.EQUITY)
    start, end = date(2020, 1, 1), date(2020, 1, 31)

    csv.register("date,close\n2020-01-02,1\n2020-01-03,2\n", meta)
    v1 = service.load("csv", "MIO", start, end)
    csv.register("date,close\n2020-01-02,1\n2020-01-03,2.5\n", meta)  # archivo corregido
    v2 = service.load("csv", "MIO", start, end)

    assert v2.close.tolist() == [1.0, 2.5]
    assert v2.meta.snapshot_id != v1.meta.snapshot_id
    assert service.load_snapshot(v1.meta.snapshot_id).close.tolist() == [1.0, 2.0]


def test_tampered_snapshot_fails_and_reloading_the_original_csv_repairs_it(tmp_path):
    """Hallazgo de la revisión: la caché no verificaba el contenido de las instantáneas."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    csv = CSVProvider(tmp_path / "uploads")
    service = DataService(ProviderRegistry({"csv": csv}), SnapshotCache(tmp_path / "cache"))
    meta = CSVMetadata(symbol="MIO", name="Mi serie", currency="USD", asset_class=AssetClass.EQUITY)
    start, end = date(2020, 1, 1), date(2020, 1, 31)
    csv.register("date,close\n2020-01-02,1\n2020-01-03,2\n2020-01-06,3\n", meta)
    original = service.load("csv", "MIO", start, end)
    sid = original.meta.snapshot_id

    path = next((tmp_path / "cache").glob(f"*/*/{sid}.parquet"))
    table = pq.read_table(path)
    df = table.to_pandas()
    df.iloc[1:, df.columns.get_loc("close")] *= 1.25  # se altera la instantánea «inmutable»
    altered = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(altered.replace_schema_metadata(table.schema.metadata), path)

    with pytest.raises(DataError) as exc_info:
        service.load_snapshot(sid)  # reproducir un run_config: nunca en silencio
    assert exc_info.value.code == "SNAPSHOT_CORRUPTO"

    repaired = service.load("csv", "MIO", start, end)  # vuelve a leer el CSV original
    assert repaired.meta.snapshot_id == sid
    assert repaired.close.tolist() == [1.0, 2.0, 3.0]
    assert service.load_snapshot(sid).close.tolist() == [1.0, 2.0, 3.0]
