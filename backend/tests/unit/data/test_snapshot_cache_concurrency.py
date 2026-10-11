"""Caché de instantáneas segura entre hilos.

La API atiende peticiones en paralelo y ejecuta trabajos en hilos: varias instancias de
``SnapshotCache`` (una por servicio) escriben a la vez en la misma carpeta. Sin el lock, dos
``put`` simultáneos leen el mismo ``index.json``, añaden cada uno su entrada y el último en
escribir borra la del otro.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from conftest import make_series

from mrlab.data.cache import INDEX_FILE, SnapshotCache
from mrlab.domain.models import PriceSeries

N_THREADS = 8
PER_THREAD = 4
JOIN_TIMEOUT = 60.0
T0 = datetime(2026, 10, 10, 11, 30, tzinfo=UTC)


def series(thread: int, k: int) -> PriceSeries:
    values = 50.0 + thread + np.linspace(0.0, 1.0 + k, 40)
    s = make_series(values, symbol=f"S{thread % 3}", start="2020-01-01", provider="yahoo")
    return s.with_meta(retrieved_at=T0)


def run_parallel(n: int, target: Callable[[int, threading.Barrier], None]) -> None:
    """Lanza ``n`` hilos que arrancan a la vez; propaga la primera excepción de cualquiera."""
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def wrapper(i: int) -> None:
        try:
            target(i, barrier)
        except BaseException as exc:  # noqa: BLE001 - se re-lanza en el hilo principal
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=wrapper, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(JOIN_TIMEOUT)
        assert not t.is_alive(), "un hilo no terminó (¿bloqueo?)"
    if errors:
        raise errors[0]


def index_ids(root: Path) -> set[str]:
    return set(json.loads((root / INDEX_FILE).read_text(encoding="utf-8"))["snapshots"])


@pytest.fixture
def slow_index(monkeypatch):
    """Ensancha la ventana de carrera: cada lectura del índice tarda un poco."""
    original = SnapshotCache._load_index

    def slow(self):
        index = original(self)
        time.sleep(0.002)
        return index

    monkeypatch.setattr(SnapshotCache, "_load_index", slow)


def test_concurrent_puts_keep_every_entry(tmp_path, slow_index):
    ids: list[str] = []
    ids_lock = threading.Lock()

    def work(i: int, barrier: threading.Barrier) -> None:
        cache = SnapshotCache(tmp_path)  # una instancia por hilo, misma carpeta
        barrier.wait()
        for k in range(PER_THREAD):
            sid = cache.put(series(i, k)).meta.snapshot_id
            with ids_lock:
                ids.append(sid)

    run_parallel(N_THREADS, work)

    assert len(ids) == len(set(ids)) == N_THREADS * PER_THREAD
    assert index_ids(tmp_path) == set(ids)  # ninguna entrada perdida
    fresh = SnapshotCache(tmp_path)
    assert {m.snapshot_id for m in fresh.list()} == set(ids)
    for sid in ids:
        assert fresh.get(sid).meta.snapshot_id == sid  # cada Parquet completo y verificable
    assert not list(tmp_path.rglob("*.tmp"))  # sin temporales abandonados


def test_concurrent_put_of_same_series_is_idempotent(tmp_path, slow_index):
    results: list[PriceSeries] = []
    lock = threading.Lock()

    def work(i: int, barrier: threading.Barrier) -> None:
        cache = SnapshotCache(tmp_path)
        s = series(0, 0)
        if i:  # descargas posteriores del mismo contenido: no cambian la instantánea
            s = s.with_meta(retrieved_at=datetime(2026, 10, 11, i, tzinfo=UTC))
        barrier.wait()
        stored = cache.put(s)
        with lock:
            results.append(stored)

    run_parallel(N_THREADS, work)

    sids = {r.meta.snapshot_id for r in results}
    assert len(sids) == 1
    (sid,) = sids
    assert index_ids(tmp_path) == {sid}
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    stored = SnapshotCache(tmp_path).get(sid)
    # Todos devuelven los mismos datos; la primera escritura fija la fecha de descarga.
    assert all(r.meta.retrieved_at == stored.meta.retrieved_at for r in results)
    assert not list(tmp_path.rglob("*.tmp"))


def test_readers_never_see_partial_files(tmp_path):
    cache = SnapshotCache(tmp_path)
    seed = cache.put(series(9, 0)).meta.snapshot_id
    known = [seed]
    known_lock = threading.Lock()
    writers = N_THREADS // 2

    def work(i: int, barrier: threading.Barrier) -> None:
        own = SnapshotCache(tmp_path)
        barrier.wait()
        if i < writers:
            for k in range(PER_THREAD):
                sid = own.put(series(i, k)).meta.snapshot_id
                with known_lock:
                    known.append(sid)
        else:
            for _ in range(3 * PER_THREAD):
                listed = {m.snapshot_id for m in own.list()}
                assert seed in listed
                with known_lock:
                    current = list(known)
                for sid in current:
                    assert own.get(sid).meta.snapshot_id == sid

    run_parallel(N_THREADS, work)
    assert index_ids(tmp_path) == set(known)


def test_index_is_always_written_under_the_lock(tmp_path, monkeypatch):
    cache = SnapshotCache(tmp_path)
    original = SnapshotCache._write_index
    owned: list[bool] = []

    def checked(self, index):
        owned.append(self._lock._is_owned())
        return original(self, index)

    monkeypatch.setattr(SnapshotCache, "_write_index", checked)
    cache.put(series(1, 0))
    cache.put(series(1, 0))  # ya existe: no reescribe
    (tmp_path / INDEX_FILE).unlink()  # índice perdido: se reconstruye al listar
    assert len(cache.list()) == 1
    cache.put(series(2, 0))
    assert owned and all(owned)


def test_lock_is_shared_by_every_spelling_of_the_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    a = SnapshotCache(tmp_path)
    b = SnapshotCache(Path(tmp_path.name))
    c = SnapshotCache(tmp_path / "sub" / "..")
    assert a._lock is b._lock is c._lock
    assert SnapshotCache(tmp_path / "otra")._lock is not a._lock
    # Reentrante: un put puede leer el índice estando dentro del lock.
    with a._lock:
        a.put(series(3, 0))
