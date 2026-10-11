"""Caché local de instantáneas inmutables en Parquet (sección 1 y 15).

Por qué instantáneas inmutables: Yahoo reajusta TODO el histórico con cada
dividendo/split. Mezclar descargas de fechas distintas corrompería la serie, y
reproducir una ejecución exige usar exactamente los mismos precios.

CONTRATO:
- ``content_hash(df)``: sha256 de una representación canónica del DataFrame
  (índice ISO AAAA-MM-DD, columnas en orden fijo close, open, high, low, volume
  si existen, floats con ``repr``). Independiente del orden de inserción de columnas.
- ``snapshot_id`` = ``f"{provider}_{simbolo_seguro}_{interval}_{hash[:12]}"``
  (``simbolo_seguro`` = símbolo con todo lo que no sea [A-Za-z0-9.-] → ``_``,
  p. ej. ``^GSPC`` → ``_GSPC``, ``GC=F`` → ``GC_F``). Direccionado por contenido:
  la misma descarga produce el mismo id (idempotente). El ``hash`` del id es
  ``snapshot_hash``: combina ``content_hash`` con la identidad de la serie
  (activo, tipo de precio, intervalo y rango pedido), para que la misma tabla
  declarada con otra moneda o pedida con otro rango no reutilice metadatos ajenos.
- ``put(series)``: escribe ``root/<provider>/<simbolo_seguro>/<snapshot_id>.parquet``
  con los metadatos (SeriesMeta en JSON) dentro de los metadatos del esquema
  Parquet (clave ``mrlab_meta``), y actualiza ``root/index.json`` de forma
  atómica (archivo temporal + ``os.replace``). Devuelve la serie con
  ``meta.snapshot_id`` asignado y coverage/n_obs actualizados. Si el id ya existe
  y el archivo verifica (ver ``get``), no reescribe datos (conserva el
  ``retrieved_at`` original); si el archivo existente está dañado o alterado, lo
  REESCRIBE con los datos nuevos (que por construcción tienen ese id). Los datos se
  guardan en forma canónica: columnas conocidas en orden fijo y float64, índice
  de ``mrlab.data.calendar.to_date_index``; el orden de filas no se toca (los
  datos crudos pueden traer desorden o duplicados que reporta la validación).
- ``get(snapshot_id)`` → PriceSeries idéntica a la guardada (mismo índice,
  columnas y dtypes float64). Inexistente → DataError(code="SNAPSHOT_NO_ENCONTRADO")
  explicando que re-descargar puede dar precios distintos.
- Verificación al LEER (direccionamiento por contenido real): se recalcula el
  id con el mismo cálculo que ``put`` (``make_snapshot_id`` → ``_id_from``) a
  partir de los datos leídos y de la identidad TAL COMO SE GUARDÓ en el JSON del
  Parquet (así un campo nuevo del modelo no invalida instantáneas antiguas); si
  no coincide con el id pedido (o con el id guardado en los metadatos), o el
  archivo no se puede leer → DataError(code="SNAPSHOT_CORRUPTO") explicando que
  la instantánea fue alterada o dañada y cómo regenerarla. Nunca se usa en
  silencio.
- ``list(provider=None, symbol=None)`` → SeriesMeta ordenados por retrieved_at desc
  (los sintéticos, sin retrieved_at, por fecha de guardado). El símbolo se
  compara sin distinguir mayúsculas.
- ``latest(provider, symbol, interval)`` → el más reciente o None.
- ``index.json`` es un catálogo reconstruible: si falta o está dañado se
  regenera leyendo los metadatos de los Parquet.
- Funciona en Windows/macOS/Linux (pathlib; sin caracteres prohibidos en nombres).
- Concurrencia (la API atiende peticiones en paralelo y ejecuta trabajos en hilos): un lock
  reentrante por carpeta raíz, compartido por todas las instancias del proceso que apuntan a la
  misma carpeta, protege ``put`` completo (comprobación, escritura del Parquet y del índice) y
  toda lectura o reescritura de ``index.json``. Los Parquet y el índice se escriben en un temporal
  único y se publican con ``os.replace`` (atómico): un lector nunca ve un archivo a medias. Dos
  ``put`` simultáneos de la misma serie dan el mismo id y un solo archivo. No protege entre
  procesos distintos (la API corre en un solo proceso).
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mrlab.data.calendar import to_date_index
from mrlab.domain.errors import DataError
from mrlab.domain.models import PriceSeries, SeriesMeta, coverage_of

PRICE_COLUMNS = ("close", "open", "high", "low", "volume")
META_KEY = b"mrlab_meta"
CORRUPT = "SNAPSHOT_CORRUPTO"
_REGENERATE = (
    "Para regenerarla, vuelve a cargar la serie desde su origen: si es un CSV (el archivo "
    "original) o datos sintéticos, al cargarlos de nuevo la instantánea se reescribe con el mismo "
    "identificador; si viene de un proveedor en línea (Yahoo), pide actualizar los datos: una "
    "descarga nueva puede traer precios distintos (otro identificador), y entonces el análisis "
    "original ya no se puede reproducir exactamente."
)
INDEX_FILE = "index.json"
INDEX_VERSION = 1

_UNSAFE = re.compile(r"[^A-Za-z0-9.-]")
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]+$")
# Nombres reservados en Windows (también con extensión): CON, PRN, AUX, NUL, COM1…, LPT1…
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)
}

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock_for(root: Path) -> threading.RLock:
    """Lock reentrante del proceso para una carpeta raíz (misma ruta resuelta → mismo lock)."""
    key = os.path.normcase(str(root.resolve()))
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.RLock()
        return lock


def safe_symbol(symbol: str) -> str:
    """Símbolo apto para nombres de archivo en cualquier sistema: ``^GSPC`` → ``_GSPC``."""
    safe = _UNSAFE.sub("_", symbol)
    # Windows no admite nombres terminados en punto ni nombres de dispositivo (PRN es un ETF real).
    safe = safe.rstrip(".") or "_"
    if safe.split(".")[0].upper() in _WINDOWS_RESERVED:
        safe += "_"
    return safe


def canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas conocidas en orden fijo, float64, e índice de fechas canónico."""
    if "close" not in df.columns:
        raise DataError(
            "La serie no tiene columna de precio de cierre («close»): no se puede guardar.",
            code="FALTA_CIERRE",
            details={"columnas": [str(c) for c in df.columns]},
        )
    cols = [c for c in PRICE_COLUMNS if c in df.columns]
    out = pd.DataFrame(
        {c: df[c].to_numpy(dtype="float64") for c in cols},
        index=to_date_index(df.index),
    )
    return out


def content_hash(df: pd.DataFrame) -> str:
    """sha256 de la representación canónica (fechas ISO, floats con ``repr``)."""
    cols = [c for c in PRICE_COLUMNS if c in df.columns]
    dates = to_date_index(df.index).strftime("%Y-%m-%d")
    columns = [[repr(float(x)) for x in df[c].tolist()] for c in cols]
    h = hashlib.sha256()
    h.update(("date," + ",".join(cols) + "\n").encode())
    for i, d in enumerate(dates):
        h.update((d + "," + ",".join(col[i] for col in columns) + "\n").encode())
    return h.hexdigest()


def _identity_of(meta_json: dict[str, Any]) -> dict[str, Any]:
    """Identidad de la serie a partir de ``SeriesMeta`` en forma JSON."""
    return {
        "asset": meta_json["asset"],
        "price_type": meta_json["price_type"],
        "interval": meta_json.get("interval", "1d"),
        "requested_start": meta_json.get("requested_start"),
        "requested_end": meta_json.get("requested_end"),
    }


def _hash_identity(identity: dict[str, Any], df: pd.DataFrame) -> str:
    payload = json.dumps(
        {**identity, "content": content_hash(df)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _id_from(identity: dict[str, Any], df: pd.DataFrame) -> str:
    asset = identity["asset"]
    return (
        f"{safe_symbol(asset['provider'])}_{safe_symbol(asset['symbol'])}_"
        f"{safe_symbol(identity['interval'])}_{_hash_identity(identity, df)[:12]}"
    )


def snapshot_hash(meta: SeriesMeta, df: pd.DataFrame) -> str:
    """Hash de la instantánea: contenido + identidad (no incluye la fecha de descarga)."""
    return _hash_identity(_identity_of(meta.model_dump(mode="json")), df)


def make_snapshot_id(meta: SeriesMeta, df: pd.DataFrame) -> str:
    return _id_from(_identity_of(meta.model_dump(mode="json")), df)


def _not_found(snapshot_id: str, root: Path) -> DataError:
    return DataError(
        f"No se encontró la instantánea de datos «{snapshot_id}» en la caché local ({root}). "
        "Para reproducir exactamente un análisis hacen falta los mismos precios: volver a "
        "descargarlos podría dar valores distintos, porque Yahoo reajusta todo el histórico con "
        "cada dividendo o split. Si aceptas usar datos nuevos, vuelve a cargar la serie desde el "
        "proveedor (el resultado podrá diferir del original).",
        code="SNAPSHOT_NO_ENCONTRADO",
        details={"snapshot_id": snapshot_id, "cache": str(root)},
    )


def _sort_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    # ISO 8601 en UTC ordena bien como texto; "" (sin fecha) queda al final en orden desc.
    return (entry["meta"].get("retrieved_at") or "", entry.get("stored_at") or "", entry["id"])


class SnapshotCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        # Se resuelve una vez: un cambio posterior del directorio de trabajo no cambia el lock.
        self._lock = _lock_for(self.root)

    # ------------------------------------------------------------------ API

    def put(self, series: PriceSeries) -> PriceSeries:
        data = canonical_frame(series.data)
        start, end, n_obs = coverage_of(data)
        meta = series.meta.model_copy(
            update={
                "coverage_start": start,
                "coverage_end": end,
                "n_obs": n_obs,
                "snapshot_id": None,
            }
        )
        sid = make_snapshot_id(meta, data)
        meta = meta.model_copy(update={"snapshot_id": sid})
        path = self._path_for(meta)
        with self._lock:
            index = self._load_index()
            if path.exists():
                try:
                    stored = self._read(path, sid)
                except DataError as exc:
                    if exc.code != CORRUPT:
                        raise
                    stored = None  # dañada o alterada: se reescribe con los datos nuevos
                if stored is not None:
                    if sid not in index:
                        index[sid] = self._entry(stored.meta, path)
                        self._write_index(index)
                    return stored
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(data, preserve_index=True)
            table = table.replace_schema_metadata(
                {**(table.schema.metadata or {}), META_KEY: meta.model_dump_json().encode()}
            )
            tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                pq.write_table(table, tmp)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
            index[sid] = self._entry(meta, path)
            self._write_index(index)
        return PriceSeries(meta=meta, data=data)

    def get(self, snapshot_id: str) -> PriceSeries:
        path = self._locate(snapshot_id)
        if path is None:
            raise _not_found(snapshot_id, self.root)
        return self._read(path, snapshot_id)

    def has(self, snapshot_id: str) -> bool:
        return self._locate(snapshot_id) is not None

    def list(
        self, provider: str | None = None, symbol: str | None = None
    ) -> builtins.list[SeriesMeta]:
        with self._lock:
            index = self._load_index()
        wanted = symbol.strip().upper() if symbol else None
        entries = [
            e
            for e in index.values()
            if (provider is None or e["meta"]["asset"]["provider"] == provider)
            and (wanted is None or e["meta"]["asset"]["symbol"].upper() == wanted)
        ]
        entries.sort(key=_sort_key, reverse=True)
        return [SeriesMeta.model_validate(e["meta"]) for e in entries]

    def latest(self, provider: str, symbol: str, interval: str = "1d") -> SeriesMeta | None:
        return next((m for m in self.list(provider, symbol) if m.interval == interval), None)

    # ------------------------------------------------------------------ internos

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILE

    def _path_for(self, meta: SeriesMeta) -> Path:
        assert meta.snapshot_id is not None
        return (
            self.root
            / safe_symbol(meta.provider)
            / safe_symbol(meta.symbol)
            / f"{meta.snapshot_id}.parquet"
        )

    def _entry(self, meta: SeriesMeta, path: Path) -> dict[str, Any]:
        return {
            "id": meta.snapshot_id,
            "path": path.relative_to(self.root).as_posix(),
            "stored_at": datetime.now(UTC).isoformat(),
            "meta": meta.model_dump(mode="json"),
        }

    def _locate(self, snapshot_id: str) -> Path | None:
        if not _VALID_ID.match(snapshot_id or ""):
            return None  # nunca construir rutas con ids arbitrarios (p. ej. «../»)
        with self._lock:
            entry = self._load_index().get(snapshot_id)
        if entry is not None:
            path = self.root / entry["path"]
            if path.exists():
                return path
        # Respaldo: el índice puede no estar al día; el archivo manda.
        return next(self.root.glob(f"*/*/{snapshot_id}.parquet"), None)

    def _read(self, path: Path, snapshot_id: str) -> PriceSeries:
        try:
            table = pq.read_table(path)
            raw_meta = (table.schema.metadata or {})[META_KEY]
            meta = SeriesMeta.model_validate_json(raw_meta)
            df = table.to_pandas()
            df.index = to_date_index(df.index)
            data = canonical_frame(df)
            # Mismo cálculo que ``put`` sobre la identidad TAL COMO SE GUARDÓ (el JSON
            # del Parquet): un campo nuevo del modelo, con su valor por defecto, no
            # invalida las instantáneas escritas antes de existir.
            actual = _id_from(_identity_of(json.loads(raw_meta)), data)
        except Exception as exc:  # archivo truncado, editado o de otra versión
            raise DataError(
                f"La instantánea «{snapshot_id}» de la caché está dañada y no se puede leer "
                f"({path}). {_REGENERATE}",
                code=CORRUPT,
                details={"snapshot_id": snapshot_id, "path": str(path), "error": str(exc)[:300]},
            ) from exc
        if actual != snapshot_id or meta.snapshot_id != snapshot_id:
            raise DataError(
                f"La instantánea «{snapshot_id}» de la caché fue alterada: su contenido ya no "
                "corresponde al identificador con que se guardó (al recalcularlo se obtiene "
                f"«{actual}»). No se usa, porque daría resultados distintos bajo el mismo "
                f"identificador. Archivo: {path}. {_REGENERATE}",
                code=CORRUPT,
                details={
                    "snapshot_id": snapshot_id,
                    "id_recalculado": actual,
                    "id_en_metadatos": meta.snapshot_id,
                    "path": str(path),
                },
            )
        return PriceSeries(meta=meta, data=data)

    def _load_index(self) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            snapshots = raw["snapshots"]
            if not isinstance(snapshots, dict):
                raise ValueError("snapshots no es un objeto")
            return snapshots
        except FileNotFoundError:
            if not self.root.exists():
                return {}
            return self._rebuild_index()
        except (ValueError, KeyError, TypeError):
            return self._rebuild_index()

    def _rebuild_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for path in sorted(self.root.glob("*/*/*.parquet")):
            try:
                meta = SeriesMeta.model_validate_json(pq.read_schema(path).metadata[META_KEY])
            except Exception:
                continue  # un archivo ilegible no debe impedir usar el resto de la caché
            if meta.snapshot_id:
                index[meta.snapshot_id] = self._entry(meta, path)
        if index:
            self._write_index(index)
        return index

    def _write_index(self, index: dict[str, dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": INDEX_VERSION, "snapshots": index},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        tmp = self.index_path.with_name(f".{INDEX_FILE}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(payload + "\n")
            os.replace(tmp, self.index_path)
        finally:
            tmp.unlink(missing_ok=True)
