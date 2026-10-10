"""Proveedor CSV: archivos con columnas ``date, close`` (opcional open, high, low, volume).

CONTRATO:
- ``parse_csv(content, meta)``: acepta bytes o str (UTF-8, con o sin BOM; si no
  es UTF-8 válido se interpreta como Windows-1252, la codificación por defecto de
  Excel en español). Detecta separador ``,`` ``;`` o tabulador (por el encabezado).
  Con ``;`` (y tabulador) acepta coma decimal (y punto de miles si hay coma).
  Encabezados sin distinguir mayúsculas ni tildes y con espacios recortados; se
  aceptan alias en español: fecha→date, cierre→close, apertura→open,
  maximo/máximo→high, minimo/mínimo→low, volumen→volume; las demás columnas se
  ignoran. Fechas ISO ``AAAA-MM-DD`` (también con hora, que se descarta). NO
  ordena ni elimina duplicados ni rellena: entrega la serie cruda para que
  ``validation`` lo reporte. Celdas vacías de close (o marcas de faltante como
  ``NaN``, ``#N/A``, ``#N/D``) → NaN. Las filas totalmente vacías se ignoran.
- Errores (``DataError``, mensajes en español que digan qué hacer):
  archivo vacío (CSV_VACIO), falta columna obligatoria (CSV_FALTA_COLUMNA, lista
  las columnas encontradas), fechas no interpretables (CSV_FECHA_INVALIDA, con
  ejemplos de filas), close no numérico (CSV_VALOR_NO_NUMERICO, con ejemplos),
  filas con más columnas que el encabezado (CSV_MAL_FORMADO), columna repetida
  (CSV_COLUMNA_DUPLICADA).
- Metadatos obligatorios declarados por el usuario (``CSVMetadata``): moneda
  (código de 3 letras, se guarda en mayúsculas), clase de activo, tipo de
  precio. Sin ellos no se puede validar la moneda ni elegir 252/365. Origen
  SINTETICO ⇔ tipo de precio SINTÉTICO (se fuerzan mutuamente) y el nombre pasa
  a empezar por «SINTÉTICO — ».
- ``CSVProvider(directory)``: cada serie es ``<archivo>.csv`` + ``<archivo>.meta.json``
  (CSVMetadata serializado). El símbolo es el del sidecar (sin distinguir
  mayúsculas). Si ``origin`` es SINTETICO, el archivo se guarda con prefijo
  ``SINTETICO_``. ``register(content, meta)`` valida (parse_csv) y guarda ambos
  archivos (reemplaza un registro previo del mismo símbolo).
  ``get_history`` filtra por [start, end] inclusive (sin datos → ProviderError
  SIMBOLO_SIN_DATOS). ``search`` busca por subcadena (sin distinguir mayúsculas)
  en símbolo o nombre. ``always_reload = True``: el servicio de datos relee el
  archivo en cada carga (puede haberse reemplazado). ``retrieved_at`` = fecha de modificación del archivo
  (UTC) o None si sintético.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import unicodedata
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from mrlab.data.cache import safe_symbol
from mrlab.data.calendar import to_date_index
from mrlab.data.providers.base import check_interval
from mrlab.data.providers.synthetic import SYNTHETIC_NAME_PREFIX, SYNTHETIC_NOTE
from mrlab.data.proxies import build_asset_info
from mrlab.domain.errors import DataError, ProviderError
from mrlab.domain.models import (
    AssetClass,
    AssetInfo,
    DataOrigin,
    PriceSeries,
    PriceType,
    SeriesMeta,
    coverage_of,
)

COLUMN_ALIASES: dict[str, str] = {
    "date": "date",
    "fecha": "date",
    "close": "close",
    "cierre": "close",
    "open": "open",
    "apertura": "open",
    "high": "high",
    "maximo": "high",
    "low": "low",
    "minimo": "low",
    "volume": "volume",
    "volumen": "volume",
}
REQUIRED_COLUMNS = ("date", "close")
VALUE_COLUMNS = ("close", "open", "high", "low", "volume")
MISSING_TOKENS = frozenset({"", "nan", "na", "n/a", "null", "#n/a", "#n/d"})
SYNTHETIC_FILE_PREFIX = "SINTETICO_"
META_SUFFIX = ".meta.json"
MAX_EXAMPLES = 5

_ISO_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[ T].*)?$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")


class CSVMetadata(BaseModel):
    symbol: str
    name: str
    currency: str
    asset_class: AssetClass
    price_type: PriceType = PriceType.CLOSE
    origin: DataOrigin = DataOrigin.HISTORICAL
    notes: tuple[str, ...] = ()

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("indica un símbolo para la serie (p. ej. «KO»)")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return v.strip()

    @field_validator("currency")
    @classmethod
    def _currency(cls, v: str) -> str:
        v = v.strip().upper()
        if not _CURRENCY.match(v):
            raise ValueError("la moneda debe ser un código de 3 letras (p. ej. USD, EUR, MXN)")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> CSVMetadata:
        if self.origin == DataOrigin.SYNTHETIC or self.price_type == PriceType.SYNTHETIC:
            self.origin = DataOrigin.SYNTHETIC
            self.price_type = PriceType.SYNTHETIC
            base = self.name.removeprefix(SYNTHETIC_NAME_PREFIX).strip() or self.symbol
            self.name = f"{SYNTHETIC_NAME_PREFIX}{base}"
        elif not self.name:
            self.name = self.symbol
        return self

    @property
    def is_synthetic(self) -> bool:
        return self.origin == DataOrigin.SYNTHETIC


# --------------------------------------------------------------------------- lectura


def _decode(content: bytes | str) -> str:
    if isinstance(content, str):
        return content.removeprefix("﻿")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def _normalize_header(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def _detect_separator(header_line: str) -> str:
    counts = {sep: header_line.count(sep) for sep in ("\t", ";", ",")}
    best = max(counts, key=counts.__getitem__)
    return best if counts[best] else ","


def _parse_number(cell: str, decimal_comma: bool) -> float:
    s = cell.strip().replace(" ", "").replace(" ", "")
    if s.lower() in MISSING_TOKENS:
        return math.nan
    if decimal_comma and "," in s:
        s = s.replace(".", "").replace(",", ".")  # «1.234,56» → «1234.56»
    value = float(s)
    if math.isinf(value):
        raise ValueError(cell)
    return value


def _parse_date(raw: str) -> date | None:
    """Fecha ISO AAAA-MM-DD; si trae hora («2024-03-15 16:00» o «…T16:00»), se descarta."""
    match = _ISO_DATE.match(raw)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:  # p. ej. 2024-02-30
        return None


def _examples(bad: list[tuple[int, str]]) -> str:
    shown = "; ".join(f"línea {line}: «{value}»" for line, value in bad[:MAX_EXAMPLES])
    rest = len(bad) - MAX_EXAMPLES
    return shown + (f" y {rest} más" if rest > 0 else "")


def _empty_error(detail: str) -> DataError:
    return DataError(
        f"El archivo CSV {detail}. Debe tener una fila de encabezado (por ejemplo «date,close») "
        "y al menos una fila con datos.",
        code="CSV_VACIO",
    )


def parse_csv(content: bytes | str, meta: CSVMetadata, provider_name: str = "csv") -> PriceSeries:
    text = _decode(content)
    if not text.strip():
        raise _empty_error("está vacío")
    first_line = next(line for line in text.splitlines() if line.strip())
    sep = _detect_separator(first_line)
    decimal_comma = sep != ","

    reader = csv.reader(io.StringIO(text), delimiter=sep)
    rows: list[tuple[int, list[str]]] = []
    try:
        for row in reader:
            if any(cell.strip() for cell in row):
                rows.append((reader.line_num, row))
    except csv.Error as exc:
        raise DataError(
            f"No se pudo leer el CSV cerca de la línea {reader.line_num}: el archivo parece dañado "
            "o no es de texto. Vuelve a exportarlo desde la hoja de cálculo.",
            code="CSV_MAL_FORMADO",
            details={"linea": reader.line_num, "error": str(exc)},
        ) from exc
    if not rows:
        raise _empty_error("no tiene contenido")
    _, header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        raise _empty_error("no tiene filas de datos, solo el encabezado")

    found = [h.strip() for h in header]
    positions: dict[str, int] = {}
    for pos, raw_name in enumerate(header):
        canonical = COLUMN_ALIASES.get(_normalize_header(raw_name))
        if canonical is None:
            continue
        if canonical in positions:
            raise DataError(
                f"La columna «{canonical}» aparece dos veces en el encabezado ({', '.join(found)}). "
                "Deja solo una (por ejemplo, «cierre» o «close», no ambas).",
                code="CSV_COLUMNA_DUPLICADA",
                details={"columnas_encontradas": found},
            )
        positions[canonical] = pos
    missing = [c for c in REQUIRED_COLUMNS if c not in positions]
    if missing:
        raise DataError(
            f"Al CSV le falta la columna obligatoria {' y '.join(f'«{c}»' for c in missing)}. "
            f"Columnas encontradas: {', '.join(f'«{c}»' for c in found) or 'ninguna'}. El archivo "
            "necesita al menos «date» (o «fecha») y «close» (o «cierre»); opcionalmente «open», "
            "«high», «low» y «volume». Revisa también el separador (coma, punto y coma o "
            "tabulador).",
            code="CSV_FALTA_COLUMNA",
            details={"faltan": missing, "columnas_encontradas": found, "separador": sep},
        )

    width = len(header)
    dates: list[date] = []
    values: dict[str, list[float]] = {c: [] for c in VALUE_COLUMNS if c in positions}
    bad_dates: list[tuple[int, str]] = []
    bad_values: dict[str, list[tuple[int, str]]] = {c: [] for c in values}
    for line, row in data_rows:
        if len(row) > width and any(cell.strip() for cell in row[width:]):
            raise DataError(
                f"La línea {line} del CSV tiene más columnas ({len(row)}) que el encabezado "
                f"({width}). Revisa el separador y que los decimales no usen el mismo carácter "
                "que el separador (en archivos con coma, los decimales van con punto).",
                code="CSV_MAL_FORMADO",
                details={"linea": line},
            )
        row = row + [""] * (width - len(row))
        raw_date = row[positions["date"]].strip()
        parsed = _parse_date(raw_date)
        if parsed is None:
            bad_dates.append((line, raw_date))
        else:
            dates.append(parsed)
        for col in values:
            cell = row[positions[col]]
            try:
                values[col].append(_parse_number(cell, decimal_comma))
            except ValueError:
                values[col].append(math.nan)
                bad_values[col].append((line, cell.strip()))

    if bad_dates:
        raise DataError(
            f"Hay {len(bad_dates)} fecha(s) que no se pueden interpretar ({_examples(bad_dates)}). "
            "Usa el formato AAAA-MM-DD, por ejemplo 2024-03-15. Si tu archivo usa DD/MM/AAAA, "
            "cambia el formato de la columna en Excel (Formato de celdas → Personalizada → "
            "aaaa-mm-dd) antes de exportar.",
            code="CSV_FECHA_INVALIDA",
            details={"ejemplos": [{"linea": n, "valor": v} for n, v in bad_dates[:MAX_EXAMPLES]]},
        )
    for col, bad in bad_values.items():
        if bad:
            raise DataError(
                f"La columna «{col}» tiene {len(bad)} valor(es) no numérico(s) ({_examples(bad)}). "
                "Usa solo números (con punto decimal, o con coma decimal si el separador es punto "
                "y coma) y deja la celda vacía si falta el dato.",
                code="CSV_VALOR_NO_NUMERICO",
                details={
                    "columna": col,
                    "ejemplos": [{"linea": n, "valor": v} for n, v in bad[:MAX_EXAMPLES]],
                },
            )

    data = pd.DataFrame(
        {c: np.asarray(v, dtype="float64") for c, v in values.items()},
        index=to_date_index(pd.DatetimeIndex(dates)),
    )
    cov_start, cov_end, n_obs = coverage_of(data)
    note = "CSV cargado por el usuario; moneda, clase de activo y tipo de precio declarados por el usuario."
    if meta.is_synthetic:
        note = f"{SYNTHETIC_NOTE} {note}"
    series_meta = SeriesMeta(
        asset=_asset_info(meta, provider_name),
        price_type=meta.price_type,
        coverage_start=cov_start,
        coverage_end=cov_end,
        n_obs=n_obs,
        source_note=note,
    )
    return PriceSeries(meta=series_meta, data=data)


def _asset_info(meta: CSVMetadata, provider_name: str) -> AssetInfo:
    notes = (SYNTHETIC_NOTE, *meta.notes) if meta.is_synthetic else meta.notes
    return build_asset_info(
        meta.symbol,
        meta.asset_class,
        name=meta.name,
        currency=meta.currency,
        provider=provider_name,
        origin=meta.origin,
        notes=notes,
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- proveedor


class CSVProvider:
    name = "csv"
    description = "Archivos CSV cargados por el usuario (columnas date, close)."
    #: El usuario puede volver a subir un archivo con el mismo símbolo: ``DataService``
    #: lo relee siempre en lugar de reutilizar una instantánea anterior.
    always_reload = True

    def __init__(self, directory: Path | str | None = None):
        from mrlab.config import get_settings

        self.directory = Path(directory) if directory else get_settings().uploads_dir

    def register(self, content: bytes | str, meta: CSVMetadata) -> PriceSeries:
        parse_csv(content, meta, self.name)  # valida antes de guardar nada
        self.directory.mkdir(parents=True, exist_ok=True)
        for csv_path, existing in list(self._entries()):
            if existing.symbol.upper() == meta.symbol.upper():
                csv_path.unlink(missing_ok=True)
                self._sidecar(csv_path).unlink(missing_ok=True)
        csv_path = self._target_path(meta)
        payload = content.encode("utf-8") if isinstance(content, str) else content
        _write_atomic(csv_path, payload)
        _write_atomic(
            self._sidecar(csv_path), (meta.model_dump_json(indent=2) + "\n").encode("utf-8")
        )
        return self._load(csv_path, meta)

    def list_symbols(self) -> list[str]:
        return sorted(meta.symbol for _, meta in self._entries())

    def search(self, query: str) -> list[AssetInfo]:
        q = query.strip().lower()
        return [
            _asset_info(meta, self.name)
            for _, meta in sorted(self._entries(), key=lambda e: e[1].symbol)
            if not q or q in meta.symbol.lower() or q in meta.name.lower()
        ]

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d") -> PriceSeries:
        check_interval(interval)
        csv_path, meta = self._find(symbol)
        series = self._load(csv_path, meta)
        idx = series.data.index
        data = series.data[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        if data.empty:
            raise ProviderError(
                f"El CSV de «{meta.symbol}» no tiene filas entre {start.isoformat()} y "
                f"{end.isoformat()} (cubre de {series.meta.coverage_start} a "
                f"{series.meta.coverage_end}).",
                code="SIMBOLO_SIN_DATOS",
                details={"symbol": meta.symbol},
            )
        cov_start, cov_end, n_obs = coverage_of(data)
        return series.with_data(data).with_meta(
            requested_start=start,
            requested_end=end,
            coverage_start=cov_start,
            coverage_end=cov_end,
            n_obs=n_obs,
        )

    def get_metadata(self, symbol: str) -> AssetInfo:
        _, meta = self._find(symbol)
        return _asset_info(meta, self.name)

    # ------------------------------------------------------------------ internos

    @staticmethod
    def _sidecar(csv_path: Path) -> Path:
        return csv_path.with_name(csv_path.stem + META_SUFFIX)

    def _entries(self) -> Iterator[tuple[Path, CSVMetadata]]:
        if not self.directory.is_dir():
            return
        for sidecar in sorted(self.directory.glob(f"*{META_SUFFIX}")):
            csv_path = sidecar.with_name(sidecar.name.removesuffix(META_SUFFIX) + ".csv")
            if not csv_path.exists():
                continue
            try:
                meta = CSVMetadata.model_validate_json(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValidationError):
                continue  # un sidecar dañado no impide usar los demás archivos
            yield csv_path, meta

    def _find(self, symbol: str) -> tuple[Path, CSVMetadata]:
        wanted = (symbol or "").strip().upper()
        for csv_path, meta in self._entries():
            if meta.symbol.upper() == wanted:
                return csv_path, meta
        available = self.list_symbols()
        raise ProviderError(
            f"No hay ningún CSV registrado con el símbolo «{symbol}». "
            f"Disponibles: {', '.join(available) or 'ninguno (carga primero un archivo)'}.",
            code="SIMBOLO_SIN_DATOS",
            details={"symbol": symbol, "disponibles": available},
        )

    def _target_path(self, meta: CSVMetadata) -> Path:
        stem = (SYNTHETIC_FILE_PREFIX if meta.is_synthetic else "") + safe_symbol(meta.symbol)
        sidecar = self.directory / f"{stem}{META_SUFFIX}"
        if sidecar.exists():  # otro símbolo con el mismo nombre seguro (p. ej. «GC=F» y «GC_F»)
            stem += "_" + hashlib.sha256(meta.symbol.encode()).hexdigest()[:6]
        return self.directory / f"{stem}.csv"

    def _load(self, csv_path: Path, meta: CSVMetadata) -> PriceSeries:
        series = parse_csv(csv_path.read_bytes(), meta, self.name)
        retrieved = (
            None if meta.is_synthetic else datetime.fromtimestamp(csv_path.stat().st_mtime, tz=UTC)
        )
        return series.with_meta(retrieved_at=retrieved)
