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
- Columna opcional ``tipo_datos`` (también «tipo de datos»), como la que
  escriben las exportaciones del laboratorio: valores HISTORICO o SINTETICO sin
  distinguir mayúsculas ni tildes; las celdas vacías no informan. Si alguna fila
  es SINTETICO, la serie es SINTÉTICA (origen SINTETICO, tipo de precio
  SINTÉTICO) aunque los metadatos declaren otra cosa, y se añade una nota en
  ``AssetInfo.notes``. Filas HISTORICO y SINTETICO mezcladas
  (CSV_TIPO_DATOS_MEZCLADO) o valores desconocidos (CSV_TIPO_DATOS_DESCONOCIDO)
  → DataError. Nunca sale una serie sintética marcada como histórica.
- Metadatos obligatorios declarados por el usuario (``CSVMetadata``): moneda
  (código ISO de 3 letras, se guarda en mayúsculas), clase de activo, tipo de
  precio. Sin ellos no se puede validar la moneda ni elegir 252/365. Origen
  SINTETICO ⇔ tipo de precio SINTÉTICO (se fuerzan mutuamente; una semilla
  ``seed`` también implica datos sintéticos) y el nombre pasa a empezar por
  «SINTÉTICO — ». ``seed`` se copia a ``SeriesMeta.seed``. Los valores de
  clase de activo, tipo de precio y origen se aceptan sin distinguir
  mayúsculas ni tildes («Acción» → accion).
- ``parse_metadata(data)`` (dict, o JSON en str/bytes, p. ej. un sidecar) y
  ``build_metadata(**campos)``: construyen ``CSVMetadata`` y convierten cualquier
  error (JSON inválido, campo que falta, valor no permitido) en
  DataError(code="CSV_METADATOS_INVALIDOS") en español: qué campo falla, el
  valor recibido y los valores válidos.
- ``origin_from_filename(nombre)`` → SINTETICO si el nombre del archivo (sin
  carpetas) empieza por ``SINTETICO_`` (sin distinguir mayúsculas ni tildes);
  si no, None: el nombre nunca prueba que los datos sean históricos.
- ``CSVProvider(directory)``: cada serie es ``<archivo>.csv`` + ``<archivo>.meta.json``
  (CSVMetadata serializado, leído con ``parse_metadata``). El símbolo es el del
  sidecar (sin distinguir mayúsculas). Si ``origin`` es SINTETICO, el archivo se
  guarda con prefijo ``SINTETICO_``. ``register(content, meta, filename=None)``
  valida (parse_csv) y guarda ambos archivos con los metadatos EFECTIVOS
  (reemplaza un registro previo del mismo símbolo); si ``filename`` (el nombre
  original del archivo) empieza por ``SINTETICO_`` se fuerza el origen
  SINTETICO con una nota. Al leer, un archivo guardado con prefijo
  ``SINTETICO_`` también fuerza el origen SINTETICO.
  ``get_history`` filtra por [start, end] inclusive (sin datos → ProviderError
  SIMBOLO_SIN_DATOS; si hay sidecars dañados, el mensaje los nombra).
  ``search`` busca por subcadena (sin distinguir mayúsculas) en símbolo o
  nombre. ``get_metadata`` relee el archivo (aplica la columna ``tipo_datos``).
  ``always_reload = True``: el servicio de datos relee el archivo en cada carga
  (puede haberse reemplazado). ``retrieved_at`` = fecha de modificación del
  archivo (UTC) o None si sintético.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import unicodedata
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Any

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
#: Columna opcional que marca el origen de cada fila (la escriben las exportaciones).
DATA_TYPE_COLUMN = "tipo_datos"
DATA_TYPE_HEADERS = frozenset({"tipo_datos", "tipo de datos", "tipo_de_datos", "tipo datos"})
SYNTHETIC_FILE_PREFIX = "SINTETICO_"
META_SUFFIX = ".meta.json"
MAX_EXAMPLES = 5
METADATA_ERROR = "CSV_METADATOS_INVALIDOS"

_ISO_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[ T].*)?$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _fold(text: str) -> str:
    """Minúsculas, sin tildes y con espacios recortados: «  Acción » → «accion»."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _enum_value(enum: type[StrEnum], value: Any) -> Any:
    """Acepta el valor de un enum sin distinguir mayúsculas ni tildes; si no coincide, lo deja."""
    if isinstance(value, str) and not isinstance(value, enum):
        folded = _fold(value)
        for member in enum:
            if folded in (_fold(member.value), member.name.lower()):
                return member
    return value


class CSVMetadata(BaseModel):
    symbol: str
    name: str
    currency: str
    asset_class: AssetClass
    price_type: PriceType = PriceType.CLOSE
    origin: DataOrigin = DataOrigin.HISTORICAL
    notes: tuple[str, ...] = ()
    seed: int | None = None  # semilla del generador (solo datos sintéticos)

    @field_validator("asset_class", mode="before")
    @classmethod
    def _asset_class(cls, v: Any) -> Any:
        return _enum_value(AssetClass, v)

    @field_validator("price_type", mode="before")
    @classmethod
    def _price_type(cls, v: Any) -> Any:
        return _enum_value(PriceType, v)

    @field_validator("origin", mode="before")
    @classmethod
    def _origin(cls, v: Any) -> Any:
        return _enum_value(DataOrigin, v)

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
            raise ValueError(
                "la moneda debe ser un código ISO 4217 de 3 letras (p. ej. USD, EUR, MXN)"
            )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> CSVMetadata:
        # Cualquier indicio de datos generados (origen, tipo de precio o semilla) los
        # marca como SINTÉTICOS: nunca se presenta un dato sintético como histórico.
        if (
            self.origin == DataOrigin.SYNTHETIC
            or self.price_type == PriceType.SYNTHETIC
            or self.seed is not None
        ):
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


# --------------------------------------------------------------------------- metadatos

FIELD_LABELS: dict[str, str] = {
    "symbol": "«symbol» (símbolo)",
    "name": "«name» (nombre)",
    "currency": "«currency» (moneda)",
    "asset_class": "«asset_class» (clase de activo)",
    "price_type": "«price_type» (tipo de precio)",
    "origin": "«origin» (tipo de datos)",
    "notes": "«notes» (notas)",
    "seed": "«seed» (semilla)",
}
VALID_VALUES: dict[str, tuple[str, ...]] = {
    "asset_class": tuple(c.value for c in AssetClass),
    "price_type": tuple(c.value for c in PriceType),
    "origin": tuple(c.value for c in DataOrigin),
}
_EXPECTED: dict[str, str] = {
    "symbol": "un texto",
    "name": "un texto",
    "currency": "un código de 3 letras",
    "notes": "una lista de textos",
    "seed": "un número entero",
}
_METADATA_HELP = (
    "Campos obligatorios: «symbol» (símbolo), «name» (nombre), «currency» (moneda, código ISO "
    "de 3 letras como USD) y «asset_class» (clase de activo: "
    + ", ".join(VALID_VALUES["asset_class"])
    + "). Opcionales: «price_type» ("
    + ", ".join(VALID_VALUES["price_type"])
    + "), «origin» ("
    + ", ".join(VALID_VALUES["origin"])
    + "), «notes» y «seed»."
)


def _shown(value: Any, limit: int = 60) -> str:
    text = "vacío" if value is None or value == "" else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe_error(err: dict[str, Any]) -> tuple[str | None, str]:
    """(campo, problema en español) para un error de validación de pydantic."""
    loc = err.get("loc") or ()
    field = str(loc[0]) if loc else None
    kind = err.get("type", "")
    value = err.get("input")
    if field is None:
        return None, "deben ser un objeto con campos (por ejemplo, un objeto JSON {…})"
    label = FIELD_LABELS.get(field, f"«{field}»")
    if kind == "missing":
        return field, f"falta el campo {label}"
    if field in VALID_VALUES:
        valid = ", ".join(VALID_VALUES[field])
        return field, (
            f"el campo {label} tiene el valor «{_shown(value)}», que no es válido; valores "
            f"válidos: {valid}"
        )
    if kind == "value_error":
        reason = str((err.get("ctx") or {}).get("error", "")) or "valor no válido"
        return field, f"el campo {label} no es válido (se recibió «{_shown(value)}»): {reason}"
    expected = _EXPECTED.get(field, "otro tipo de valor")
    return field, (
        f"el campo {label} tiene un valor no válido (se recibió «{_shown(value)}»); se esperaba "
        f"{expected}"
    )


def _metadata_error(problems: list[tuple[str | None, str]], source: str | None) -> DataError:
    where = f"de «{source}»" if source else "del CSV"
    text = "; ".join(problem for _, problem in problems)
    return DataError(
        f"Los metadatos {where} no son válidos: {text}. {_METADATA_HELP}",
        code=METADATA_ERROR,
        details={
            "fuente": source,
            "errores": [{"campo": field, "problema": problem} for field, problem in problems],
        },
    )


def _from_validation_error(exc: ValidationError, source: str | None) -> DataError:
    problems = [_describe_error(err) for err in exc.errors(include_url=False)]
    return _metadata_error(problems, source)


def parse_metadata(data: dict | str | bytes, source: str | None = None) -> CSVMetadata:
    """``CSVMetadata`` desde un dict o un JSON (p. ej. un sidecar ``.meta.json``).

    Cualquier problema (JSON inválido, campo que falta, valor no permitido) se
    convierte en DataError(code="CSV_METADATOS_INVALIDOS") con un mensaje en
    español que dice qué campo falla, el valor recibido y los valores válidos.
    ``source`` (opcional) nombra el archivo en el mensaje.
    """
    where = f" «{source}»" if source else ""
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DataError(
                f"El archivo de metadatos{where} no es texto UTF-8. Vuelve a generarlo (lo crea el "
                "laboratorio al cargar el CSV) o guárdalo como UTF-8.",
                code=METADATA_ERROR,
                details={"fuente": source},
            ) from exc
    if isinstance(data, str):
        try:
            data = json.loads(data.removeprefix("\ufeff"))
        except json.JSONDecodeError as exc:
            raise DataError(
                f"El archivo de metadatos{where} no es un JSON válido (error de sintaxis en la "
                f"línea {exc.lineno}, columna {exc.colno}). Puede estar incompleto o haberse "
                f"editado a mano: corrígelo o vuelve a generarlo. {_METADATA_HELP}",
                code=METADATA_ERROR,
                details={"fuente": source, "linea": exc.lineno, "columna": exc.colno},
            ) from exc
    if not isinstance(data, dict):
        problem = "deben ser un objeto JSON con campos ({…}), no una lista ni un valor suelto"
        raise _metadata_error([(None, problem)], source)
    try:
        return CSVMetadata.model_validate(data)
    except ValidationError as exc:
        raise _from_validation_error(exc, source) from None


def build_metadata(**fields: Any) -> CSVMetadata:
    """``CSVMetadata(**fields)`` con errores en español (DataError CSV_METADATOS_INVALIDOS)."""
    try:
        return CSVMetadata(**fields)
    except ValidationError as exc:
        raise _from_validation_error(exc, None) from None


def origin_from_filename(name: str | os.PathLike[str] | None) -> DataOrigin | None:
    """SINTETICO si el nombre del archivo empieza por ``SINTETICO_``; si no, None.

    Se mira solo el nombre (sin carpetas, con separadores de Windows o POSIX), sin
    distinguir mayúsculas ni tildes. El nombre nunca prueba que los datos sean
    históricos: por eso no hay respuesta HISTORICO.
    """
    if not name:
        return None
    base = PureWindowsPath(os.fspath(name)).name  # entiende «\» y «/»
    if _fold(base).upper().startswith(SYNTHETIC_FILE_PREFIX):
        return DataOrigin.SYNTHETIC
    return None


def _as_synthetic(meta: CSVMetadata, note: str) -> CSVMetadata:
    """Copia de los metadatos marcada como SINTÉTICA (con nombre y tipo de precio coherentes)."""
    data = meta.model_dump()
    data["origin"] = DataOrigin.SYNTHETIC
    data["notes"] = meta.notes if note in meta.notes else (*meta.notes, note)
    return CSVMetadata.model_validate(data)


def apply_filename_origin(
    meta: CSVMetadata, filename: str | os.PathLike[str] | None
) -> CSVMetadata:
    """Fuerza el origen SINTETICO si el nombre del archivo lo indica (con una nota)."""
    if meta.is_synthetic or origin_from_filename(filename) != DataOrigin.SYNTHETIC:
        return meta
    base = PureWindowsPath(os.fspath(filename)).name if filename else ""
    return _as_synthetic(
        meta,
        f"El nombre del archivo «{base}» empieza por «{SYNTHETIC_FILE_PREFIX}»: la serie se trata "
        "como datos SINTÉTICOS aunque los metadatos declaraban datos históricos.",
    )


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


def _data_type_origin(cells: list[tuple[int, str]], meta: CSVMetadata) -> CSVMetadata:
    """Metadatos efectivos según la columna ``tipo_datos`` (celdas vacías: sin información)."""
    found: dict[DataOrigin, list[int]] = {}
    unknown: list[tuple[int, str]] = []
    for line, raw in cells:
        value = _fold(raw).upper()
        if not value:
            continue
        try:
            found.setdefault(DataOrigin(value), []).append(line)
        except ValueError:
            unknown.append((line, raw.strip()))
    valid = " o ".join(o.value for o in DataOrigin)
    if unknown:
        raise DataError(
            f"La columna «{DATA_TYPE_COLUMN}» tiene {len(unknown)} valor(es) desconocido(s) "
            f"({_examples(unknown)}). Los valores válidos son {valid} (sin importar mayúsculas "
            "ni tildes); deja la celda vacía si no lo sabes.",
            code="CSV_TIPO_DATOS_DESCONOCIDO",
            details={
                "valores_validos": [o.value for o in DataOrigin],
                "ejemplos": [{"linea": n, "valor": v} for n, v in unknown[:MAX_EXAMPLES]],
            },
        )
    if len(found) > 1:
        first = {o.value: lines[0] for o, lines in found.items()}
        raise DataError(
            f"El CSV mezcla filas HISTÓRICAS y SINTÉTICAS en la columna «{DATA_TYPE_COLUMN}» "
            f"(primera fila {DataOrigin.HISTORICAL.value}: línea {first[DataOrigin.HISTORICAL.value]}; "
            f"primera fila {DataOrigin.SYNTHETIC.value}: línea {first[DataOrigin.SYNTHETIC.value]}). "
            "Una serie no puede combinar precios reales y generados: separa las filas en dos "
            "archivos.",
            code="CSV_TIPO_DATOS_MEZCLADO",
            details={"primeras_lineas": first},
        )
    if DataOrigin.SYNTHETIC in found and not meta.is_synthetic:
        return _as_synthetic(
            meta,
            f"La columna «{DATA_TYPE_COLUMN}» del archivo marca las filas como SINTÉTICAS: la "
            "serie se trata como datos SINTÉTICOS aunque los metadatos declaraban datos "
            "históricos.",
        )
    if DataOrigin.HISTORICAL in found and meta.is_synthetic:
        note = (
            f"La columna «{DATA_TYPE_COLUMN}» dice {DataOrigin.HISTORICAL.value}, pero los "
            "metadatos declaran datos SINTÉTICOS: la serie se trata como SINTÉTICA."
        )
        if note not in meta.notes:
            return meta.model_copy(update={"notes": (*meta.notes, note)})
    return meta


def parse_csv(content: bytes | str, meta: CSVMetadata, provider_name: str = "csv") -> PriceSeries:
    """Serie cruda del CSV con los metadatos efectivos (ver el contrato del módulo)."""
    return _parse(content, meta, provider_name)[0]


def _parse(
    content: bytes | str, meta: CSVMetadata, provider_name: str
) -> tuple[PriceSeries, CSVMetadata]:
    """(serie, metadatos efectivos): la columna ``tipo_datos`` puede forzar el origen SINTETICO."""
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
        normalized = _normalize_header(raw_name)
        canonical = (
            DATA_TYPE_COLUMN if normalized in DATA_TYPE_HEADERS else COLUMN_ALIASES.get(normalized)
        )
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
    data_types: list[tuple[int, str]] = []
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
        if DATA_TYPE_COLUMN in positions:
            data_types.append((line, row[positions[DATA_TYPE_COLUMN]]))
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

    meta = _data_type_origin(data_types, meta)
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
        seed=meta.seed,
    )
    return PriceSeries(meta=series_meta, data=data), meta


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

    def register(
        self, content: bytes | str, meta: CSVMetadata, filename: str | None = None
    ) -> PriceSeries:
        """Valida y guarda el CSV con sus metadatos EFECTIVOS.

        ``filename`` es el nombre original del archivo (opcional): si empieza por
        ``SINTETICO_`` la serie se guarda como SINTÉTICA aunque ``meta`` diga otra cosa.
        """
        meta = apply_filename_origin(meta, filename)
        _, meta = _parse(content, meta, self.name)  # valida antes de guardar nada
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
        csv_path, meta = self._find(symbol)
        # Se relee el archivo: la columna «tipo_datos» puede marcarlo como SINTÉTICO.
        return self._load(csv_path, meta).meta.asset

    # ------------------------------------------------------------------ internos

    @staticmethod
    def _sidecar(csv_path: Path) -> Path:
        return csv_path.with_name(csv_path.stem + META_SUFFIX)

    def _scan(self) -> tuple[list[tuple[Path, CSVMetadata]], list[tuple[Path, str]]]:
        """(entradas válidas, sidecars dañados con el motivo en español)."""
        entries: list[tuple[Path, CSVMetadata]] = []
        broken: list[tuple[Path, str]] = []
        if not self.directory.is_dir():
            return entries, broken
        for sidecar in sorted(self.directory.glob(f"*{META_SUFFIX}")):
            csv_path = sidecar.with_name(sidecar.name.removesuffix(META_SUFFIX) + ".csv")
            if not csv_path.exists():
                continue
            try:
                meta = parse_metadata(sidecar.read_bytes(), source=sidecar.name)
            except OSError as exc:
                broken.append((sidecar, f"no se pudo leer ({type(exc).__name__})"))
                continue
            except DataError as exc:
                broken.append((sidecar, exc.message))
                continue  # un sidecar dañado no impide usar los demás archivos
            # Un archivo guardado como «SINTETICO_…» nunca se presenta como histórico.
            entries.append((csv_path, apply_filename_origin(meta, csv_path.name)))
        return entries, broken

    def _entries(self) -> Iterator[tuple[Path, CSVMetadata]]:
        yield from self._scan()[0]

    def _find(self, symbol: str) -> tuple[Path, CSVMetadata]:
        wanted = (symbol or "").strip().upper()
        entries, broken = self._scan()
        for csv_path, meta in entries:
            if meta.symbol.upper() == wanted:
                return csv_path, meta
        available = sorted(meta.symbol for _, meta in entries)
        damaged = ""
        if broken:
            damaged = (
                " Además, hay archivos de metadatos dañados que no se pudieron leer (si tu serie "
                "es uno de ellos, corrígelo o vuelve a cargar el CSV): "
                + " | ".join(f"{path.name}: {reason}" for path, reason in broken)
            )
        raise ProviderError(
            f"No hay ningún CSV registrado con el símbolo «{symbol}». "
            f"Disponibles: {', '.join(available) or 'ninguno (carga primero un archivo)'}."
            + damaged,
            code="SIMBOLO_SIN_DATOS",
            details={
                "symbol": symbol,
                "disponibles": available,
                "metadatos_danados": [path.name for path, _ in broken],
            },
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
