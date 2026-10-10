"""DTO compartidos de la API y utilidades de serialización (contrato estable).

Convenciones de la API:
- JSON con claves en inglés (código) y textos para el usuario en español.
- Fechas ISO ``AAAA-MM-DD``. NaN/±inf → ``null`` (JSON no admite NaN).
- Series temporales en formato columnar: ``{"dates": [...], "values": [...]}``.
- Toda respuesta de análisis, backtest o simulación lleva ``disclaimer`` (aviso
  educativo) y ``synthetic`` (True si interviene algún dato SINTÉTICO).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mrlab import DISCLAIMER


class ErrorBody(BaseModel):
    code: str = Field(description="Código estable del error (para programas)")
    message: str = Field(description="Mensaje en español para el usuario")
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


#: Respuestas de error documentadas en OpenAPI para todos los endpoints.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Configuración o petición inválida"},
    404: {"model": ErrorResponse, "description": "Recurso no encontrado"},
    422: {"model": ErrorResponse, "description": "Datos no válidos o análisis bloqueado"},
    500: {"model": ErrorResponse, "description": "Error interno inesperado"},
    502: {"model": ErrorResponse, "description": "Fallo del proveedor de datos externo"},
}


class Educational(BaseModel):
    """Campos comunes a toda respuesta con resultados."""

    disclaimer: str = DISCLAIMER
    synthetic: bool = Field(False, description="True si algún dato usado es SINTÉTICO")


class SeriesOut(BaseModel):
    dates: list[str]
    values: list[float | None]


class ColumnsOut(BaseModel):
    """Tabla columnar: ``columns[nombre] = lista``; todas las listas miden lo mismo."""

    columns: dict[str, list[Any]]


def clean_float(x: Any) -> float | None:
    """float JSON-seguro: NaN/±inf/None → None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).date().isoformat()


def series_out(s: pd.Series | None) -> SeriesOut | None:
    if s is None:
        return None
    return SeriesOut(
        dates=[_iso(t) for t in s.index], values=[clean_float(v) for v in s.to_numpy()]
    )


def _clean_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return _iso(v)
    if hasattr(v, "isoformat") and not isinstance(v, str):
        return v.isoformat()
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return clean_float(v)
    if isinstance(v, (str,)):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def frame_columns(df: pd.DataFrame | None, index_name: str = "date") -> ColumnsOut | None:
    """DataFrame → tabla columnar. Si el índice es de fechas, se incluye como ``index_name``."""
    if df is None:
        return None
    cols: dict[str, list[Any]] = {}
    if isinstance(df.index, pd.DatetimeIndex) and index_name not in df.columns:
        cols[index_name] = [_iso(t) for t in df.index]
    for name in df.columns:
        cols[str(name)] = [_clean_cell(v) for v in df[name].tolist()]
    return ColumnsOut(columns=cols)


def any_synthetic(flags: Iterable[bool]) -> bool:
    return any(bool(f) for f in flags)
