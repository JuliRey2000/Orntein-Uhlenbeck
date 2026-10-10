"""Validación de datos previa a cualquier análisis (sección 2.3) y alineación (2.4).

CONTRATO (a implementar):
- ``clean_series(series)`` → (serie limpia, SeriesValidation):
  * desorden temporal → se ordena y se reporta (DESORDEN_TEMPORAL, advertencia);
  * fechas duplicadas → se conserva la ÚLTIMA aparición, aviso con las fechas
    (DUPLICADOS, advertencia);
  * close NaN → se ELIMINA la fila (nunca forward-fill), se reporta con fechas
    (FALTANTES, advertencia);
  * close ≤ 0 → se reporta (PRECIO_NO_POSITIVO, error) y NO se elimina: la
    decisión de bloquear la toma ``validate_pair`` según el tipo de spread;
  * |rendimiento simple diario| > 50 % → advertencia (SALTO_SOSPECHOSO) con
    fechas: posible error de ajuste;
  * cripto (calendario 7 días): huecos de más de 1 día natural → advertencia
    (HUECO_CRIPTO) con fechas. Para bolsa no se infieren festivos.
- ``align_pair(a, b)`` → AlignedPair: intersección de fechas de las series
  limpias; NUNCA se rellenan fechas. Registra las fechas que solo estaban en una
  de las dos series. Calendario según ``mrlab.data.calendar.calendar_type``.
- ``validate_pair(a, b, spread_type, min_train_obs=500, train_fraction=0.6)``
  → (ValidationReport, AlignedPair | None):
  * monedas distintas → blocked=True, currency_mismatch=True, error MONEDA_DISTINTA
    explicando que la conversión de moneda queda como extensión de la v1;
    moneda desconocida (None) → blocked=True, MONEDA_DESCONOCIDA;
  * precios ≤ 0 en cualquier serie → log_spread_allowed=False; si spread_type es
    LOG → blocked=True (SPREAD_LOG_BLOQUEADO, explica que el log exige precios > 0);
  * n_train = floor(n_alineadas · train_fraction) < min_train_obs →
    insufficient_data=True, blocked=True, error DATOS_INSUFICIENTES con números
    (cuántas hay, cuántas hacen falta, cuántas totales se necesitan);
  * calendario mixto → advertencia DESFASE_CIERRE_CRIPTO (texto en calendar.py);
  * ``status_message``: resumen en una frase, en español.
  Si está bloqueado por moneda, se devuelve AlignedPair=None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel

from mrlab.domain.models import CalendarType, PriceSeries, SpreadType

SUSPICIOUS_JUMP = 0.50
MIN_TRAIN_OBS = 500


class Severity(StrEnum):
    INFO = "info"
    WARNING = "advertencia"
    ERROR = "error"


class ValidationIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    symbol: str | None = None
    dates: list[date] = []
    details: dict = {}


class SeriesValidation(BaseModel):
    symbol: str
    n_raw: int
    n_clean: int
    unsorted: bool = False
    duplicates_removed: int = 0
    duplicate_dates: list[date] = []
    missing_values: int = 0
    missing_dates: list[date] = []
    non_positive: int = 0
    non_positive_dates: list[date] = []
    suspicious_jumps: list[date] = []
    crypto_gaps: list[date] = []
    issues: list[ValidationIssue] = []


class ValidationReport(BaseModel):
    series: list[SeriesValidation]
    calendar: CalendarType | None = None
    n_aligned: int = 0
    n_train_estimate: int = 0
    min_train_obs: int = MIN_TRAIN_OBS
    only_in_a: int = 0
    only_in_b: int = 0
    currency_mismatch: bool = False
    log_spread_allowed: bool = True
    insufficient_data: bool = False
    blocked: bool = False
    issues: list[ValidationIssue] = []
    status_message: str = ""


@dataclass(frozen=True)
class AlignedPair:
    prices: pd.DataFrame  # columnas "a", "b"; índice = fechas comunes (ordenadas, únicas)
    symbol_a: str
    symbol_b: str
    calendar: CalendarType
    only_in_a: list[date] = field(default_factory=list)
    only_in_b: list[date] = field(default_factory=list)


def clean_series(series: PriceSeries) -> tuple[PriceSeries, SeriesValidation]:
    raise NotImplementedError


def align_pair(a: PriceSeries, b: PriceSeries) -> AlignedPair:
    raise NotImplementedError


def validate_pair(
    a: PriceSeries,
    b: PriceSeries,
    spread_type: SpreadType = SpreadType.LOG,
    min_train_obs: int = MIN_TRAIN_OBS,
    train_fraction: float = 0.6,
) -> tuple[ValidationReport, AlignedPair | None]:
    raise NotImplementedError
