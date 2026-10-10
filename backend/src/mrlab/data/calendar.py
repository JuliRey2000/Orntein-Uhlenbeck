"""Política de calendarios (sección 2.4) y convenciones de anualización."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from mrlab.domain.models import AssetClass, CalendarType

#: Resolución temporal única para todos los índices de fechas del núcleo.
DATE_UNIT = "us"

CLOSE_TIME_MISMATCH_WARNING = (
    "Par mixto cripto + bolsa: se usan solo los días de negociación de la bolsa (nunca se rellenan "
    "fines de semana ni festivos). En Yahoo, el cierre diario de cripto corresponde a la medianoche "
    "UTC, mientras que la bolsa de EE. UU. cierra a las 16:00 hora de Nueva York: hay un desfase de "
    "varias horas entre ambos precios «de cierre». Para fines educativos con datos diarios se acepta, "
    "pero los resultados pueden verse afectados."
)


def to_date_index(values) -> pd.DatetimeIndex:
    """Índice de fechas canónico del núcleo (ver ``mrlab.domain.models``).

    Sin zona horaria, normalizado a medianoche, resolución ``DATE_UNIT``, sin
    ``freq`` y llamado ``date``. Si llega con zona horaria se conserva la fecha
    LOCAL (la de la bolsa): ``tz_localize(None)``, no ``tz_convert``, para que
    un cierre de Tokio no se convierta en la fecha UTC del día anterior.
    """
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize().as_unit(DATE_UNIT)
    return pd.DatetimeIndex(idx.values, name="date")


def calendar_type(classes: Sequence[AssetClass]) -> CalendarType:
    crypto = [c == AssetClass.CRYPTO for c in classes]
    if all(crypto):
        return CalendarType.CRYPTO_7D
    if any(crypto):
        return CalendarType.MIXED
    return CalendarType.EXCHANGE


def periods_per_year(cal: CalendarType) -> int:
    """252 para acciones, ETF y pares mixtos; 365 para pares solo de cripto (sección 10)."""
    return 365 if cal == CalendarType.CRYPTO_7D else 252


def time_unit_label(cal: CalendarType) -> str:
    if cal == CalendarType.CRYPTO_7D:
        return "día de negociación (calendario de 7 días)"
    return "día de negociación"


def calendar_warnings(cal: CalendarType) -> list[str]:
    return [CLOSE_TIME_MISMATCH_WARNING] if cal == CalendarType.MIXED else []
