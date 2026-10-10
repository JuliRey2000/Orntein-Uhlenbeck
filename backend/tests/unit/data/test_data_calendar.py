"""Índice de fechas canónico y política de calendarios."""

from __future__ import annotations

from datetime import date

import pandas as pd

from mrlab.data.calendar import (
    calendar_type,
    calendar_warnings,
    periods_per_year,
    time_unit_label,
    to_date_index,
)
from mrlab.domain.models import AssetClass, CalendarType


def test_to_date_index_drops_timezone_keeping_local_date():
    tokyo = pd.DatetimeIndex(["2020-01-06 00:00", "2020-01-07 00:00"]).tz_localize("Asia/Tokyo")
    idx = to_date_index(tokyo)
    assert idx.tz is None and idx.name == "date" and idx.freq is None
    assert [ts.date() for ts in idx] == [date(2020, 1, 6), date(2020, 1, 7)]
    assert idx.dtype == "datetime64[us]"


def test_to_date_index_normalizes_times_and_drops_freq():
    idx = to_date_index(pd.date_range("2020-01-01 16:00", periods=3, freq="D"))
    assert (idx == idx.normalize()).all() and idx.freq is None


def test_calendar_policy():
    assert calendar_type([AssetClass.EQUITY, AssetClass.ETF]) == CalendarType.EXCHANGE
    assert calendar_type([AssetClass.CRYPTO, AssetClass.CRYPTO]) == CalendarType.CRYPTO_7D
    assert calendar_type([AssetClass.CRYPTO, AssetClass.ETF]) == CalendarType.MIXED
    assert periods_per_year(CalendarType.CRYPTO_7D) == 365
    assert periods_per_year(CalendarType.MIXED) == 252
    assert "7 días" in time_unit_label(CalendarType.CRYPTO_7D)
    assert calendar_warnings(CalendarType.EXCHANGE) == []
    assert len(calendar_warnings(CalendarType.MIXED)) == 1
