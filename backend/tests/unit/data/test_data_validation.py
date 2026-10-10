"""Validación de datos (sección 2.3), alineación de calendarios (2.4) y sección 14."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_series

from mrlab.data.calendar import CLOSE_TIME_MISMATCH_WARNING
from mrlab.data.validation import (
    Severity,
    align_pair,
    clean_series,
    format_dates,
    format_int,
    validate_pair,
)
from mrlab.domain.models import AssetClass, CalendarType, SpreadType


def walk(n: int, seed: int = 0, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))


def codes(issues) -> list[str]:
    return [i.code for i in issues]


def issue(issues, code: str):
    found = [i for i in issues if i.code == code]
    assert len(found) == 1, f"se esperaba exactamente una incidencia {code}: {codes(issues)}"
    return found[0]


D = date.fromisoformat


# --------------------------------------------------------------------------- una serie


def test_duplicates_keep_last_value_and_warn_with_dates():
    s = make_series(
        [10.0, 11.0, 12.0, 13.0, 14.0],
        dates=["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-06", "2020-01-07"],
    )
    clean, val = clean_series(s)

    assert clean.data.index.is_unique
    assert clean.data.loc["2020-01-03", "close"] == 12.0  # la ÚLTIMA aparición
    assert val.duplicates_removed == 1
    assert val.duplicate_dates == [D("2020-01-03")]
    dup = issue(val.issues, "DUPLICADOS")
    assert dup.severity == Severity.WARNING
    assert dup.dates == [D("2020-01-03")]
    assert "2020-01-03" in dup.message and "último" in dup.message
    assert clean.meta.n_obs == 4 and val.n_raw == 5 and val.n_clean == 4


def test_unsorted_dates_are_sorted_and_reported():
    s = make_series([3.0, 1.0, 2.0], dates=["2020-01-06", "2020-01-02", "2020-01-03"])
    clean, val = clean_series(s)

    assert val.unsorted
    assert clean.data.index.is_monotonic_increasing
    assert clean.data["close"].tolist() == [1.0, 2.0, 3.0]
    assert issue(val.issues, "DESORDEN_TEMPORAL").severity == Severity.WARNING


def test_unsorted_duplicates_keep_last_occurrence_of_original_order():
    s = make_series(
        [5.0, 1.0, 1.2, 1.1],
        dates=["2020-01-06", "2020-01-02", "2020-01-06", "2020-01-03"],
    )
    clean, val = clean_series(s)

    assert clean.data["close"].tolist() == [1.0, 1.1, 1.2]
    assert codes(val.issues) == ["DESORDEN_TEMPORAL", "DUPLICADOS"]


def test_missing_values_are_dropped_and_never_forward_filled():
    raw_dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    s = make_series([100.0, np.nan, 102.0, 103.0], dates=raw_dates)
    clean, val = clean_series(s)

    # La fila desaparece: ni se rellena con 100 ni aparece ninguna fecha nueva.
    assert D("2020-01-03") not in {ts.date() for ts in clean.data.index}
    assert set(clean.data.index) <= set(s.data.index)
    assert clean.data["close"].tolist() == [100.0, 102.0, 103.0]
    assert not clean.data["close"].isna().any()
    assert val.missing_values == 1 and val.missing_dates == [D("2020-01-03")]
    miss = issue(val.issues, "FALTANTES")
    assert "2020-01-03" in miss.message and "forward-fill" in miss.message


def test_missing_values_are_not_filled_in_the_aligned_pair():
    n = 900
    a_close = walk(n, seed=1)
    a_close[10] = np.nan
    a = make_series(a_close, symbol="A")
    b = make_series(walk(n, seed=2), symbol="B")
    report, aligned = validate_pair(a, b)

    missing_day = a.data.index[10]
    assert missing_day not in aligned.prices.index
    assert len(aligned.prices) == n - 1
    assert not aligned.prices.isna().any().any()
    assert set(aligned.prices.index) <= set(a.data.index) & set(b.data.index)
    assert "FALTANTES" in codes(report.issues)
    assert aligned.only_in_b == [missing_day.date()]


def test_infinite_close_is_treated_as_missing():
    s = make_series([1.0, np.inf, 3.0])
    clean, val = clean_series(s)
    assert len(clean.data) == 2 and val.missing_values == 1


def test_non_positive_prices_are_reported_but_kept():
    s = make_series(
        [10.0, 0.0, -1.0, 11.0], dates=["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    )
    clean, val = clean_series(s)

    assert len(clean.data) == 4
    assert val.non_positive == 2
    assert val.non_positive_dates == [D("2020-01-03"), D("2020-01-06")]
    assert issue(val.issues, "PRECIO_NO_POSITIVO").severity == Severity.ERROR


def test_suspicious_jump_over_50_percent_is_a_warning():
    close = [100.0, 101.0, 202.0, 203.0, 80.0, 81.0, 113.0]  # +100 %, −60.6 %, +39.5 %
    s = make_series(close, dates=pd.bdate_range("2020-01-01", periods=len(close)))
    _, val = clean_series(s)

    idx = s.data.index
    assert val.suspicious_jumps == [idx[2].date(), idx[4].date()]
    jump = issue(val.issues, "SALTO_SOSPECHOSO")
    assert jump.severity == Severity.WARNING
    assert "50 %" in jump.message and "ajuste" in jump.message


@pytest.mark.parametrize(
    "prices, flagged",
    [
        ([100.0, 101.0, 50.7, 51.0], True),  # split 2:1 sin ajustar con el mercado al alza
        ([100.0, 101.0, 202.8, 203.0], True),  # contrasplit 1:2
        ([100.0, 66.0, 66.5], True),  # −34 %: |ln| = 0,416 > ln 1,5
        ([100.0, 68.0, 68.5], False),  # −32 %: |ln| = 0,386
        ([100.0, 149.0, 149.5], False),  # +49 %
        ([100.0, 151.0, 151.5], True),  # +51 %
    ],
)
def test_suspicious_jump_threshold_is_symmetric_in_logs(prices, flagged):
    """Hallazgo de la revisión: un split 2:1 sin ajustar (≈ −50 %) no se marcaba."""
    s = make_series(prices, dates=pd.bdate_range("2020-01-01", periods=len(prices)))
    _, val = clean_series(s)
    assert bool(val.suspicious_jumps) is flagged
    if flagged:
        message = issue(val.issues, "SALTO_SOSPECHOSO").message
        assert "split" in message and "33,3 %" in message and "50 %" in message


def test_crypto_gap_is_reported_but_not_filled():
    days = pd.date_range("2021-01-01", periods=10, freq="D").delete([3, 4])
    s = make_series(np.arange(1.0, 9.0), dates=days, asset_class=AssetClass.CRYPTO)
    clean, val = clean_series(s)

    assert val.crypto_gaps == [D("2021-01-04"), D("2021-01-05")]
    assert issue(val.issues, "HUECO_CRIPTO").severity == Severity.WARNING
    assert len(clean.data) == 8  # sin relleno


def test_exchange_series_does_not_infer_holidays():
    s = make_series(walk(30))  # días hábiles: sin fines de semana, no es un hueco
    _, val = clean_series(s)
    assert val.crypto_gaps == [] and val.issues == []


# --------------------------------------------------------------------------- par


def test_clean_pair_is_valid():
    report, aligned = validate_pair(
        make_series(walk(1000, 1), symbol="A"), make_series(walk(1000, 2), symbol="B")
    )

    assert not report.blocked and not report.insufficient_data and report.log_spread_allowed
    assert report.calendar == CalendarType.EXCHANGE
    assert report.n_aligned == 1000 and report.n_train_estimate == 600
    assert list(aligned.prices.columns) == ["a", "b"]
    assert aligned.prices.index.name == "date"
    assert report.status_message.startswith("Datos aptos")
    assert report.issues == []


def test_non_positive_prices_block_log_spread_only():
    close = walk(1000, 3)
    close[100] = 0.0
    close[200] = -5.0
    a = make_series(close, symbol="A")
    b = make_series(walk(1000, 4), symbol="B")

    report, aligned = validate_pair(a, b, SpreadType.LOG)
    assert report.blocked and not report.log_spread_allowed
    blocked = issue(report.issues, "SPREAD_LOG_BLOQUEADO")
    assert blocked.severity == Severity.ERROR
    assert "logarítmico" in blocked.message and "positivos" in blocked.message
    assert blocked.dates == [a.data.index[100].date(), a.data.index[200].date()]
    assert "logarítmico" in report.status_message
    assert aligned is not None  # el bloqueo no es por moneda

    report_p, aligned_p = validate_pair(a, b, SpreadType.PRICE)
    assert not report_p.blocked
    assert not report_p.log_spread_allowed
    assert "SPREAD_LOG_BLOQUEADO" not in codes(report_p.issues)
    assert issue(report_p.issues, "PRECIOS_NO_POSITIVOS").severity == Severity.WARNING
    assert aligned_p is not None


def test_status_message_is_not_apt_when_there_are_error_issues():
    """Hallazgo de la revisión: «Datos aptos» con incidencias de severidad error."""
    close = walk(1000, 4)
    close[500] = 0.0
    close[501] = -0.5
    a = make_series(walk(1000, 3), symbol="A")
    b = make_series(close, symbol="B")
    report, _ = validate_pair(a, b, SpreadType.PRICE)
    assert not report.blocked
    assert any(i.severity == Severity.ERROR for i in report.issues)
    assert "aptos" not in report.status_message
    assert report.status_message.startswith("Datos utilizables con incidencias")
    assert "«B»" in report.status_message and "cero" in report.status_message
    assert "1.000 observaciones comunes, 600 en formación" in report.status_message


def test_non_positive_price_outside_common_dates_does_not_block():
    a_close = walk(1001, 3)
    a_close[-1] = 0.0  # la última fecha solo existe en A
    a = make_series(a_close, symbol="A")
    b = make_series(walk(1000, 4), symbol="B")
    report, _ = validate_pair(a, b)
    assert report.log_spread_allowed and not report.blocked


def test_currency_mismatch_blocks_and_mentions_conversion_extension():
    a = make_series(walk(1000, 1), symbol="SAP", currency="EUR")
    b = make_series(walk(1000, 2), symbol="MSFT", currency="USD")
    report, aligned = validate_pair(a, b)

    assert report.blocked and report.currency_mismatch
    assert aligned is None
    err = issue(report.issues, "MONEDA_DISTINTA")
    assert err.severity == Severity.ERROR
    assert "EUR" in err.message and "USD" in err.message
    assert "conversión de moneda" in err.message and "extensión" in err.message
    assert "monedas distintas" in report.status_message


def test_unknown_currency_blocks():
    a = make_series(walk(1000, 1), symbol="A", currency=None)
    b = make_series(walk(1000, 2), symbol="B")
    report, aligned = validate_pair(a, b)

    assert report.blocked and not report.currency_mismatch
    assert aligned is None
    err = issue(report.issues, "MONEDA_DESCONOCIDA")
    assert "«A»" in err.message and "moneda" in err.message


def test_insufficient_training_data_reports_numbers():
    report, aligned = validate_pair(
        make_series(walk(800, 1), symbol="A"), make_series(walk(800, 2), symbol="B")
    )

    assert report.insufficient_data and report.blocked
    assert report.n_aligned == 800 and report.n_train_estimate == 480
    err = issue(report.issues, "DATOS_INSUFICIENTES")
    for number in ("800", "480", "500", "834", "60 %"):
        assert number in err.message
    assert err.details["total_necesario"] == 834
    assert "480" in report.status_message
    assert aligned is not None


@pytest.mark.parametrize(("n", "insufficient"), [(833, True), (834, False)])
def test_minimum_training_boundary_matches_floor_rule(n, insufficient):
    report, _ = validate_pair(
        make_series(walk(n, 1), symbol="A"), make_series(walk(n, 2), symbol="B")
    )
    assert report.insufficient_data is insufficient


def test_all_blocking_causes_are_listed():
    a = make_series(walk(100, 1), symbol="A", currency="EUR")
    b = make_series(walk(100, 2), symbol="B", currency="USD")
    report, _ = validate_pair(a, b)
    assert {"MONEDA_DISTINTA", "DATOS_INSUFICIENTES"} <= set(codes(report.issues))
    assert "monedas distintas" in report.status_message and "formación" in report.status_message


def test_report_includes_series_issues_with_symbol():
    a_close = walk(1000, 1)
    a_close[5] = np.nan
    report, _ = validate_pair(
        make_series(a_close, symbol="A"), make_series(walk(1000, 2), symbol="B")
    )
    missing = issue(report.issues, "FALTANTES")
    assert missing.symbol == "A"
    assert report.series[0].missing_values == 1
    assert not report.blocked and "advertencia" in report.status_message


def test_mixed_crypto_equity_pair_uses_only_exchange_days():
    crypto_days = pd.date_range("2020-01-01", periods=1400, freq="D")
    btc = make_series(
        walk(1400, 5), symbol="BTC-USD", dates=crypto_days, asset_class=AssetClass.CRYPTO
    )
    exchange_days = pd.bdate_range("2020-01-01", "2023-10-31").delete(10)  # un festivo de bolsa
    qqq = make_series(
        walk(len(exchange_days), 6), symbol="QQQ", dates=exchange_days, asset_class=AssetClass.ETF
    )

    report, aligned = validate_pair(qqq, btc)

    assert report.calendar == aligned.calendar == CalendarType.MIXED
    idx = aligned.prices.index
    assert (idx.dayofweek < 5).all()  # fines de semana cripto descartados
    assert set(idx) == set(exchange_days)  # solo días de bolsa (intersección)
    assert pd.Timestamp("2020-01-15") not in set(idx)  # el festivo no se rellena con el dato cripto
    # sin relleno: cada precio es exactamente el de su propia fecha
    np.testing.assert_array_equal(
        aligned.prices["b"].to_numpy(), btc.data["close"].reindex(idx).to_numpy()
    )
    np.testing.assert_array_equal(
        aligned.prices["a"].to_numpy(), qqq.data["close"].reindex(idx).to_numpy()
    )
    assert not aligned.prices.isna().any().any()
    assert report.only_in_b == len(crypto_days) - len(idx)
    assert all(d.weekday() >= 5 or d == D("2020-01-15") for d in aligned.only_in_b)

    lag = issue(report.issues, "DESFASE_CIERRE_CRIPTO")
    assert lag.severity == Severity.WARNING and lag.message == CLOSE_TIME_MISMATCH_WARNING
    assert "medianoche UTC" in lag.message
    assert "fines de semana" in issue(report.issues, "FECHAS_NO_COMUNES").message


def test_two_crypto_series_keep_seven_day_calendar():
    days = pd.date_range("2020-01-01", periods=900, freq="D")
    a = make_series(walk(900, 1), symbol="BTC-USD", dates=days, asset_class=AssetClass.CRYPTO)
    b = make_series(walk(900, 2), symbol="ETH-USD", dates=days, asset_class=AssetClass.CRYPTO)
    report, aligned = validate_pair(a, b)
    assert report.calendar == CalendarType.CRYPTO_7D
    assert (aligned.prices.index.dayofweek >= 5).any()
    assert "DESFASE_CIERRE_CRIPTO" not in codes(report.issues)


def test_align_pair_accepts_raw_series():
    a = make_series([1.0, 2.0, 3.0], dates=["2020-01-03", "2020-01-02", "2020-01-03"], symbol="A")
    b = make_series([5.0, 6.0], dates=["2020-01-02", "2020-01-03"], symbol="B")
    aligned = align_pair(a, b)
    assert aligned.prices.index.is_unique and aligned.prices.index.is_monotonic_increasing
    assert aligned.prices.loc["2020-01-03", "a"] == 3.0
    assert aligned.symbol_a == "A" and aligned.symbol_b == "B"


def test_message_formatting_helpers():
    dates = [D("2020-01-02") + pd.Timedelta(days=i) for i in range(10)]
    text = format_dates(dates, limit=3)
    assert text.endswith("y 7 más") and text.startswith("2020-01-02")
    assert format_int(4214) == "4.214"
