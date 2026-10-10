"""Explorador de relaciones: correlaciones de log-rendimientos, correlación móvil y base 100 (sección 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrlab.domain.errors import ConfigError, DataError
from mrlab.stats.correlation import (
    correlation_matrix,
    log_returns,
    normalized_prices,
    rolling_correlation,
)


def price_frame(
    n: int = 400, seed: int = 0, start: str = "2020-01-01", freq: str = "B"
) -> pd.DataFrame:
    """Tres activos con rendimientos correlacionados (para que las correlaciones no sean ~0)."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.01, n)
    rets = np.column_stack(
        [
            common + rng.normal(0.0, 0.01, n),
            0.5 * common + rng.normal(0.0, 0.01, n),
            rng.normal(0.0, 0.01, n),
        ]
    )
    idx = pd.date_range(start, periods=n, freq=freq, name="date")
    return pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=["X", "Y", "Z"])


def test_matches_pandas_on_aligned_log_returns() -> None:
    df = price_frame()
    res = correlation_matrix({c: df[c] for c in df})
    rets = np.log(df).diff().dropna()

    np.testing.assert_allclose(
        res.pearson.to_numpy(), rets.corr().to_numpy(), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        res.spearman.to_numpy(), rets.corr(method="spearman").to_numpy(), rtol=1e-12, atol=1e-15
    )
    assert list(res.pearson.index) == list(res.pearson.columns) == ["X", "Y", "Z"]
    assert (res.nobs.to_numpy() == len(rets)).all()
    assert res.pearson.loc["X", "Y"] > 0.3  # el factor común se refleja


def test_nobs_per_cell_uses_common_dates() -> None:
    df = price_frame(n=300)
    x = df["X"]
    y = df["Y"].iloc[100:]  # empieza más tarde
    z = df["Z"].copy()
    z.iloc[[10, 150, 151]] = np.nan  # faltantes: no son observaciones y no se rellenan
    res = correlation_matrix({"X": x, "Y": y, "Z": z})

    assert res.nobs.loc["X", "X"] == 299
    assert res.nobs.loc["Y", "Y"] == 199
    assert res.nobs.loc["Z", "Z"] == 296
    assert res.nobs.loc["X", "Y"] == res.nobs.loc["Y", "X"] == 199
    assert res.nobs.loc["X", "Z"] == 296
    assert res.nobs.loc["Y", "Z"] == 197  # 200 fechas comunes − 2 faltantes − 1
    assert res.nobs.dtypes.map(pd.api.types.is_integer_dtype).all()
    # La celda X–Z usa rendimientos que saltan el hueco (sin forward-fill).
    common = pd.concat([x, z], axis=1, join="inner").dropna()
    expected = np.log(common).diff().dropna().corr().iloc[0, 1]
    assert res.pearson.loc["X", "Z"] == pytest.approx(expected, rel=1e-12)
    assert np.allclose(np.diag(res.pearson), 1.0) and np.allclose(np.diag(res.spearman), 1.0)
    pd.testing.assert_frame_equal(res.pearson, res.pearson.T)


def test_crypto_and_equity_returns_on_common_dates() -> None:
    rng = np.random.default_rng(7)
    days = pd.date_range("2021-01-01", periods=420, freq="D", name="date")  # cripto: 7 días
    crypto = pd.Series(
        30000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days)))), index=days, name="BTC-USD"
    )
    bdays = days[days.dayofweek < 5]  # bolsa: lunes a viernes
    equity = pd.Series(
        400 * np.exp(np.cumsum(rng.normal(0, 0.01, len(bdays)))), index=bdays, name="SPY"
    )

    res = correlation_matrix({"BTC-USD": crypto, "SPY": equity})
    assert res.nobs.loc["BTC-USD", "BTC-USD"] == len(days) - 1
    assert res.nobs.loc["SPY", "SPY"] == len(bdays) - 1
    assert res.nobs.loc["BTC-USD", "SPY"] == len(bdays) - 1

    # El rendimiento cripto del lunes incluye el fin de semana: ln(P_lunes / P_viernes).
    crypto_on_bdays = np.log(crypto.reindex(bdays)).diff().dropna()
    monday = crypto_on_bdays.index[crypto_on_bdays.index.dayofweek == 0][0]
    friday = monday - pd.Timedelta(days=3)
    assert crypto_on_bdays[monday] == pytest.approx(
        np.log(crypto[monday] / crypto[friday]), rel=1e-12
    )

    expected = crypto_on_bdays.corr(np.log(equity).diff().dropna())
    assert res.pearson.loc["BTC-USD", "SPY"] == pytest.approx(expected, rel=1e-12)
    # Si se usaran los rendimientos diarios propios de la cripto, la correlación sería otra.
    naive = log_returns(crypto).reindex(bdays[1:]).corr(np.log(equity).diff().dropna())
    assert not np.isclose(naive, expected)


def test_rolling_correlation_values_and_warmup() -> None:
    df = price_frame(n=300, seed=3)
    roll = rolling_correlation(df["X"], df["Y"], window=63)
    rets = np.log(df).diff().dropna()

    assert len(roll) == len(rets)
    assert roll.iloc[:62].isna().all() and roll.iloc[62:].notna().all()
    t = 200
    window = rets.iloc[t - 62 : t + 1]
    assert roll.iloc[t] == pytest.approx(np.corrcoef(window["X"], window["Y"])[0, 1], rel=1e-9)


def test_rolling_correlation_has_no_lookahead() -> None:
    df = price_frame(n=300, seed=4)
    base = rolling_correlation(df["X"], df["Y"])
    cutoff = df.index[180]
    tampered = df.copy()
    rng = np.random.default_rng(99)
    tampered.loc[tampered.index > cutoff] *= np.exp(rng.normal(0, 0.2, (119, 3)))
    after = rolling_correlation(tampered["X"], tampered["Y"])

    pd.testing.assert_series_equal(base.loc[:cutoff], after.loc[:cutoff])
    assert not np.allclose(base.loc[base.index > cutoff], after.loc[after.index > cutoff])


def test_rolling_correlation_aligns_mixed_calendars() -> None:
    rng = np.random.default_rng(5)
    days = pd.date_range("2022-01-01", periods=200, freq="D")
    bdays = days[days.dayofweek < 5]
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))), index=days)
    b = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(bdays)))), index=bdays)
    roll = rolling_correlation(a, b, window=20)
    assert roll.index.equals(bdays[1:])


def test_normalized_prices_base_100_on_first_common_date() -> None:
    df = price_frame(n=100)
    late = df["Y"].iloc[10:]
    norm = normalized_prices({"X": df["X"], "Y": late})
    assert norm.index[0] == df.index[10]
    assert (norm.iloc[0] == 100.0).all()
    np.testing.assert_allclose(norm["X"], 100 * df["X"].iloc[10:] / df["X"].iloc[10], rtol=1e-12)
    # Acepta también un DataFrame y otra base.
    norm_df = normalized_prices(df, base=1.0)
    assert (norm_df.iloc[0] == 1.0).all() and list(norm_df.columns) == ["X", "Y", "Z"]


def test_non_positive_prices_raise_data_error() -> None:
    df = price_frame(n=50)
    bad = df["X"].copy()
    bad.iloc[20] = 0.0
    bad.name = "MALO"
    with pytest.raises(DataError, match="MALO") as exc:
        log_returns(bad)
    assert "estrictamente positivos" in exc.value.message
    with pytest.raises(
        DataError, match="CLAVE"
    ):  # el mensaje usa la clave aunque la serie no tenga nombre
        correlation_matrix({"CLAVE": bad.rename(None), "Y": df["Y"]})
    with pytest.raises(DataError):
        rolling_correlation(bad, df["Y"], window=10)
    negative = df["Y"].copy()
    negative.iloc[0] = -1.0
    with pytest.raises(DataError):
        normalized_prices({"X": df["X"], "Y": negative})


def test_normalized_prices_without_common_dates() -> None:
    a = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    b = pd.Series([1.0, 2.0], index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
    with pytest.raises(DataError, match="fechas en común"):
        normalized_prices({"A": a, "B": b})


def test_invalid_rolling_window() -> None:
    # Error del laboratorio (código estable y mensaje en español), no un ValueError crudo.
    df = price_frame(n=50)
    with pytest.raises(ConfigError, match="al menos 2 observaciones") as exc:
        rolling_correlation(df["X"], df["Y"], window=1)
    assert exc.value.code == "VENTANA_INVALIDA"


def _with_duplicate(series: pd.Series, position: int, first_value: float) -> pd.Series:
    """La serie con la fecha ``position`` repetida: primero ``first_value`` y después el valor
    original (como llega cruda del proveedor o de la caché)."""
    dup = pd.Series([first_value], index=series.index[[position]], name=series.name)
    return pd.concat([series.iloc[:position], dup, series.iloc[position:]])


def test_duplicate_dates_keep_the_last_value_like_the_validation() -> None:
    df = price_frame(n=120)
    x, y = df["X"], df["Y"]
    y_dup = _with_duplicate(y, 30, first_value=y.iloc[30] * 3)  # el primero es erróneo
    assert y_dup.index.has_duplicates

    expected = correlation_matrix({"X": x, "Y": y})
    got = correlation_matrix({"X": x, "Y": y_dup})
    pd.testing.assert_frame_equal(got.pearson, expected.pearson)
    pd.testing.assert_frame_equal(got.spearman, expected.spearman)
    pd.testing.assert_frame_equal(got.nobs, expected.nobs)
    pd.testing.assert_series_equal(
        rolling_correlation(x, y_dup, window=20),
        rolling_correlation(x, y, window=20),
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        normalized_prices({"X": x, "Y": y_dup}),
        normalized_prices({"X": x, "Y": y}),
        check_freq=False,
    )
    pd.testing.assert_series_equal(log_returns(y_dup), log_returns(y), check_freq=False)


def test_duplicate_dates_in_unsorted_input_keep_the_last_appearance() -> None:
    df = price_frame(n=60)
    y = df["Y"]
    # Desordenada y con la fecha 10 repetida: el orden estable conserva la última aparición.
    y_dup = _with_duplicate(y, 10, first_value=y.iloc[10] * 2).iloc[::-1]
    # Al invertir, la aparición «última» del archivo es ahora el valor erróneo (×2).
    returns = log_returns(y_dup)
    assert not returns.index.has_duplicates
    assert returns.index.is_monotonic_increasing
    expected_level = y.copy()
    expected_level.iloc[10] = y.iloc[10] * 2
    pd.testing.assert_series_equal(returns, log_returns(expected_level), check_freq=False)


def test_series_without_dates_raise_data_error() -> None:
    with pytest.raises(DataError, match="indexada por fecha"):
        log_returns(pd.Series([1.0, 2.0, 3.0], name="SIN_FECHAS"))
