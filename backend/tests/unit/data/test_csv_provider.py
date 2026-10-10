"""Proveedor CSV: formatos tolerantes, errores claros y registro con metadatos."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mrlab.data.providers.csv_provider import CSVMetadata, CSVProvider, parse_csv
from mrlab.domain.errors import DataError, ProviderError
from mrlab.domain.models import AssetClass, DataOrigin, PriceType

META = CSVMetadata(symbol="KO", name="Coca-Cola", currency="USD", asset_class=AssetClass.EQUITY)


def dates_of(series) -> list[date]:
    return [ts.date() for ts in series.data.index]


def test_comma_separated():
    s = parse_csv("date,close,open\n2020-01-02,10.5,10\n2020-01-03,11.25,10.6\n", META)
    assert dates_of(s) == [date(2020, 1, 2), date(2020, 1, 3)]
    assert s.close.tolist() == [10.5, 11.25]
    assert list(s.data.columns) == ["close", "open"]
    assert (s.data.dtypes == "float64").all()
    assert s.data.index.name == "date" and s.data.index.tz is None
    assert s.meta.asset.symbol == "KO" and s.meta.asset.currency == "USD"
    assert s.meta.price_type == PriceType.CLOSE and s.meta.n_obs == 2
    assert s.meta.asset.provider == "csv"


def test_semicolon_with_decimal_comma_and_thousands():
    text = "fecha;cierre;volumen\n2020-01-02;1.234,56;1000\n2020-01-03;10,5;\n2020-01-06;12.75;3\n"
    s = parse_csv(text, META)
    assert s.close.tolist() == [1234.56, 10.5, 12.75]
    assert s.data["volume"].iloc[0] == 1000.0 and np.isnan(s.data["volume"].iloc[1])


def test_tab_separated():
    s = parse_csv("date\tclose\n2020-01-02\t3,5\n", META)
    assert s.close.tolist() == [3.5]


def test_spanish_aliases_case_accents_and_spaces():
    text = " Fecha ; Cierre ; Máximo ; MINIMO ; Apertura ; Volumen ; Notas\n2020-01-02;10;11;9;9,5;100;x\n"
    s = parse_csv(text, META)
    assert list(s.data.columns) == ["close", "open", "high", "low", "volume"]
    assert s.data.iloc[0].tolist() == [10.0, 9.5, 11.0, 9.0, 100.0]


def test_utf8_bom_bytes():
    content = "﻿date,close\n2020-01-02,1.5\n".encode()
    assert content.startswith(b"\xef\xbb\xbf")
    s = parse_csv(content, META)
    assert s.close.tolist() == [1.5]


def test_excel_windows_1252_bytes():
    content = "fecha;cierre;máximo\n2020-01-02;1,5;2\n".encode("cp1252")
    s = parse_csv(content, META)
    assert s.data["high"].tolist() == [2.0]


def test_datetime_values_drop_the_time():
    text = "date,close\n2020-01-02 16:00:00,1\n2020-01-03T00:00:00Z,2\n"
    s = parse_csv(text, META)
    assert dates_of(s) == [date(2020, 1, 2), date(2020, 1, 3)]


def test_raw_rows_are_not_sorted_deduplicated_or_filled():
    text = "date,close\n2020-01-03,2\n2020-01-02,1\n2020-01-03,2.5\n2020-01-06,\n2020-01-07,#N/D\n"
    s = parse_csv(text, META)
    assert dates_of(s) == [
        date(2020, 1, 3),
        date(2020, 1, 2),
        date(2020, 1, 3),
        date(2020, 1, 6),
        date(2020, 1, 7),
    ]
    assert s.close.iloc[:3].tolist() == [2.0, 1.0, 2.5]
    assert s.close.iloc[3:].isna().all()


def test_blank_lines_are_ignored():
    s = parse_csv("date;close\n\n2020-01-02;1\n;\n", META)
    assert len(s.data) == 1


@pytest.mark.parametrize("content", [b"", "", "   \n\n", "date,close\n", "date,close\n,\n"])
def test_empty_file(content):
    with pytest.raises(DataError) as exc_info:
        parse_csv(content, META)
    assert exc_info.value.code == "CSV_VACIO"
    assert "encabezado" in exc_info.value.message


def test_missing_required_column_lists_found_columns():
    with pytest.raises(DataError) as exc_info:
        parse_csv("fecha,precio\n2020-01-02,1\n", META)
    err = exc_info.value
    assert err.code == "CSV_FALTA_COLUMNA"
    assert "«close»" in err.message and "«fecha»" in err.message and "«precio»" in err.message
    assert err.details["faltan"] == ["close"]


def test_wrong_separator_hint_when_columns_merge():
    with pytest.raises(DataError) as exc_info:
        parse_csv("date|close\n2020-01-02|1\n", META)
    assert exc_info.value.code == "CSV_FALTA_COLUMNA"
    assert "separador" in exc_info.value.message


@pytest.mark.parametrize("bad", ["02/01/2020", "2020-02-30", "ayer", ""])
def test_invalid_dates_show_examples(bad):
    with pytest.raises(DataError) as exc_info:
        parse_csv(f"date,close\n2020-01-02,1\n{bad},2\n", META)
    err = exc_info.value
    assert err.code == "CSV_FECHA_INVALIDA"
    assert "línea 3" in err.message and f"«{bad}»" in err.message
    assert "AAAA-MM-DD" in err.message
    assert err.details["ejemplos"] == [{"linea": 3, "valor": bad}]


def test_non_numeric_close_shows_examples():
    with pytest.raises(DataError) as exc_info:
        parse_csv("date,close\n2020-01-02,abc\n2020-01-03,1\n2020-01-06,1O\n", META)
    err = exc_info.value
    assert err.code == "CSV_VALOR_NO_NUMERICO"
    assert "«close»" in err.message and "«abc»" in err.message and "«1O»" in err.message
    assert "2 valor" in err.message


def test_decimal_comma_in_comma_file_is_malformed():
    with pytest.raises(DataError) as exc_info:
        parse_csv("date,close\n2020-01-02,1,5\n", META)
    assert exc_info.value.code == "CSV_MAL_FORMADO"
    assert "línea 2" in exc_info.value.message


def test_duplicate_column_alias():
    with pytest.raises(DataError) as exc_info:
        parse_csv("fecha,date,close\n2020-01-02,2020-01-02,1\n", META)
    assert exc_info.value.code == "CSV_COLUMNA_DUPLICADA"


def test_metadata_normalization_and_validation():
    meta = CSVMetadata(symbol=" ko ", name="", currency="usd", asset_class=AssetClass.EQUITY)
    assert meta.symbol == "ko" and meta.name == "ko" and meta.currency == "USD"
    with pytest.raises(ValidationError, match="3 letras"):
        CSVMetadata(symbol="KO", name="KO", currency="dólares", asset_class=AssetClass.EQUITY)
    with pytest.raises(ValidationError, match="símbolo"):
        CSVMetadata(symbol="  ", name="KO", currency="USD", asset_class=AssetClass.EQUITY)


def test_synthetic_metadata_is_always_labelled():
    meta = CSVMetadata(
        symbol="SINT-X",
        name="Mi par",
        currency="USD",
        asset_class=AssetClass.EQUITY,
        origin=DataOrigin.SYNTHETIC,
    )
    assert meta.price_type == PriceType.SYNTHETIC
    assert meta.name == "SINTÉTICO — Mi par"
    by_price_type = CSVMetadata(
        symbol="SINT-Y",
        name="SINTÉTICO — Y",
        currency="USD",
        asset_class=AssetClass.EQUITY,
        price_type=PriceType.SYNTHETIC,
    )
    assert by_price_type.origin == DataOrigin.SYNTHETIC and by_price_type.name == "SINTÉTICO — Y"


# --------------------------------------------------------------------------- proveedor

CONTENT = "date,close\n2020-01-02,1\n2020-01-03,2\n2020-01-06,3\n2020-01-07,4\n"


def test_register_get_history_and_sidecar(tmp_path):
    provider = CSVProvider(tmp_path / "uploads")
    registered = provider.register(CONTENT, META)

    csv_file = tmp_path / "uploads" / "KO.csv"
    sidecar = tmp_path / "uploads" / "KO.meta.json"
    assert csv_file.read_text() == CONTENT
    assert CSVMetadata.model_validate(json.loads(sidecar.read_text(encoding="utf-8"))) == META
    assert registered.meta.n_obs == 4

    s = provider.get_history("ko", date(2020, 1, 3), date(2020, 1, 6))
    assert dates_of(s) == [date(2020, 1, 3), date(2020, 1, 6)]
    assert s.meta.requested_start == date(2020, 1, 3) and s.meta.requested_end == date(2020, 1, 6)
    assert s.meta.coverage_start == date(2020, 1, 3) and s.meta.n_obs == 2
    mtime = datetime.fromtimestamp(csv_file.stat().st_mtime, tz=UTC)
    assert s.meta.retrieved_at == mtime

    assert provider.list_symbols() == ["KO"]
    assert provider.get_metadata("KO").name == "Coca-Cola"
    assert [a.symbol for a in provider.search("coca")] == ["KO"]
    assert [a.symbol for a in provider.search("")] == ["KO"]
    assert provider.search("pepsi") == []


def test_unknown_symbol_and_empty_range(tmp_path):
    provider = CSVProvider(tmp_path)
    assert provider.list_symbols() == []
    provider.register(CONTENT, META)
    with pytest.raises(ProviderError) as exc_info:
        provider.get_history("PEP", date(2020, 1, 1), date(2020, 12, 31))
    assert exc_info.value.code == "SIMBOLO_SIN_DATOS" and "KO" in exc_info.value.message
    with pytest.raises(ProviderError) as exc_info:
        provider.get_history("KO", date(2021, 1, 1), date(2021, 12, 31))
    assert exc_info.value.code == "SIMBOLO_SIN_DATOS"


def test_synthetic_origin_uses_prefixed_file_and_labels(tmp_path):
    provider = CSVProvider(tmp_path)
    meta = CSVMetadata(
        symbol="SINT-COI-A",
        name="Par",
        currency="USD",
        asset_class=AssetClass.EQUITY,
        origin=DataOrigin.SYNTHETIC,
    )
    s = provider.register(CONTENT, meta)
    assert (tmp_path / "SINTETICO_SINT-COI-A.csv").exists()
    assert (tmp_path / "SINTETICO_SINT-COI-A.meta.json").exists()
    assert s.is_synthetic and s.meta.retrieved_at is None
    assert s.meta.price_type == PriceType.SYNTHETIC
    assert s.meta.asset.name.startswith("SINTÉTICO — ")
    assert "SINTÉTICOS" in s.meta.source_note
    assert provider.get_history("SINT-COI-A", date(2020, 1, 1), date(2020, 12, 31)).is_synthetic


def test_invalid_content_is_not_saved(tmp_path):
    provider = CSVProvider(tmp_path)
    with pytest.raises(DataError):
        provider.register("date,close\n2020-01-02,abc\n", META)
    assert list(tmp_path.iterdir()) == []


def test_re_registering_a_symbol_replaces_previous_files(tmp_path):
    provider = CSVProvider(tmp_path)
    provider.register(
        CONTENT, CSVMetadata.model_validate({**META.model_dump(), "origin": DataOrigin.SYNTHETIC})
    )
    provider.register("date,close\n2020-01-02,9\n", META)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["KO.csv", "KO.meta.json"]
    assert provider.get_history("KO", date(2020, 1, 1), date(2020, 1, 31)).close.tolist() == [9.0]


def test_index_symbol_gets_safe_file_name_and_proxy_suggestion(tmp_path):
    provider = CSVProvider(tmp_path)
    meta = CSVMetadata(symbol="^GSPC", name="S&P 500", currency="USD", asset_class=AssetClass.INDEX)
    provider.register(CONTENT, meta)
    assert (tmp_path / "_GSPC.csv").exists()
    asset = provider.get_metadata("^gspc")
    assert not asset.executable and asset.suggested_proxy == "SPY"


def test_symbols_with_same_safe_name_do_not_overwrite_each_other(tmp_path):
    provider = CSVProvider(tmp_path)
    provider.register(
        CONTENT, META.model_copy(update={"symbol": "GC=F", "asset_class": AssetClass.FUTURE})
    )
    provider.register("date,close\n2020-01-02,7\n", META.model_copy(update={"symbol": "GC_F"}))
    assert provider.list_symbols() == ["GC=F", "GC_F"]
    assert provider.get_history("GC=F", date(2020, 1, 1), date(2020, 1, 31)).meta.n_obs == 4
    assert provider.get_history("GC_F", date(2020, 1, 1), date(2020, 1, 31)).close.tolist() == [7.0]


def test_default_directory_comes_from_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("MRLAB_UPLOADS_DIR", str(tmp_path / "subidas"))
    assert CSVProvider().directory == tmp_path / "subidas"


def test_index_dtype_matches_other_providers():
    s = parse_csv(CONTENT, META)
    assert (
        s.data.index.dtype
        == pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=1)).as_unit("us").dtype
    )
