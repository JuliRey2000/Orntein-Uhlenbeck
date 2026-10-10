"""Exportación de extremo a extremo: marca de tipo de datos en cada fila, prefijos, sin marcas de tiempo."""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import date

import pytest
from conftest import cointegrated_log_prices
from integration_helpers import csv_provider, pair_config, series_pair, service

from mrlab import DISCLAIMER
from mrlab.analysis.runner import run_pair
from mrlab.backtest.costs import DIVIDENDS_NOTE
from mrlab.backtest.engine import EXECUTION_CONVENTION
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.domain.models import DataOrigin, Segment
from mrlab.reporting.export import (
    FILE_ORDER,
    ZIP_DATE_TIME,
    export_basename,
    export_files,
    export_run,
    export_zip_bytes,
)


@pytest.fixture(autouse=True)
def fixed_code_version(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "prueba-exportacion")


def csv_rows(content: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(content.decode("utf-8"))))


@pytest.fixture(scope="module")
def synthetic_result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sintetico")
    data = service(tmp / "cache", synthetic=SyntheticProvider())
    cfg = pair_config("synthetic", ("SINT-COI-A", "SINT-COI-B"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_COMMIT", "prueba-exportacion")
        return run_pair(cfg, data, include_test=True)


@pytest.fixture(scope="module")
def historical_result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("historico")
    log_a, log_b = cointegrated_log_prices(n=1200, seed=31, lam=0.1, sigma=0.012)
    data = service(tmp / "cache", csv=csv_provider(tmp / "uploads", log_a, log_b))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_COMMIT", "prueba-exportacion")
        return run_pair(pair_config("csv"), data)


def test_synthetic_export_marks_every_row_and_every_file(synthetic_result):
    files = export_files(synthetic_result)
    assert list(files) == [f"SINTETICO_{name}" for name in FILE_ORDER]
    assert export_basename(synthetic_result) == "SINTETICO_SINT-COI-A_SINT-COI-B"
    for name, content in files.items():
        if not name.endswith(".csv"):
            continue
        rows = csv_rows(content)
        assert rows[0][0] == "tipo_datos"
        assert len(rows) > 1, name
        assert {row[0] for row in rows[1:]} == {"SINTETICO"}, name
    readme = files["SINTETICO_LEEME.txt"].decode("utf-8")
    assert "DATOS SINTÉTICOS" in readme and "NO son precios históricos" in readme
    config = files["SINTETICO_run_config.json"].decode("utf-8")
    assert '"data_origin": "SINTETICO"' in config


def test_historical_export_is_marked_historical(historical_result):
    files = export_files(historical_result)
    assert list(files) == list(FILE_ORDER)  # sin prefijo SINTETICO_
    assert export_basename(historical_result) == "AAA_BBB"
    for name in ("operaciones.csv", "diario.csv", "pruebas_estadisticas.csv"):
        rows = csv_rows(files[name])
        assert {row[0] for row in rows[1:]} == {"HISTORICO"}, name
    readme = files["LEEME.txt"].decode("utf-8")
    assert "Tipo de datos: HISTORICO" in readme
    assert "SINTÉTICO" not in readme


def test_readme_has_disclaimer_execution_convention_and_dividends(historical_result):
    readme = export_files(historical_result)["LEEME.txt"].decode("utf-8")
    assert readme.count(DISCLAIMER) == 2  # al principio y al final
    assert EXECUTION_CONVENTION in readme
    assert DIVIDENDS_NOTE in readme  # cierre ajustado: dividendos ya incluidos
    # Sin prueba no se afirma «no contaminada»: no aplica (antes decía «no.» siempre).
    assert "Prueba contaminada: no aplica: la prueba no se ejecutó." in readme
    assert "Prueba contaminada: no." not in readme
    assert "NO ejecutado" in readme  # la prueba no se ejecutó


def test_contaminated_test_is_reported(historical_result):
    historical_result.test_contaminated = True
    try:
        readme = export_files(historical_result)["LEEME.txt"].decode("utf-8")
    finally:
        historical_result.test_contaminated = None
    assert "Prueba contaminada: SÍ" in readme


def test_readme_states_where_the_contamination_check_comes_from(synthetic_result):
    # La prueba se ejecutó por primera vez con esta configuración: «no», con su fuente.
    assert synthetic_result.test_contaminated is False
    files = export_files(synthetic_result)
    readme = files["SINTETICO_LEEME.txt"].decode("utf-8")
    assert (
        "Prueba contaminada: no (según el historial registrado en run_config.json, la prueba "
        "solo se ha ejecutado con esta configuración)." in readme
    )
    assert "fase 2" in readme  # el registro persistente de estudios llega después
    config = files["SINTETICO_run_config.json"].decode("utf-8")
    assert '"include_test": true' in config
    assert synthetic_result.config.config_hash in config.split('"test_history"')[1]


@pytest.mark.parametrize(
    ("contaminated", "expected"),
    [(None, "no aplica"), (False, "no"), (True, "sí")],
)
def test_statistics_file_reports_the_test_state(historical_result, contaminated, expected):
    historical_result.test_contaminated = contaminated
    try:
        rows = csv_rows(export_files(historical_result)["pruebas_estadisticas.csv"])
    finally:
        historical_result.test_contaminated = None
    test_rows = {r[3]: r[4] for r in rows if r[1] == "prueba"}
    assert test_rows["contaminada"] == expected
    assert test_rows["solicitada"] == "no" and test_rows["ejecutada"] == "no"
    assert test_rows["configuraciones_en_historial"] == "0"


def test_export_has_no_variable_timestamps(historical_result, synthetic_result):
    today = date.today().isoformat()
    for result in (historical_result, synthetic_result):
        data = export_zip_bytes(result)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            assert [i.filename for i in infos] == list(export_files(result))
            for info in infos:
                assert info.date_time == ZIP_DATE_TIME
                assert info.compress_type == zipfile.ZIP_DEFLATED
                assert today not in zf.read(info).decode("utf-8")
        assert export_zip_bytes(result) == data


def test_statistics_file_has_both_directions_with_raw_and_adjusted_p(historical_result):
    rows = csv_rows(export_files(historical_result)["pruebas_estadisticas.csv"])
    eg = [r for r in rows if r[1] == "engle_granger" and r[3] in ("p_valor", "p_valor_ajustado")]
    assert len(eg) == 4  # 2 direcciones × (original, ajustado)
    sections = {r[1] for r in rows[1:]}
    assert {
        "validacion",
        "estado",
        "integracion",
        "engle_granger",
        "estabilidad_eg",
        "hedge_ratio",
        "ou",
        "metricas",
    } <= sections


def test_daily_and_trades_files_match_the_backtests(historical_result):
    files = export_files(historical_result)
    daily = csv_rows(files["diario.csv"])
    n_rows = sum(len(bt.daily) for bt in historical_result.backtests.values())
    assert len(daily) - 1 == n_rows
    assert {r[1] for r in daily[1:]} == {Segment.TRAIN.value, Segment.VALIDATION.value}
    trades = csv_rows(files["operaciones.csv"])
    assert len(trades) - 1 == sum(len(bt.trades) for bt in historical_result.backtests.values())


def test_export_run_writes_the_same_bytes(historical_result, tmp_path):
    paths = export_run(historical_result, tmp_path / "salida")
    files = export_files(historical_result)
    assert [p.name for p in paths] == list(files)
    for path in paths:
        assert path.read_bytes() == files[path.name]


def test_blocked_analysis_still_exports(tmp_path):
    from mrlab.analysis.pair_pipeline import analyze_pair
    from mrlab.analysis.runner import RunResult
    from mrlab.reporting.run_config import PairSpec, RunConfig, SeriesRef

    log_a, log_b = cointegrated_log_prices(n=1000, seed=1)
    a, b = series_pair(log_a, log_b, currency_b="EUR")
    analysis = analyze_pair(a, b)
    cfg = RunConfig(
        pair=PairSpec(
            a=SeriesRef(provider="test", symbol="AAA"), b=SeriesRef(provider="test", symbol="BBB")
        ),
        data_origin=DataOrigin.HISTORICAL,
    )
    result = RunResult(config=cfg, analysis=analysis, series={"a": a, "b": b})
    files = export_files(result)
    assert csv_rows(files["operaciones.csv"]) == [csv_rows(files["operaciones.csv"])[0]]
    readme = files["LEEME.txt"].decode("utf-8")
    assert "⛔ Datos insuficientes" in readme and "EUR" in readme
    assert "el análisis se detuvo antes" in readme


# --------------------------------------------------------------------------- campos en español

#: Campos del OU que se exportan con su símbolo matemático (iguales en ambos idiomas).
MATH_FIELDS = {"a", "b", "s2", "nobs", "dt", "mu", "sigma", "sigma_eq"}


def test_statistics_file_names_every_ou_field_in_spanish(historical_result):
    # Hallazgo de integración: df_crit, df_alpha y df_estimated_residual (decisión O7) salían
    # con su nombre en inglés en pruebas_estadisticas.csv.
    from mrlab.model.ou import OUFit

    rows = csv_rows(export_files(historical_result)["pruebas_estadisticas.csv"])
    ou_fields = {r[3] for r in rows if r[1] == "ou"}
    english = (set(OUFit.model_fields) - MATH_FIELDS) & ou_fields
    assert not english, f"campos del OU sin traducir: {sorted(english)}"
    assert {
        "estadistico_dickey_fuller",
        "valor_critico_dickey_fuller",
        "nivel_dickey_fuller",
        "valores_criticos_residuo_estimado",
    } <= ou_fields


def test_statistics_file_reports_non_evaluable_stability_windows(tmp_path):
    # Una ventana de estabilidad no evaluable (tramo plano) se exporta con su motivo.
    log_a, log_b = cointegrated_log_prices(n=1500, seed=11, lam=0.1)
    log_a[300:600], log_b[300:600] = log_a[299], log_b[299]
    data = service(tmp_path / "cache", csv=csv_provider(tmp_path / "uploads", log_a, log_b))
    result = run_pair(pair_config("csv"), data)
    rows = csv_rows(export_files(result)["pruebas_estadisticas.csv"])
    windows = [r for r in rows if r[1] == "estabilidad_eg" and r[2].startswith("ventana")]
    evaluable = {r[2]: r[4] for r in windows if r[3] == "evaluable"}
    reasons = {r[2]: r[4] for r in windows if r[3] == "motivo"}
    assert set(evaluable.values()) == {"sí", "no"}
    for element, flag in evaluable.items():
        if flag == "no":
            assert "constante" in reasons[element]
        else:
            assert reasons[element] == ""
