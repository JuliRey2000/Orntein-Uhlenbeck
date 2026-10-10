"""La demo reproducible (criterio de terminado de la fase 1) se ejecuta de punta a punta."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import cointegrated_log_prices

from mrlab import DISCLAIMER
from mrlab.config import REPO_ROOT

DEMO = REPO_ROOT / "examples" / "demo_ko_pep.py"
GENERATOR = REPO_ROOT / "examples" / "generar_csv_sinteticos.py"
EXAMPLE_DATA = REPO_ROOT / "examples" / "data"
SYNTHETIC_FOLDER = "SINTETICO_SINT-COI-A_SINT-COI-B"
EXPORTED = (
    "operaciones.csv",
    "diario.csv",
    "pruebas_estadisticas.csv",
    "run_config.json",
    "LEEME.txt",
)
#: Fragmentos que delatan mensajes crudos de Python o de pydantic (en inglés).
ENGLISH_TRACES = (
    "Traceback",
    "validation error",
    "Input should",
    "Field required",
    "Invalid JSON",
    "For further information",
    "usage:",
    "invalid choice",
)


def run(script: Path, *args: str | Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "GIT_COMMIT": "prueba-demo"}
    return subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=REPO_ROOT,
        timeout=180,
    )


def run_demo(*args: str | Path) -> subprocess.CompletedProcess:
    return run(DEMO, *args)


def reproduce_args(stdout: str) -> list[str]:
    """Argumentos de la orden «Para reproducir» que imprime la demo (sin «python <script>»)."""
    line = next(ln for ln in stdout.splitlines() if "Para reproducir" in ln)
    parts = shlex.split(line.split("byte a byte):", 1)[1])
    assert parts[:2] == ["python", "examples/demo_ko_pep.py"]
    return parts[2:]


def assert_same_files(original: Path, copy: Path) -> None:
    names = sorted(p.name for p in original.iterdir())
    assert names == sorted(p.name for p in copy.iterdir())
    for name in names:
        assert (copy / name).read_bytes() == (original / name).read_bytes(), name


def assert_no_english_trace(text: str) -> None:
    found = [fragment for fragment in ENGLISH_TRACES if fragment in text]
    assert not found, f"mensaje con texto crudo en inglés: {found}\n{text}"


def write_csv(path: Path, log_prices: np.ndarray, start: str = "2019-01-01") -> Path:
    dates = pd.bdate_range(start, periods=len(log_prices))
    lines = ["date,close"]
    lines += [
        f"{d.date().isoformat()},{v:.6f}" for d, v in zip(dates, np.exp(log_prices), strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- flujo básico


def test_demo_runs_on_synthetic_csv_and_exports(tmp_path):
    out, cache = tmp_path / "salida", tmp_path / "cache"
    proc = run_demo("--output-dir", out, "--cache-dir", cache)
    assert proc.returncode == 0, proc.stderr
    stdout = proc.stdout
    assert DISCLAIMER in stdout
    assert "SINTÉTICO" in stdout
    assert "✅ Evidencia de cointegración en la ventana evaluada" in stdout
    assert "Engle–Granger" in stdout and "p ajustado" in stdout
    assert "Vida media" in stdout and "Formación" in stdout and "Validación" in stdout
    assert "Prueba: NO ejecutada" in stdout  # la prueba solo con --incluir-prueba
    assert "Prueba contaminada: no aplica: la prueba no se ejecutó." in stdout
    assert "Dickey–Fuller del AR(1)" in stdout and "se rechaza la raíz unitaria" in stdout
    assert "Semilla del generador: 7" in stdout

    folder = out / SYNTHETIC_FOLDER
    names = sorted(p.name for p in folder.iterdir())
    assert names == sorted(f"SINTETICO_{n}" for n in EXPORTED)
    config = json.loads((folder / "SINTETICO_run_config.json").read_text(encoding="utf-8"))
    assert config["seeds"] == {"SINT-COI-A": 7, "SINT-COI-B": 7}
    assert config["include_test"] is False

    # Reproducir el run_config.json exportado (solo caché) produce los mismos archivos.
    replay_out = tmp_path / "reproduccion"
    proc2 = run_demo(*reproduce_args(stdout), "--output-dir", replay_out)
    assert proc2.returncode == 0, proc2.stderr
    assert_same_files(folder, replay_out / folder.name)


def test_example_data_matches_the_generator_byte_for_byte(tmp_path):
    # Los CSV de ejemplo llevan tipo_datos=SINTETICO en cada fila y la semilla en el sidecar,
    # y el generador los reproduce exactamente.
    proc = run(GENERATOR, tmp_path)
    assert proc.returncode == 0, proc.stderr
    generated = sorted(p.name for p in tmp_path.iterdir())
    assert generated == sorted(p.name for p in EXAMPLE_DATA.glob("SINTETICO_*"))
    for name in generated:
        assert (tmp_path / name).read_bytes() == (EXAMPLE_DATA / name).read_bytes(), name
    rows = (EXAMPLE_DATA / "SINTETICO_SINT-COI-A.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "date,close,tipo_datos"
    assert all(row.endswith(",SINTETICO") for row in rows[1:])
    meta = json.loads((EXAMPLE_DATA / "SINTETICO_SINT-COI-A.meta.json").read_text("utf-8"))
    assert meta["seed"] == 7 and meta["origin"] == "SINTETICO"


# --------------------------------------------------------------------------- datos sintéticos


@pytest.mark.parametrize("keep_name", [False, True], ids=["renombrado", "nombre_original"])
def test_synthetic_csv_without_sidecar_is_still_synthetic(tmp_path, keep_name):
    """Hallazgo datos #1: copiar solo el CSV y declararlo como histórico no lo vuelve histórico.

    Renombrado, lo detecta la columna tipo_datos; con el nombre original y SIN esa columna
    (formato anterior de los ejemplos), lo detecta el prefijo SINTETICO_ del nombre."""
    inputs = tmp_path / "entrada"
    inputs.mkdir()
    paths = []
    for leg in ("A", "B"):
        source = EXAMPLE_DATA / f"SINTETICO_SINT-COI-{leg}.csv"
        if keep_name:
            rows = source.read_text(encoding="utf-8").splitlines()
            content = "\n".join(",".join(r.split(",")[:2]) for r in rows) + "\n"
            target = inputs / source.name
        else:
            content = source.read_text(encoding="utf-8")
            target = inputs / f"serie_{leg.lower()}.csv"
        target.write_text(content, encoding="utf-8")
        paths.append(target)
    out = tmp_path / "salida"
    proc = run_demo(
        "--csv-a", paths[0], "--csv-b", paths[1],
        "--currency", "USD", "--asset-class", "accion", "--price-type", "cierre_ajustado",
        "--output-dir", out, "--cache-dir", tmp_path / "cache",
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert "*** DATOS SINTÉTICOS" in proc.stdout
    reason = "el nombre del archivo empieza por" if keep_name else "la columna «tipo_datos»"
    assert reason in proc.stdout
    (folder,) = out.iterdir()
    assert folder.name.startswith("SINTETICO_")
    for path in folder.iterdir():
        assert path.name.startswith("SINTETICO_")
        text = path.read_text(encoding="utf-8")
        assert "HISTORICO" not in text.replace("HISTORICO o SINTETICO", "")
    config = json.loads(next(folder.glob("*run_config.json")).read_text(encoding="utf-8"))
    assert config["data_origin"] == "SINTETICO"


# --------------------------------------------------------------------------- errores en español


def test_demo_requires_metadata_for_csv_without_sidecar(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    for path in (csv_a, csv_b):
        path.write_text("date,close\n2020-01-02,10\n", encoding="utf-8")
    proc = run_demo("--csv-a", csv_a, "--csv-b", csv_b, "--cache-dir", tmp_path / "cache")
    assert proc.returncode != 0
    assert "--currency" in proc.stderr and "--asset-class" in proc.stderr
    assert_no_english_trace(proc.stderr)


@pytest.mark.parametrize(
    ("sidecar", "flags", "fragment"),
    [
        (None, ("--currency", "US"), "«currency» (moneda)"),
        ('{\n  "symbol": "A",\n', (), "no es un JSON válido"),
        (
            '{"symbol": "A", "name": "A", "currency": "USD", "asset_class": "bono"}',
            (),
            "valores válidos: accion",
        ),
        ('{"symbol": "A", "name": "A"}', (), "falta el campo «currency» (moneda)"),
    ],
    ids=["moneda_de_2_letras", "sidecar_truncado", "clase_desconocida", "faltan_campos"],
)
def test_invalid_metadata_is_a_spanish_message_without_traceback(
    tmp_path, sidecar, flags, fragment
):
    """Hallazgo datos #9: antes la demo terminaba con un traceback de pydantic en inglés."""
    csv_a, csv_b = tmp_path / "a.csv", tmp_path / "b.csv"
    for path in (csv_a, csv_b):
        path.write_text("date,close\n2020-01-02,10\n2020-01-03,11\n", encoding="utf-8")
    if sidecar is not None:
        (tmp_path / "a.meta.json").write_text(sidecar, encoding="utf-8")
    proc = run_demo(
        "--csv-a", csv_a, "--csv-b", csv_b,
        "--currency", "USD", "--asset-class", "accion", "--price-type", "cierre",
        *flags,
        "--output-dir", tmp_path / "salida", "--cache-dir", tmp_path / "cache",
    )  # fmt: skip
    assert proc.returncode == 2
    assert "ERROR (CSV_METADATOS_INVALIDOS)" in proc.stderr
    assert fragment in proc.stderr
    assert_no_english_trace(proc.stderr)
    assert not (tmp_path / "salida").exists()


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (("--asset-class", "bono"), "opción no válida"),
        (("--start", "2020-13-01"), "fecha no válida «2020-13-01»"),
        (("--reproducir", "x.json", "--csv-a", "a.csv"), "--reproducir no admite --csv-a"),
        (("--exec-a", "x.csv"), "úsalos junto con --csv-a y --csv-b"),
    ],
    ids=["clase_no_valida", "fecha_no_valida", "reproducir_con_csv", "exec_sin_csv"],
)
def test_argument_errors_are_in_spanish(tmp_path, args, fragment):
    proc = run_demo(*args, "--cache-dir", tmp_path / "cache", "--output-dir", tmp_path / "out")
    assert proc.returncode == 2
    assert fragment in proc.stderr
    assert_no_english_trace(proc.stderr)


# --------------------------------------------------------------------------- prueba y reproducción


def test_reproducing_a_run_with_the_test_gives_the_same_bytes_and_edits_are_contaminated(
    tmp_path,
):
    """Hallazgos datos #2 y backtest #1: la orden «Para reproducir» (sin --incluir-prueba)
    vuelve a ejecutar la prueba porque el run_config.json la incluye; editar los umbrales a
    mano y volver a ejecutar marca la prueba como contaminada."""
    out, cache = tmp_path / "salida", tmp_path / "cache"
    proc = run_demo("--incluir-prueba", "--excel-es", "--output-dir", out, "--cache-dir", cache)
    assert proc.returncode == 0, proc.stderr
    assert "Prueba (fuera de muestra, configuración congelada)" in proc.stdout
    clean = "Prueba contaminada: no (según el historial registrado en run_config.json"
    assert clean in proc.stdout
    args = reproduce_args(proc.stdout)
    assert "--incluir-prueba" not in args and "--excel-es" in args
    folder = out / SYNTHETIC_FOLDER
    config_path = folder / "SINTETICO_run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["include_test"] is True and len(config["test_history"]) == 1

    replay = tmp_path / "reproduccion"
    proc2 = run_demo(*args, "--output-dir", replay)
    assert proc2.returncode == 0, proc2.stderr
    assert clean in proc2.stdout
    assert_same_files(folder, replay / folder.name)

    # Umbrales editados a mano (sin hash) y nueva ejecución: la prueba queda contaminada.
    config["strategy"]["entry_z"] = 1.5
    del config["config_hash"]
    edited = tmp_path / "editado.json"
    edited.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    proc3 = run_demo(
        "--reproducir", edited, "--output-dir", tmp_path / "editada", "--cache-dir", cache
    )
    assert proc3.returncode == 0, proc3.stderr
    assert "Prueba (CONTAMINADA" in proc3.stdout
    assert "Prueba contaminada: SÍ" in proc3.stdout
    readme = next((tmp_path / "editada").glob("*/SINTETICO_LEEME.txt")).read_text("utf-8")
    assert "Prueba contaminada: SÍ" in readme


# --------------------------------------------------------------------------- índices y proxy


@pytest.fixture
def index_csvs(tmp_path):
    """Índice IDX (no ejecutable), acción STK cointegrada con él y un ETF que replica el índice."""
    log_idx, log_stk = cointegrated_log_prices(n=1200, seed=21, lam=0.1, sigma=0.012)
    rng = np.random.default_rng(5)
    log_etf = log_idx + np.log(0.1) + rng.normal(0.0, 0.0005, len(log_idx))
    folder = tmp_path / "entrada"
    folder.mkdir()
    return {
        "idx": write_csv(folder / "IDX.csv", log_idx + np.log(30)),
        "stk": write_csv(folder / "STK.csv", log_stk),
        "etf": write_csv(folder / "ETF.csv", log_etf + np.log(30)),
    }


INDEX_FLAGS = ("--currency", "USD", "--asset-class", "accion", "--asset-class-a", "indice")


def test_index_without_proxy_blocks_the_strategy_with_proxy_required(tmp_path, index_csvs):
    """Hallazgo datos #6: un índice sin proxy ya no se simula; el análisis se muestra igual."""
    out = tmp_path / "salida"
    proc = run_demo(
        "--csv-a", index_csvs["idx"], "--csv-b", index_csvs["stk"], *INDEX_FLAGS,
        "--price-type", "cierre", "--output-dir", out, "--cache-dir", tmp_path / "cache",
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    stdout = proc.stdout
    assert "clase indice (NO ejecutable)" in stdout
    assert "«IDX» es un índice" in stdout  # razón de descarte PROXY_REQUERIDO
    assert "Bloqueo de la simulación: PROXY_REQUERIDO." in stdout
    assert "--exec-a" in stdout
    assert "Engle–Granger" in stdout and "Vida media" in stdout  # el análisis se muestra
    stats = next(out.glob("*/pruebas_estadisticas.csv")).read_text(encoding="utf-8")
    assert "PROXY_REQUERIDO" in stats and "estrategia_permitida,no" in stats


def test_index_with_executable_proxy_runs_the_backtest(tmp_path, index_csvs):
    out, cache = tmp_path / "salida", tmp_path / "cache"
    common = (
        "--csv-a", index_csvs["idx"], "--csv-b", index_csvs["stk"], *INDEX_FLAGS,
        "--price-type", "cierre", "--exec-a", index_csvs["etf"],
        "--output-dir", out, "--cache-dir", cache,
    )  # fmt: skip
    missing = run_demo(*common)  # la clase del ejecutable no se hereda de --asset-class
    assert missing.returncode == 2 and "--asset-class-exec-a" in missing.stderr
    assert_no_english_trace(missing.stderr)

    proc = run_demo(*common, "--asset-class-exec-a", "etf")
    assert proc.returncode == 0, proc.stderr
    stdout = proc.stdout
    assert "Instrumento ejecutable de A — ETF" in stdout
    assert "Se analiza «IDX» y se opera «ETF»" in stdout  # riesgo de base
    assert "Validación (fuera de muestra)" in stdout and "Backtest NO ejecutado" not in stdout
    folder = next(out.iterdir())
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    assert config["pair"]["exec_a"]["symbol"] == "ETF" and config["pair"]["exec_a"]["snapshot_id"]
    trades = (folder / "operaciones.csv").read_text(encoding="utf-8")
    assert "ETF" in trades and "IDX" not in trades  # se opera el proxy, no el índice

    # La configuración con proxy también se reproduce byte a byte desde la caché.
    replay = tmp_path / "reproduccion"
    proc2 = run_demo(*reproduce_args(stdout), "--output-dir", replay)
    assert proc2.returncode == 0, proc2.stderr
    assert_same_files(folder, replay / folder.name)
