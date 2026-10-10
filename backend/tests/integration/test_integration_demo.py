"""La demo reproducible (criterio de terminado de la fase 1) se ejecuta de punta a punta."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mrlab import DISCLAIMER
from mrlab.config import REPO_ROOT

DEMO = REPO_ROOT / "examples" / "demo_ko_pep.py"


def run_demo(*args: str | Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "GIT_COMMIT": "prueba-demo"}
    return subprocess.run(
        [sys.executable, str(DEMO), *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=REPO_ROOT,
        timeout=180,
    )


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

    folder = out / "SINTETICO_SINT-COI-A_SINT-COI-B"
    names = sorted(p.name for p in folder.iterdir())
    assert names == sorted(
        f"SINTETICO_{n}"
        for n in (
            "operaciones.csv",
            "diario.csv",
            "pruebas_estadisticas.csv",
            "run_config.json",
            "LEEME.txt",
        )
    )

    # Reproducir el run_config.json exportado (solo caché) produce los mismos archivos.
    replay_out = tmp_path / "reproduccion"
    config = folder / "SINTETICO_run_config.json"
    proc2 = run_demo("--reproducir", config, "--output-dir", replay_out, "--cache-dir", cache)
    assert proc2.returncode == 0, proc2.stderr
    for path in folder.iterdir():
        assert (replay_out / folder.name / path.name).read_bytes() == path.read_bytes()


def test_demo_requires_metadata_for_csv_without_sidecar(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    for path in (csv_a, csv_b):
        path.write_text("date,close\n2020-01-02,10\n", encoding="utf-8")
    proc = run_demo("--csv-a", csv_a, "--csv-b", csv_b, "--cache-dir", tmp_path / "cache")
    assert proc.returncode != 0
    assert "--currency" in proc.stderr and "--asset-class" in proc.stderr
