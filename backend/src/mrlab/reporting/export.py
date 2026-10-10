"""Exportación determinista (sección 15): misma configuración + mismos datos = mismos bytes.

CONTRATO (a implementar):
- CSV estándar: separador ",", punto decimal, UTF-8 sin BOM, fin de línea LF,
  floats con ``format(x, ".12g")``, NaN/None → celda vacía. Opción
  ``excel_es=True``: separador ";" y coma decimal (para Excel en español).
- Encabezados en español. TODA fila lleva la columna ``tipo_datos``
  (HISTORICO | SINTETICO). Los archivos de datos sintéticos llevan prefijo
  ``SINTETICO_``.
- Archivos: ``operaciones.csv`` (una fila por operación y segmento),
  ``diario.csv`` (serie diaria de posiciones, PnL y capital por segmento),
  ``pruebas_estadisticas.csv`` (ADF/KPSS por serie, EG por dirección con p
  original y ajustado, estabilidad por ventana, OU), ``run_config.json``,
  ``LEEME.txt`` (convención de ejecución, aviso educativo, dividendos incluidos
  en el precio ajustado, si hay prueba contaminada).
- Sin marcas de tiempo de generación dentro de los archivos.
- ``export_zip_bytes``: ZIP con entradas en orden fijo y ``ZipInfo.date_time``
  fijo (1980, 1, 1, 0, 0, 0), compresión DEFLATED.
"""

from __future__ import annotations

from pathlib import Path

from mrlab.analysis.runner import RunResult


def trades_csv(result: RunResult, excel_es: bool = False) -> str:
    raise NotImplementedError


def daily_csv(result: RunResult, excel_es: bool = False) -> str:
    raise NotImplementedError


def stats_csv(result: RunResult, excel_es: bool = False) -> str:
    raise NotImplementedError


def readme_txt(result: RunResult) -> str:
    raise NotImplementedError


def export_files(result: RunResult, excel_es: bool = False) -> dict[str, bytes]:
    """{nombre_archivo: contenido} en orden fijo."""
    raise NotImplementedError


def export_run(result: RunResult, out_dir: Path | str, excel_es: bool = False) -> list[Path]:
    raise NotImplementedError


def export_zip_bytes(result: RunResult, excel_es: bool = False) -> bytes:
    raise NotImplementedError
