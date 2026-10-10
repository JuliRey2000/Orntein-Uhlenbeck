"""Exportación reproducible y reproducción desde run_config.json (sección 15).

CONTRATO (a implementar):
- POST /export  ExportRequest {config: RunConfig, excel_es: bool=False} → application/zip
  (export_zip_bytes) con Content-Disposition: attachment; filename="<base>.zip" (prefijo
  SINTETICO_ si hay datos sintéticos). include_test=True → 400 (la prueba se exporta desde el
  estudio). Misma configuración + misma caché → mismos bytes.
- POST /runs/replay  multipart file=run_config.json (+ excel_es) → ZIP reproducido EXACTAMENTE
  (solo caché, nunca descarga: si falta una instantánea → 404 SNAPSHOT_NO_ENCONTRADO en
  español). Un run_config con include_test=True se reproduce tal cual (es una reproducción, no
  una nueva evaluación) y conserva su test_history / marca de contaminación.
- POST /runs/replay/json  mismo, pero devuelve AnalysisOut + BacktestOut en JSON (para la UI).
- Helper público ``zip_response(result, excel_es) -> Response`` reutilizable por studies.py.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["exportación"])
