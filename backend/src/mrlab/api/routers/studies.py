"""Estudios: congelar una configuración y ejecutar la prueba fuera de muestra (una vez).

CONTRATO (a implementar), sobre mrlab.studies.StudyStore:
- POST /studies  StudyCreate {config: RunConfig, name: str = ""} → StudyOut (201).
- GET /studies → list[StudySummaryOut] {id, name, symbols, created_at, updated_at,
  test_viewed, contaminated}.
- GET /studies/{id} → StudyOut {study (Study), test_viewed, contaminated, contamination_reason,
  note (español sobre la regla de la prueba)}.
- PUT /studies/{id}/config  StudyUpdate {strategy?, costs?, analysis?} → StudyOut. Si la prueba
  ya se vio, la respuesta incluye el aviso de contaminación.
- POST /studies/{id}/test → StudyTestOut {study: StudyOut, backtest: BacktestOut (con el
  segmento "prueba"), contaminated, contamination_text (texto veraz en español)}.
- GET /studies/{id}/export?excel_es=false → ZIP (ver exports.py) del estudio; incluye la prueba
  solo si ya se ejecutó; refleja la contaminación.
- DELETE /studies/{id} → 204.
Ids inválidos o inexistentes → 404 ESTUDIO_NO_ENCONTRADO en español.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["estudios"])
