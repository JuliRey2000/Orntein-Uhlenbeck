"""Trabajos en segundo plano.

CONTRATO (a implementar):
- GET /jobs → list[JobSummary].
- GET /jobs/{id} → Job (con result cuando está completado). Inexistente → 404 TRABAJO_NO_ENCONTRADO.
- DELETE /jobs/{id} → Job (cancelación; si ya terminó, 409 en español).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["trabajos"])
