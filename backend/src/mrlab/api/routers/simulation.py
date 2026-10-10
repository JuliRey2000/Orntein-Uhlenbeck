"""Simulación SINTÉTICA del OU (modo educativo, sección 6.3).

CONTRATO (a implementar):
- POST /ou/simulate  SimulateRequest {lam, mu, sigma, x0, horizon (≤ MAX_SIM_HORIZON),
  n_paths (1..500), seed, paths_returned: int = 50 (máx. trayectorias devueltas para dibujar)}
  → SimulationOut: el resumen de mrlab.model.ou.simulate_paths_summary calculado con TODAS las
  trayectorias, pero ``paths`` recortado a paths_returned (indicar n_paths_returned y una nota en
  español); label "SINTÉTICO", synthetic=True, disclaimer. Además, el histograma del tiempo de
  cruce: crossing_hist {bins, counts} y n_not_crossed.
- La simulación NUNCA toca datos históricos ni la caché.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["simulación"])
