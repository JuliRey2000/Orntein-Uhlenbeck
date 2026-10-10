"""Meta: estado del servicio, proveedores, ejemplos y proxies sugeridos.

CONTRATO (a implementar):
- GET /health → HealthOut {status: "ok", version, code_version (git_version()), disclaimer}.
- GET /providers → list[ProviderOut] {name, description (español), is_default}. NUNCA
  incluir claves ni valores de entorno (solo nombres de proveedor y descripciones).
- GET /examples → list[PresetPair] (mrlab.data.presets.PRESET_PAIRS), cada uno indicando si es
  sintético.
- GET /proxies → list[ProxySuggestion] (mrlab.data.proxies.PROXY_SUGGESTIONS).
- GET /glossary (opcional) → textos de conceptos (stats.correlation.CONCEPTS, FDR_EXPLANATION,
  METRIC_HELP, EXECUTION_CONVENTION, DIVIDENDS_NOTE) para tooltips del frontend.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["meta"])
