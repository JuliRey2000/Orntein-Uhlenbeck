"""Activos: búsqueda y metadatos.

CONTRATO (a implementar):
- GET /assets/search?q=&provider= (provider por defecto = registry.default_name; q no vacío,
  máx. 100 caracteres) → list[AssetInfo].
- GET /assets/metadata?provider=&symbol= → AssetMetadataOut {asset: AssetInfo,
  proxy: ProxySuggestion | None, requires_proxy: bool, note (español)}. Para índices y futuros
  requires_proxy=True y se SUGIERE el proxy; nunca se sustituye.
Los símbolos van por query (pueden contener ^ o =).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["activos"])
