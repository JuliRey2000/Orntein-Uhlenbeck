"""Aplicación FastAPI del Laboratorio de Reversión a la Media.

Arranque: ``uvicorn mrlab.api.main:app`` (un solo proceso: la caché y los
estudios viven en disco local y sus locks son por proceso; decisión F1).
Documentación interactiva en ``/docs`` (Swagger) y ``/redoc``; el esquema se
exporta a ``backend/openapi.json`` con ``backend/scripts/export_openapi.py``.

Importar este módulo construye la aplicación sin tocar la red ni crear carpetas:
la caché, los estudios y las subidas se crean en el primer uso.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mrlab import DISCLAIMER, __version__
from mrlab.api.errors import register_exception_handlers
from mrlab.api.openapi_docs import install_openapi
from mrlab.api.routers import (
    analysis,
    assets,
    backtest,
    data,
    exports,
    jobs,
    meta,
    simulation,
    studies,
)
from mrlab.api.state import AppState, build_state
from mrlab.config import Settings, get_settings
from mrlab.data.registry import ProviderRegistry

API_PREFIX = "/api"

DESCRIPTION = f"""
API del **Laboratorio de Reversión a la Media**: herramienta educativa para investigar y simular
pairs trading con cointegración y procesos Ornstein–Uhlenbeck.

> **{DISCLAIMER}**

Convenciones: textos para el usuario en español; fechas ISO (AAAA-MM-DD); series temporales en
formato columnar (`dates` + `values`); `NaN` se envía como `null`. Los errores siguen siempre el
formato `{{"error": {{"code", "message", "details"}}}}`.

La señal se calcula con el cierre del día t y se ejecuta al cierre del día t+1. El periodo de
**prueba** solo se ejecuta a través de un **estudio** (`/api/studies/{{id}}/test`), una vez y con la
configuración congelada; cambiar umbrales después lo marca como *prueba contaminada*.
"""

TAGS = [
    {
        "name": "meta",
        "description": "Estado del servicio, proveedores, ejemplos y proxies sugeridos.",
    },
    {"name": "activos", "description": "Búsqueda y metadatos de activos."},
    {
        "name": "datos",
        "description": "Descarga, carga de CSV, datos SINTÉTICOS, caché y validación.",
    },
    {
        "name": "análisis",
        "description": "Correlaciones, cointegración, spread y modelo OU de un par.",
    },
    {"name": "simulación", "description": "Simulación SINTÉTICA de procesos OU (modo educativo)."},
    {
        "name": "backtest",
        "description": "Backtest de formación y validación; comparación entre pares.",
    },
    {
        "name": "estudios",
        "description": "Estudios congelados: única vía para ejecutar la prueba fuera de muestra.",
    },
    {
        "name": "exportación",
        "description": "Exportación reproducible (ZIP) y reproducción desde run_config.json.",
    },
    {"name": "trabajos", "description": "Cálculos en segundo plano con progreso."},
]


def create_app(
    settings: Settings | None = None, registry: ProviderRegistry | None = None
) -> FastAPI:
    settings = settings or get_settings()
    state: AppState = build_state(settings, registry)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        state.jobs.shutdown()

    app = FastAPI(
        lifespan=lifespan,
        title="Laboratorio de Reversión a la Media — API",
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=TAGS,
    )
    app.state.lab = state
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    register_exception_handlers(app)
    for module in (meta, assets, data, analysis, simulation, backtest, studies, exports, jobs):
        app.include_router(module.router, prefix=API_PREFIX)
    install_openapi(app)
    return app


def openapi_text(application: FastAPI | None = None) -> str:
    """Esquema OpenAPI como texto JSON estable (el que se versiona en ``backend/openapi.json``).

    UTF-8 sin escapar los acentos, sangría de 2 espacios y salto de línea final: el mismo código
    produce siempre el mismo texto (no depende de la configuración local ni de la fecha)."""
    spec = (application or app).openapi()
    return json.dumps(spec, ensure_ascii=False, indent=2) + "\n"


app = create_app()
