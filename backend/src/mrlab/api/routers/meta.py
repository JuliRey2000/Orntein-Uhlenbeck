"""Meta: estado del servicio, proveedores, ejemplos, proxies sugeridos y glosario.

CONTRATO:
- GET /health → HealthOut {status: "ok", version, code_version (git_version()), disclaimer}.
- GET /providers → list[ProviderOut] {name, label, description (español), kind, is_default,
  synthetic, notices}. NUNCA incluye claves ni valores de entorno: solo se leen ``name`` y
  ``description`` de cada proveedor, y si la descripción contuviera el valor de un atributo o de
  una variable de entorno con aspecto de secreto (key, secret, token, password…) se oculta.
- GET /examples → list[ExampleOut] (mrlab.data.presets.PRESET_PAIRS), cada uno indicando si es
  sintético y si su proveedor está habilitado.
- GET /proxies → list[ProxySuggestionOut] (mrlab.data.proxies.PROXY_SUGGESTIONS).
- GET /glossary → textos de conceptos para tooltips del frontend.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from mrlab import DISCLAIMER, __version__
from mrlab.analysis.pair_pipeline import STATUS_LABELS
from mrlab.api.openapi_models import ExampleOut, ProxySuggestionOut
from mrlab.api.schemas import ERROR_RESPONSES
from mrlab.api.state import AppState, get_state
from mrlab.backtest.costs import DIVIDENDS_NOTE
from mrlab.backtest.engine import EXECUTION_CONVENTION
from mrlab.backtest.metrics import METRIC_HELP
from mrlab.data.presets import PRESET_PAIRS
from mrlab.data.providers.csv_provider import CSVProvider
from mrlab.data.providers.synthetic import SYNTHETIC_NOTE, SyntheticProvider
from mrlab.data.providers.yahoo import YAHOO_NOTICE
from mrlab.data.proxies import PROXY_SUGGESTIONS
from mrlab.domain.models import PRICE_TYPE_LABELS
from mrlab.reporting.run_config import git_version
from mrlab.stats.correlation import CONCEPTS
from mrlab.stats.integration import INTEGRATION_LABELS
from mrlab.stats.multitest import FDR_EXPLANATION

router = APIRouter(tags=["meta"])

OK = "Respuesta correcta."

#: Nombres de atributo o de variable de entorno que pueden contener secretos.
_SECRET_NAME = re.compile(r"key|secret|token|passw|credential|auth", re.IGNORECASE)
_MIN_SECRET_LEN = 8
_MAX_DESCRIPTION = 500
HIDDEN = "[oculto]"

PROVIDER_LABELS = {"yahoo": "Yahoo Finance", "csv": "Archivos CSV", "synthetic": "Datos SINTÉTICOS"}
CSV_NOTICE = (
    "Moneda, clase de activo y tipo de precio los declara el usuario al cargar el archivo. Un CSV "
    "con la columna tipo_datos = SINTETICO o con nombre SINTETICO_… se trata como SINTÉTICO."
)
CORRELATION_NOTE = (
    "La correlación es un filtro exploratorio opcional: nunca prueba la cointegración. Dos activos "
    "muy correlacionados pueden no estar cointegrados, y al revés."
)


class HealthOut(BaseModel):
    status: Literal["ok"] = Field("ok", description="«ok» si el servicio responde.")
    version: str = Field(description="Versión del laboratorio.")
    code_version: str = Field(
        description=(
            "Versión del código: hash de git (con «-dirty» si hay cambios sin confirmar), la "
            "variable GIT_COMMIT en Docker, o «desconocido»."
        )
    )
    disclaimer: str = Field(DISCLAIMER, description="Aviso educativo obligatorio.")


class ProviderOut(BaseModel):
    name: str = Field(description="Nombre del proveedor (se usa en las peticiones).")
    label: str = Field(description="Nombre para mostrar.")
    description: str = Field(description="Descripción en español.")
    kind: Literal["en_linea", "archivo", "sintetico"] = Field(
        description=(
            "«en_linea»: descarga datos de un servicio externo; «archivo»: CSV cargado por el "
            "usuario; «sintetico»: datos generados con semilla (nunca históricos)."
        )
    )
    is_default: bool = Field(description="True si es el proveedor por defecto.")
    synthetic: bool = Field(description="True si el proveedor genera datos SINTÉTICOS.")
    notices: list[str] = Field(
        default_factory=list, description="Avisos para mostrar junto al proveedor (en español)."
    )


class GlossaryOut(BaseModel):
    disclaimer: str = Field(DISCLAIMER, description="Aviso educativo obligatorio.")
    concepts: dict[str, str] = Field(
        description="Correlación, cointegración y aptitud para OU: tres conceptos distintos."
    )
    correlation_note: str = Field(description="La correlación nunca prueba la cointegración.")
    fdr_explanation: str = Field(description="Por qué se ajustan los p-values (FDR).")
    status_labels: dict[str, str] = Field(description="Etiqueta de cada estado posible de un par.")
    integration_labels: dict[str, str] = Field(
        description="Interpretación combinada ADF + KPSS de cada serie."
    )
    price_type_labels: dict[str, str] = Field(description="Etiqueta de cada tipo de precio.")
    metric_help: dict[str, str] = Field(description="Ayuda de una línea por métrica del backtest.")
    execution_convention: str = Field(description="Convención de ejecución (señal t, orden t+1).")
    dividends_note: str = Field(description="Por qué los dividendos no se suman aparte.")
    yahoo_notice: str = Field(description="Aviso sobre Yahoo Finance (fuente no oficial).")
    synthetic_note: str = Field(description="Aviso que acompaña a todo dato SINTÉTICO.")


# --------------------------------------------------------------------------- helpers


@lru_cache(maxsize=1)
def _code_version() -> str:
    """Versión del código en ejecución (se calcula una vez por proceso)."""
    return git_version()


def _secret_values(provider: Any) -> list[str]:
    """Valores de atributos y variables de entorno cuyo nombre parece de un secreto."""
    values: list[str] = []
    sources: list[dict[str, Any]] = [dict(os.environ)]
    for source in (vars(type(provider)), getattr(provider, "__dict__", {})):
        sources.append(dict(source))
    for source in sources:
        for name, value in source.items():
            if (
                isinstance(name, str)
                and _SECRET_NAME.search(name)
                and isinstance(value, str)
                and len(value.strip()) >= _MIN_SECRET_LEN
            ):
                values.append(value.strip())
    return sorted(set(values), key=len, reverse=True)


def safe_description(provider: Any) -> str:
    """Descripción del proveedor sin secretos (y acotada)."""
    raw = getattr(provider, "description", "")
    text = raw.strip() if isinstance(raw, str) else ""
    for secret in _secret_values(provider):
        text = text.replace(secret, HIDDEN)
    if len(text) > _MAX_DESCRIPTION:
        text = text[: _MAX_DESCRIPTION - 1] + "…"
    return text


def _kind(provider: Any) -> Literal["en_linea", "archivo", "sintetico"]:
    if isinstance(provider, SyntheticProvider):
        return "sintetico"
    if isinstance(provider, CSVProvider):
        return "archivo"
    return "en_linea"


def _notices(name: str, kind: str) -> list[str]:
    if name == "yahoo":
        return [YAHOO_NOTICE, DIVIDENDS_NOTE]
    if kind == "sintetico":
        return [SYNTHETIC_NOTE]
    if kind == "archivo":
        return [CSV_NOTICE]
    return ["Fuente de datos externa: puede fallar o limitar peticiones; los datos se guardan "
            "en la caché local para poder reproducir cada análisis."]


# --------------------------------------------------------------------------- endpoints


@router.get(
    "/health",
    response_model=HealthOut,
    summary="Estado del servicio",
    description="Comprueba que la API responde y devuelve la versión del laboratorio y del código.",
    response_description=OK,
    responses={500: ERROR_RESPONSES[500]},
)
def health() -> HealthOut:
    return HealthOut(version=__version__, code_version=_code_version())


@router.get(
    "/providers",
    response_model=list[ProviderOut],
    summary="Proveedores de datos",
    description=(
        "Lista los proveedores habilitados con su descripción y avisos (p. ej. que Yahoo Finance "
        "es una fuente gratuita no oficial). Nunca incluye claves de API ni variables de entorno."
    ),
    response_description=OK,
    responses={500: ERROR_RESPONSES[500]},
)
def providers(state: AppState = Depends(get_state)) -> list[ProviderOut]:
    out: list[ProviderOut] = []
    for name in state.registry.names():
        provider = state.registry.get(name)
        kind = _kind(provider)
        out.append(
            ProviderOut(
                name=name,
                label=PROVIDER_LABELS.get(name, name),
                description=safe_description(provider),
                kind=kind,
                is_default=name == state.registry.default_name,
                synthetic=kind == "sintetico",
                notices=_notices(name, kind),
            )
        )
    return out


@router.get(
    "/examples",
    response_model=list[ExampleOut],
    summary="Pares de ejemplo",
    description=(
        "Pares precargados para empezar (KO/PEP, SPY/GLD, QQQ/BTC-USD, XOM/CVX y un par "
        "SINTÉTICO). No se asume que ninguno esté cointegrado: hay que comprobarlo con el análisis."
    ),
    response_description=OK,
    responses={500: ERROR_RESPONSES[500]},
)
def examples(state: AppState = Depends(get_state)) -> list[ExampleOut]:
    enabled = set(state.registry.names())
    return [
        ExampleOut(**p.model_dump(), available=p.provider in enabled) for p in PRESET_PAIRS
    ]


@router.get(
    "/proxies",
    response_model=list[ProxySuggestionOut],
    summary="Proxies ejecutables sugeridos",
    description=(
        "Índices y futuros no ejecutables con el instrumento sugerido para simular. La sugerencia "
        "nunca se aplica en silencio: el usuario debe elegir el proxy."
    ),
    response_description=OK,
    responses={500: ERROR_RESPONSES[500]},
)
def proxies() -> list[ProxySuggestionOut]:
    return [ProxySuggestionOut(**s.model_dump()) for s in PROXY_SUGGESTIONS.values()]


@router.get(
    "/glossary",
    response_model=GlossaryOut,
    summary="Glosario",
    description="Textos en español de conceptos, estados y métricas (para tooltips).",
    response_description=OK,
    responses={500: ERROR_RESPONSES[500]},
)
def glossary() -> GlossaryOut:
    return GlossaryOut(
        concepts=dict(CONCEPTS),
        correlation_note=CORRELATION_NOTE,
        fdr_explanation=FDR_EXPLANATION,
        status_labels={str(k): v for k, v in STATUS_LABELS.items()},
        integration_labels=dict(INTEGRATION_LABELS),
        price_type_labels={str(k): v for k, v in PRICE_TYPE_LABELS.items()},
        metric_help=dict(METRIC_HELP),
        execution_convention=EXECUTION_CONVENTION,
        dividends_note=DIVIDENDS_NOTE,
        yahoo_notice=YAHOO_NOTICE,
        synthetic_note=SYNTHETIC_NOTE,
    )
