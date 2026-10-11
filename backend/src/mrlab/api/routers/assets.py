"""Activos: búsqueda y metadatos.

CONTRATO:
- GET /assets/search?q=&provider= (provider por defecto = registry.default_name; q no vacío,
  máx. 100 caracteres) → list[AssetInfoOut].
- GET /assets/metadata?provider=&symbol= → AssetMetadataOut {asset: AssetInfo,
  proxy: ProxySuggestion | None, requires_proxy: bool, synthetic, note (español)}. Para índices y
  futuros requires_proxy=True y se SUGIERE el proxy; nunca se sustituye.
Los símbolos van por query (pueden contener ^ o =).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from mrlab.api.openapi_models import AssetInfoOut, ProxySuggestionOut
from mrlab.api.schemas import ERROR_RESPONSES, ErrorResponse
from mrlab.api.state import AppState, get_state
from mrlab.data.proxies import suggest_proxy
from mrlab.domain.models import NON_EXECUTABLE_CLASSES, AssetClass, AssetInfo

router = APIRouter(tags=["activos"])

OK = "Respuesta correcta."
MAX_QUERY_CHARS = 100
MAX_SYMBOL_CHARS = 40

#: Errores posibles al consultar un proveedor externo.
PROVIDER_RESPONSES = {
    **ERROR_RESPONSES,
    429: {"model": ErrorResponse, "description": "El proveedor limitó las peticiones"},
    503: {"model": ErrorResponse, "description": "Sin conexión con el proveedor de datos"},
}

PROVIDER_QUERY = Query(
    None,
    description=(
        "Proveedor de datos (p. ej. «yahoo», «csv», «synthetic»). Si se omite, el proveedor por "
        "defecto (ver /api/providers)."
    ),
)


class AssetMetadataOut(BaseModel):
    asset: AssetInfoOut = Field(description="Metadatos del activo.")
    requires_proxy: bool = Field(
        description=(
            "True si el activo no es ejecutable (índice o futuro): para simular hay que elegir "
            "explícitamente un instrumento ejecutable."
        )
    )
    proxy: ProxySuggestionOut | None = Field(
        None,
        description="Proxy SUGERIDO (solo si requires_proxy). Nunca se sustituye en silencio.",
    )
    synthetic: bool = Field(description="True si la serie es SINTÉTICA.")
    note: str = Field(description="Explicación en español para mostrar junto al activo.")


def _invalid(field: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "code": "PETICION_INVALIDA",
            "message": message,
            "details": {"campos": [{"campo": field, "motivo": message, "recibido": None}]},
        },
    )


def requires_proxy(asset: AssetInfo) -> bool:
    """No ejecutable = ``executable`` False o clase índice/futuro (defensa doble, decisión A2)."""
    return (not asset.executable) or asset.asset_class in NON_EXECUTABLE_CLASSES


def metadata_note(asset: AssetInfo) -> tuple[bool, ProxySuggestionOut | None, str]:
    """(requiere proxy, proxy sugerido, nota en español)."""
    sym = asset.symbol
    if requires_proxy(asset):
        kind = "un índice" if asset.asset_class == AssetClass.INDEX else "un futuro continuo"
        suggestion = suggest_proxy(sym)
        base = (
            f"«{sym}» es {kind}: no se puede comprar ni vender directamente. El análisis "
            "estadístico se puede hacer sobre esta serie, pero para simular la estrategia hay que "
            "elegir un instrumento ejecutable."
        )
        if suggestion is not None and suggestion.proxy.upper() != sym.upper():
            proxy = ProxySuggestionOut(**suggestion.model_dump())
            note = (
                f"{base} Se sugiere «{suggestion.proxy}», pero no se aplica automáticamente: "
                f"debes confirmarlo. {suggestion.note}"
            )
            return True, proxy, note
        return True, None, (
            f"{base} No hay un proxy sugerido: elige un instrumento ejecutable que lo replique "
            "(por ejemplo, un ETF)."
        )
    parts: list[str] = []
    if asset.is_synthetic:
        parts.append(f"«{sym}» es una serie SINTÉTICA: no son precios históricos.")
    else:
        parts.append(f"«{sym}» es ejecutable: se puede usar directamente en la simulación.")
    if asset.currency is None:
        parts.append(
            "El proveedor no informa la moneda: el análisis quedará bloqueado hasta conocerla."
        )
    if asset.asset_class == AssetClass.CRYPTO:
        parts.append(
            "Cripto spot: vender en corto requiere derivados (perpetuos con funding), fuera del "
            "alcance de la v1."
        )
    return False, None, " ".join(parts)


@router.get(
    "/assets/search",
    response_model=list[AssetInfoOut],
    summary="Buscar activos",
    description=(
        "Busca activos por símbolo o nombre en un proveedor. Los índices y futuros aparecen como "
        "no ejecutables, con un proxy SUGERIDO (nunca se sustituye en silencio)."
    ),
    response_description=OK,
    responses=PROVIDER_RESPONSES,
)
def search_assets(
    q: str = Query(
        ...,
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Texto a buscar (símbolo o parte del nombre), hasta 100 caracteres.",
    ),
    provider: str | None = PROVIDER_QUERY,
    state: AppState = Depends(get_state),
) -> list[AssetInfo]:
    query = q.strip()
    if not query:
        raise _invalid("q", "La búsqueda está vacía: escribe al menos un carácter.")
    source = state.registry.get(provider or state.registry.default_name or "")
    return source.search(query)


@router.get(
    "/assets/metadata",
    response_model=AssetMetadataOut,
    summary="Metadatos de un activo",
    description=(
        "Metadatos de un símbolo (nombre, clase, moneda, si es ejecutable). Para índices y futuros "
        "indica que hace falta un proxy ejecutable y SUGIERE uno, sin sustituirlo. El símbolo va "
        "como parámetro de consulta porque puede contener «^» o «=» (p. ej. «^GSPC», «GC=F»)."
    ),
    response_description=OK,
    responses=PROVIDER_RESPONSES,
)
def asset_metadata(
    symbol: str = Query(
        ...,
        min_length=1,
        max_length=MAX_SYMBOL_CHARS,
        description="Símbolo en el proveedor (p. ej. «KO», «^GSPC», «BTC-USD»).",
    ),
    provider: str | None = PROVIDER_QUERY,
    state: AppState = Depends(get_state),
) -> AssetMetadataOut:
    sym = symbol.strip()
    if not sym:
        raise _invalid("symbol", "El símbolo está vacío: indica uno (p. ej. «KO»).")
    source = state.registry.get(provider or state.registry.default_name or "")
    asset = source.get_metadata(sym)
    needs_proxy, proxy, note = metadata_note(asset)
    return AssetMetadataOut(
        asset=AssetInfoOut(**asset.model_dump()),
        requires_proxy=needs_proxy,
        proxy=proxy,
        synthetic=asset.is_synthetic,
        note=note,
    )
