"""Índices/referencias no ejecutables y su proxy ejecutable sugerido (sección 2.2).

La app SUGIERE el proxy y pide confirmación; nunca lo sustituye en silencio.
"""

from __future__ import annotations

from pydantic import BaseModel

from mrlab.domain.models import NON_EXECUTABLE_CLASSES, AssetClass, AssetInfo, DataOrigin


class ProxySuggestion(BaseModel):
    reference: str  # símbolo de la referencia (Yahoo)
    name: str
    proxy: str
    note: str


PROXY_SUGGESTIONS: dict[str, ProxySuggestion] = {
    s.reference: s
    for s in [
        ProxySuggestion(
            reference="^GSPC",
            name="S&P 500",
            proxy="SPY",
            note="^GSPC es un índice de precios (sin dividendos); SPY ajustado sí los incluye.",
        ),
        ProxySuggestion(
            reference="^NDX",
            name="Nasdaq 100",
            proxy="QQQ",
            note="^NDX es un índice de precios (sin dividendos); QQQ ajustado sí los incluye.",
        ),
        ProxySuggestion(
            reference="GC=F",
            name="Oro (futuro continuo)",
            proxy="GLD",
            note="GC=F es una serie de futuros con vencimientos (roll); GLD es un ETF con comisión de gestión.",
        ),
        ProxySuggestion(
            reference="BTC-USD",
            name="Bitcoin",
            proxy="BTC-USD",
            note=(
                "BTC-USD es spot y se puede usar directamente para posiciones largas. "
                "Vender en corto requiere derivados (perpetuos con funding), fuera del alcance de la v1."
            ),
        ),
        ProxySuggestion(
            reference="^DJI",
            name="Dow Jones Industrial Average",
            proxy="DIA",
            note="^DJI es un índice de precios; DIA ajustado incluye dividendos.",
        ),
        ProxySuggestion(
            reference="^RUT",
            name="Russell 2000",
            proxy="IWM",
            note="^RUT es un índice de precios; IWM ajustado incluye dividendos.",
        ),
        ProxySuggestion(
            reference="SI=F",
            name="Plata (futuro continuo)",
            proxy="SLV",
            note="SI=F es una serie de futuros con roll; SLV es un ETF.",
        ),
    ]
}


def suggest_proxy(symbol: str) -> ProxySuggestion | None:
    return PROXY_SUGGESTIONS.get(symbol.upper()) or PROXY_SUGGESTIONS.get(symbol)


def looks_non_executable(symbol: str) -> bool:
    """Heurística por símbolo de Yahoo: ``^…`` = índice, ``…=F`` = futuro."""
    s = symbol.upper()
    return s.startswith("^") or s.endswith("=F")


def build_asset_info(
    symbol: str,
    asset_class: AssetClass,
    *,
    name: str | None,
    currency: str | None,
    provider: str,
    exchange: str | None = None,
    timezone: str | None = None,
    origin: DataOrigin = DataOrigin.HISTORICAL,
    notes: tuple[str, ...] = (),
) -> AssetInfo:
    """AssetInfo con la política de proxies: índices y futuros quedan como no
    ejecutables y se SUGIERE un proxy (con la explicación en ``notes``), sin
    sustituirlo nunca en silencio."""
    suggestion = suggest_proxy(symbol)
    executable = asset_class not in NON_EXECUTABLE_CLASSES
    extra: list[str] = []
    if not executable:
        kind = "un índice" if asset_class == AssetClass.INDEX else "un futuro continuo"
        if suggestion:
            extra.append(
                f"{symbol} es {kind}: no se puede comprar ni vender directamente. Se sugiere "
                f"{suggestion.proxy} como instrumento ejecutable, pero debes confirmarlo (nunca "
                f"se sustituye en silencio). {suggestion.note}"
            )
        else:
            extra.append(
                f"{symbol} es {kind}: no se puede comprar ni vender directamente y no hay un proxy "
                "sugerido. Elige un instrumento ejecutable que lo replique (por ejemplo, un ETF)."
            )
    elif suggestion:
        extra.append(suggestion.note)
    elif asset_class == AssetClass.CRYPTO:
        extra.append(
            "Cripto spot: vender en corto requiere derivados (perpetuos con funding), fuera del "
            "alcance de la v1."
        )
    return AssetInfo(
        symbol=symbol,
        name=name or (suggestion.name if suggestion else symbol),
        asset_class=asset_class,
        currency=currency.strip() if currency and currency.strip() else None,
        provider=provider,
        exchange=exchange or None,
        timezone=timezone or None,
        origin=origin,
        executable=executable,
        suggested_proxy=suggestion.proxy if suggestion and not executable else None,
        notes=(*notes, *extra),
    )
