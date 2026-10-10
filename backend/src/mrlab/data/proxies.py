"""Índices/referencias no ejecutables y su proxy ejecutable sugerido (sección 2.2).

La app SUGIERE el proxy y pide confirmación; nunca lo sustituye en silencio.
"""

from __future__ import annotations

from pydantic import BaseModel


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
