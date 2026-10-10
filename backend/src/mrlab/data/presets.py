"""Pares de ejemplo precargados. NO se asume que ninguno esté cointegrado."""

from __future__ import annotations

from pydantic import BaseModel


class PresetPair(BaseModel):
    key: str
    provider: str
    symbol_a: str
    symbol_b: str
    title: str
    description: str
    synthetic: bool = False


PRESET_PAIRS: list[PresetPair] = [
    PresetPair(
        key="ko_pep",
        provider="yahoo",
        symbol_a="KO",
        symbol_b="PEP",
        title="Coca-Cola / PepsiCo",
        description="Dos empresas del mismo sector. Ejemplo clásico de manual: compruébalo, no lo asumas.",
    ),
    PresetPair(
        key="spy_gld",
        provider="yahoo",
        symbol_a="SPY",
        symbol_b="GLD",
        title="S&P 500 (SPY) / Oro (GLD)",
        description="Activos de naturaleza económica distinta. Útil para ver un par que probablemente se descarta.",
    ),
    PresetPair(
        key="qqq_btc",
        provider="yahoo",
        symbol_a="QQQ",
        symbol_b="BTC-USD",
        title="Nasdaq 100 (QQQ) / Bitcoin",
        description="Par mixto bolsa + cripto: calendario de bolsa y desfase entre horas de cierre.",
    ),
    PresetPair(
        key="xom_cvx",
        provider="yahoo",
        symbol_a="XOM",
        symbol_b="CVX",
        title="ExxonMobil / Chevron",
        description="Petroleras integradas de EE. UU. Otro candidato habitual que hay que verificar.",
    ),
    PresetPair(
        key="sint_coi",
        provider="synthetic",
        symbol_a="SINT-COI-A",
        symbol_b="SINT-COI-B",
        title="SINTÉTICO — par cointegrado",
        description="Par generado con un spread OU conocido. Sirve para practicar el flujo completo.",
        synthetic=True,
    ),
]
