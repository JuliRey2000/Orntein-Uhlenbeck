"""Dimensionamiento de las piernas (7.3).

Spread logarítmico: β relaciona VALORES NOCIONALES, no cantidades:
    N = nocional de la pierna A = G / (1 + |β|)
    q_A = ± N / P_A ;  q_B = ∓ β · N / P_B
    Exposición bruta = |q_A·P_A| + |q_B·P_B| = N·(1 + |β|) = G
Spread en precios: q_B = ∓ β · q_A en unidades:
    q_A = ± G / (P_A + |β|·P_B)

G = gross_exposure_target × capital (decidido con el capital al cierre de la
señal), acotado por max_leverage × capital (si se reduce, registrar la razón).
Signo: LARGO spread → q_A > 0; q_B = −β·(...) (si β < 0, ambas piernas largas).

CONTRATO:
- ``requires_short(side, beta)``: True si alguna pierna quedaría corta.
- ``size_position(side, beta, spread_type, price_y, price_x, equity, params)``:
  * equity ≤ 0 → executable=False (razón en español);
  * si not allow_short y requires_short → executable=False, razón explicando que
    la estrategia de pares requiere ambas piernas (y que en cripto spot el corto
    real requiere derivados, fuera del alcance de la v1);
  * integer_units: redondeo hacia cero de AMBAS cantidades (nunca supera
    límites); si alguna queda en 0 → executable=False con razón;
  * devuelve cantidades, nocionales, exposición bruta y apalancamiento resultante.
  Además: precios no positivos o β no finito → executable=False; β = 0 deja la
  pierna B en 0 unidades → executable=False (una sola pierna no es un par).
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from mrlab.domain.models import SpreadType
from mrlab.strategy.params import StrategyParams
from mrlab.strategy.state_machine import Side, side_sign

PAIRS_NEED_BOTH_LEGS = (
    "La estrategia de pares necesita ambas piernas a la vez (operar solo una no es una estrategia "
    "de pares). En cripto spot, el corto real requiere derivados, que están fuera del alcance de la v1."
)

NO_SHORT_REASON = (
    "Señal no ejecutable: requiere vender en corto al menos una pierna y las ventas en corto "
    "están desactivadas. " + PAIRS_NEED_BOTH_LEGS
)


def _fmt(x: float, decimals: int = 2) -> str:
    """Número con formato «10.000,50»."""
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


class SizingResult(BaseModel):
    executable: bool
    q_y: float = 0.0
    q_x: float = 0.0
    notional_y: float = 0.0  # con signo
    notional_x: float = 0.0  # con signo
    gross_exposure: float = 0.0
    leverage: float = 0.0
    target_gross: float = 0.0
    reduced: bool = False
    reasons: list[str] = []


def short_legs(side: Side, beta: float) -> tuple[bool, bool]:
    """(¿A queda corta?, ¿B queda corta?) para ese lado y β."""
    if side == Side.SHORT:
        return True, beta < 0
    return False, beta > 0


def requires_short(side: Side, beta: float) -> bool:
    """Corto spread siempre vende A; largo spread vende B solo si β > 0."""
    return any(short_legs(side, beta))


def size_position(
    side: Side,
    beta: float,
    spread_type: SpreadType,
    price_y: float,
    price_x: float,
    equity: float,
    params: StrategyParams,
) -> SizingResult:
    if not (math.isfinite(equity) and equity > 0):
        return SizingResult(
            executable=False,
            reasons=["El capital disponible es cero o negativo: no se abren nuevas posiciones."],
        )
    if not all(math.isfinite(p) and p > 0 for p in (price_y, price_x)):
        return SizingResult(
            executable=False,
            reasons=[
                "Algún precio de ejecución no es un número positivo: no se puede dimensionar."
            ],
        )
    if not math.isfinite(beta):
        return SizingResult(executable=False, reasons=["El hedge ratio β no es un número finito."])
    if not params.allow_short and requires_short(side, beta):
        return SizingResult(executable=False, reasons=[NO_SHORT_REASON])

    target = params.gross_exposure_target * equity
    cap = params.max_leverage * equity
    gross_goal = target
    reasons: list[str] = []
    reduced = target > cap
    if reduced:
        gross_goal = cap
        reasons.append(
            f"La exposición bruta objetivo ({_fmt(target)}) supera el apalancamiento máximo de "
            f"{_fmt(params.max_leverage, 2)}× el capital; se reduce a {_fmt(cap)}."
        )

    sign = side_sign(side)
    if spread_type == SpreadType.LOG:
        notional_a = gross_goal / (1.0 + abs(beta))
        q_y = sign * notional_a / price_y
        q_x = -sign * beta * notional_a / price_x
    else:
        q_y = sign * gross_goal / (price_y + abs(beta) * price_x)
        q_x = -beta * q_y

    if params.integer_units:
        q_y, q_x = float(math.trunc(q_y)), float(math.trunc(q_x))

    if q_y == 0 or q_x == 0:
        if beta == 0:
            why = "Con β = 0 la pierna B queda en 0 unidades"
        else:
            why = "Al redondear a unidades enteras (hacia cero) una de las piernas queda en 0 unidades"
        return SizingResult(
            executable=False,
            target_gross=target,
            reduced=reduced,
            reasons=[
                *reasons,
                f"{why}: no se opera, porque una sola pierna no es una estrategia de pares.",
            ],
        )

    notional_y = q_y * price_y
    notional_x = q_x * price_x
    gross = abs(notional_y) + abs(notional_x)
    return SizingResult(
        executable=True,
        q_y=q_y,
        q_x=q_x,
        notional_y=notional_y,
        notional_x=notional_x,
        gross_exposure=gross,
        leverage=gross / equity,
        target_gross=target,
        reduced=reduced,
        reasons=reasons,
    )
