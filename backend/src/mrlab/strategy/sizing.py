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

CONTRATO (a implementar):
- ``requires_short(side, beta)``: True si alguna pierna quedaría corta.
- ``size_position(side, beta, spread_type, price_y, price_x, equity, params)``:
  * equity ≤ 0 → executable=False (razón en español);
  * si not allow_short y requires_short → executable=False, razón explicando que
    la estrategia de pares requiere ambas piernas (y que en cripto spot el corto
    real requiere derivados, fuera del alcance de la v1);
  * integer_units: redondeo hacia cero de AMBAS cantidades (nunca supera
    límites); si alguna queda en 0 → executable=False con razón;
  * devuelve cantidades, nocionales, exposición bruta y apalancamiento resultante.
"""

from __future__ import annotations

from pydantic import BaseModel

from mrlab.domain.models import SpreadType
from mrlab.strategy.params import StrategyParams
from mrlab.strategy.state_machine import Side


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


def requires_short(side: Side, beta: float) -> bool:
    raise NotImplementedError


def size_position(
    side: Side,
    beta: float,
    spread_type: SpreadType,
    price_y: float,
    price_x: float,
    equity: float,
    params: StrategyParams,
) -> SizingResult:
    raise NotImplementedError
