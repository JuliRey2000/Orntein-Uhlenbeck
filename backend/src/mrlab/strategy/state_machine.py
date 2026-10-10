"""Máquina de estados de la estrategia (7.2 y 7.5).

Estados: SIN_POSICION, LARGO_SPREAD, CORTO_SPREAD, CERRANDO, ENFRIAMIENTO.

Reglas (convención de ejecución en ``mrlab.backtest.engine``):
- z > +entrada → vender el spread (corto A, largo β·B); z < −entrada → comprarlo.
- Salidas DIRECCIONALES: un corto sale cuando z ≤ +salida; un largo cuando
  z ≥ −salida. Stop z: corto si z ≥ +stop; largo si z ≤ −stop.
- Stop monetario: PnL neto acumulado de la operación abierta ≤ −stop_loss_pct ×
  capital inicial del par.
- Tiempo máximo: observaciones mantenidas desde la ejecución de la entrada ≥ máximo.
- Prioridad si coinciden: stop monetario > stop z > tiempo máximo > salida normal.
  Se registran TODAS las condiciones activas y la aplicada.
- No hay entradas con posición abierta, en CERRANDO, ni durante ENFRIAMIENTO.
- Rearme (si ``rearm``): tras salir por stop (z o monetario) o tiempo máximo de
  un lado, no se reentra por ESE lado hasta que z vuelva dentro de la banda
  (corto: z ≤ +entrada; largo: z ≥ −entrada) en algún cierre posterior a la
  señal de salida. El lado contrario no se bloquea.

Nota: con salidas direccionales y salida < entrada < stop, el stop por z y la
salida normal son mutuamente excluyentes (corto: z ≥ stop y z ≤ salida no pueden
cumplirse a la vez). En una misma observación pueden coincidir como máximo tres
condiciones «normales»: stop monetario, una de las dos de z y tiempo máximo.

Además de las cuatro razones de la especificación existe ``FORCED`` (salida
forzada), que solo decide el motor: capital agotado con posición abierta, o una
reducción por apalancamiento que dejaría una pierna en 0 unidades enteras. Tiene
la máxima prioridad y no activa enfriamiento ni rearme.

Funciones puras usadas por el motor; no conocen el estado (eso lo hace el motor).
"""

from __future__ import annotations

import math
from enum import StrEnum

from mrlab.strategy.params import StrategyParams


class PositionState(StrEnum):
    FLAT = "SIN_POSICION"
    LONG = "LARGO_SPREAD"
    SHORT = "CORTO_SPREAD"
    CLOSING = "CERRANDO"
    COOLDOWN = "ENFRIAMIENTO"


class Side(StrEnum):
    LONG = "largo_spread"  # largo A, corto β·B (si β > 0)
    SHORT = "corto_spread"  # corto A, largo β·B (si β > 0)


class ExitReason(StrEnum):
    STOP_MONEY = "stop_monetario"
    STOP_Z = "stop_z"
    MAX_TIME = "tiempo_maximo"
    NORMAL = "salida_normal"
    FORCED = (
        "salida_forzada"  # solo la decide el motor (capital agotado, apalancamiento no reducible)
    )


EXIT_PRIORITY: tuple[ExitReason, ...] = (
    ExitReason.FORCED,
    ExitReason.STOP_MONEY,
    ExitReason.STOP_Z,
    ExitReason.MAX_TIME,
    ExitReason.NORMAL,
)

#: Salidas que activan enfriamiento.
STOP_REASONS = frozenset({ExitReason.STOP_MONEY, ExitReason.STOP_Z})
#: Salidas que activan el rearme (si está habilitado).
REARM_REASONS = frozenset({ExitReason.STOP_MONEY, ExitReason.STOP_Z, ExitReason.MAX_TIME})

EXIT_REASON_LABELS = {
    ExitReason.STOP_MONEY: "Stop monetario",
    ExitReason.STOP_Z: "Stop por z",
    ExitReason.MAX_TIME: "Tiempo máximo",
    ExitReason.NORMAL: "Salida normal (reversión)",
    ExitReason.FORCED: "Salida forzada",
}

SIDE_LABELS = {
    Side.LONG: "largo spread",
    Side.SHORT: "corto spread",
}

STATE_FOR_SIDE = {
    Side.LONG: PositionState.LONG,
    Side.SHORT: PositionState.SHORT,
}


def side_sign(side: Side) -> int:
    """+1 para largo spread (largo A), −1 para corto spread (corto A)."""
    return 1 if side == Side.LONG else -1


def entry_signal(z: float, params: StrategyParams) -> Side | None:
    """Lado de entrada que indica z (o None). No mira el estado: eso lo hace el motor."""
    if not math.isfinite(z):
        return None
    if z > params.entry_z:
        return Side.SHORT
    if z < -params.entry_z:
        return Side.LONG
    return None


def active_exit_conditions(
    side: Side,
    z: float,
    trade_net_pnl: float,
    bars_held: int,
    max_bars: int,
    params: StrategyParams,
) -> list[ExitReason]:
    """Todas las condiciones de salida activas en esta observación, en orden de prioridad.

    Con z no finito (p. ej. z móvil sin historia suficiente) solo se evalúan el
    stop monetario y el tiempo máximo.
    """
    active: list[ExitReason] = []
    if trade_net_pnl <= -params.stop_loss_pct * params.initial_capital:
        active.append(ExitReason.STOP_MONEY)
    stop_hit = normal_hit = False
    if math.isfinite(z):
        if side == Side.SHORT:
            stop_hit = z >= params.stop_z
            normal_hit = z <= params.exit_z
        else:
            stop_hit = z <= -params.stop_z
            normal_hit = z >= -params.exit_z
    if stop_hit:
        active.append(ExitReason.STOP_Z)
    if bars_held >= max_bars:
        active.append(ExitReason.MAX_TIME)
    if normal_hit:
        active.append(ExitReason.NORMAL)
    return active


def select_exit(active: list[ExitReason]) -> ExitReason | None:
    """La condición de mayor prioridad (EXIT_PRIORITY) o None."""
    for reason in EXIT_PRIORITY:
        if reason in active:
            return reason
    return None


def rearm_cleared(side: Side, z: float, params: StrategyParams) -> bool:
    """True si z volvió dentro de la banda de entrada para ese lado."""
    if not math.isfinite(z):
        return False
    if side == Side.SHORT:
        return z <= params.entry_z
    return z >= -params.entry_z
