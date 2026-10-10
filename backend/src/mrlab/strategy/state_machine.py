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

CONTRATO (a implementar): funciones puras usadas por el motor.
"""

from __future__ import annotations

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


EXIT_PRIORITY: tuple[ExitReason, ...] = (
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
}


def entry_signal(z: float, params: StrategyParams) -> Side | None:
    """Lado de entrada que indica z (o None). No mira el estado: eso lo hace el motor."""
    raise NotImplementedError


def active_exit_conditions(
    side: Side,
    z: float,
    trade_net_pnl: float,
    bars_held: int,
    max_bars: int,
    params: StrategyParams,
) -> list[ExitReason]:
    """Todas las condiciones de salida activas en esta observación, en orden de prioridad."""
    raise NotImplementedError


def select_exit(active: list[ExitReason]) -> ExitReason | None:
    """La condición de mayor prioridad (EXIT_PRIORITY) o None."""
    raise NotImplementedError


def rearm_cleared(side: Side, z: float, params: StrategyParams) -> bool:
    """True si z volvió dentro de la banda de entrada para ese lado."""
    raise NotImplementedError
