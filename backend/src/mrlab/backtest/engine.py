"""Motor de backtest de un par, sin información futura (secciones 7, 8 y 9).

CONVENCIÓN DE EJECUCIÓN (8.2): la señal se calcula con el cierre del día t
usando solo información hasta t; se ejecuta al cierre del día t+1 (siguiente
observación disponible).

PnL (sección 9), nunca como diferencia del spread:
    ΔΠ_t = q_y,t−1·ΔP_y,t + q_x,t−1·ΔP_x,t − C_t

CONTRATO (a implementar) — bucle por observación i = 0..n−1 del segmento:
 1. Marcar a mercado con las cantidades del cierre anterior:
    pnl_gross_i = q_y·(P_y,i − P_y,i−1) + q_x·(P_x,i − P_x,i−1)  (0 en i = 0).
 2. Costes de mantenimiento del intervalo (i−1, i] con días naturales entre
    fechas, sobre nocional corto y exposición bruta/capital del cierre i−1
    (``costs.carry_costs``).
 3. Ejecutar al cierre i la orden decidida en i−1 (si la hay): entrada, salida
    o reducción por apalancamiento. Cantidades con el precio de ejecución P_i
    (orden al cierre por nocional); el nocional objetivo se decidió con el
    capital del cierre i−1. Costes de transacción por pierna sobre |Δq|·P_i.
    Entrada → estado LARGO/CORTO. Salida → SIN_POSICION, o ENFRIAMIENTO si la
    razón es un stop (bloquea entradas evaluadas en los cierres i..i+cooldown−1;
    la primera entrada posible se evalúa en el cierre i+cooldown y se ejecuta en
    i+cooldown+1). Si la razón activa el rearme, se bloquea ese lado.
 4. equity_i = equity_i−1 + pnl_gross_i − costes_i.
 5. Decidir en el cierre i (solo con z_i y datos ≤ i) la orden para i+1:
    - con posición: condiciones de salida (``state_machine.active_exit_conditions``,
      stop monetario con el PnL neto acumulado de la operación hasta i, tiempo
      máximo con barras desde la ejecución de la entrada). Si hay salida →
      estado CERRANDO y orden de salida pendiente; registrar todas las
      condiciones activas y la aplicada. Si no hay salida y
      exposición_bruta_i / equity_i > max_leverage → orden de reducción
      proporcional hasta max_leverage (evento registrado, con costes al ejecutarse).
    - sin posición (y no CERRANDO, no ENFRIAMIENTO, lado no bloqueado por rearme):
      ``entry_signal``. Si el lado requiere corto y not allow_short → evento
      "señal no ejecutable" (se registra, no se ejecuta). Si no, orden de entrada.
    - en el último cierre del segmento no se crean órdenes.
    - el rearme se libera en el cierre en que z vuelve a la banda de ese lado.
    - capital ≤ 0 con posición → salida forzada (evento "capital agotado").
 6. Fin del segmento: la posición abierta queda NO realizada y la operación se
    marca ``open_at_end=True`` (no se fuerza el cierre).

Contabilidad obligatoria (se prueba): Σ pnl_gross − Σ costes = Σ pnl_net;
capital_final = capital_inicial + Σ pnl_net; realizado + no realizado = total.
Realizado = suma del PnL neto de operaciones cerradas; no realizado = PnL neto
acumulado (bruto − costes ya incurridos) de la operación abierta.

Columnas de ``daily`` (en este orden): date, z, state, q_y, q_x, price_y,
price_x, pnl_gross, cost_commission, cost_slippage, cost_borrow,
cost_financing, cost_total, pnl_net, equity, gross_exposure, net_exposure,
leverage, realized_cum, unrealized, trade_id (o None), event (texto o "").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from pydantic import BaseModel

from mrlab.backtest.metrics import Metrics
from mrlab.domain.models import AssetClass, CalendarType, DataOrigin, Segment, SpreadType
from mrlab.strategy.params import CostParams, StrategyParams
from mrlab.strategy.state_machine import ExitReason, Side

EXECUTION_CONVENTION = (
    "La señal se calcula con el cierre del día t usando solo información disponible hasta t, "
    "y se ejecuta al cierre del día t+1 (siguiente observación disponible). Con datos diarios no se "
    "afirma ninguna ejecución intradía."
)

DAILY_COLUMNS = [
    "date",
    "z",
    "state",
    "q_y",
    "q_x",
    "price_y",
    "price_x",
    "pnl_gross",
    "cost_commission",
    "cost_slippage",
    "cost_borrow",
    "cost_financing",
    "cost_total",
    "pnl_net",
    "equity",
    "gross_exposure",
    "net_exposure",
    "leverage",
    "realized_cum",
    "unrealized",
    "trade_id",
    "event",
]


@dataclass(frozen=True)
class LegSpec:
    symbol: str
    asset_class: AssetClass


@dataclass(frozen=True)
class BacktestInput:
    """Entrada del motor. ``prices`` (columnas "y", "x") son precios EJECUTABLES;
    ``z`` está indexado igual y se calculó solo con información ≤ t.
    ``y`` es la pierna dependiente (A) del spread elegido."""

    prices: pd.DataFrame
    z: pd.Series
    beta: float
    spread_type: SpreadType
    half_life: float | None
    leg_y: LegSpec
    leg_x: LegSpec
    calendar: CalendarType
    segment: Segment
    origin: DataOrigin = DataOrigin.HISTORICAL


class Trade(BaseModel):
    trade_id: int
    side: Side
    entry_signal_date: date
    entry_date: date
    entry_z: float  # z en el cierre de la señal de entrada
    exit_signal_date: date | None = None
    exit_date: date | None = None
    exit_z: float | None = None
    exit_reason: ExitReason | None = None
    active_exit_conditions: list[ExitReason] = []
    bars_held: int = 0
    q_y: float
    q_x: float
    price_y_entry: float
    price_x_entry: float
    price_y_exit: float | None = None
    price_x_exit: float | None = None
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    borrow: float = 0.0
    financing: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    return_on_capital: float = 0.0  # net_pnl / capital inicial del par
    notional_traded: float = 0.0
    open_at_end: bool = False


class Event(BaseModel):
    date: date
    kind: str  # p. ej. "senal_no_ejecutable", "reduccion_apalancamiento", "tamano_reducido", "capital_agotado", "rearme_bloqueado"
    message: str
    details: dict = {}


@dataclass
class BacktestResult:
    segment: Segment
    daily: pd.DataFrame
    trades: list[Trade]
    events: list[Event]
    metrics: Metrics
    params: StrategyParams
    costs: CostParams
    max_holding_bars: int
    leg_y: LegSpec
    leg_x: LegSpec
    beta: float
    spread_type: SpreadType
    calendar: CalendarType
    origin: DataOrigin
    execution_convention: str = EXECUTION_CONVENTION
    warnings: list[str] = field(default_factory=list)


def run_backtest(inp: BacktestInput, params: StrategyParams, costs: CostParams) -> BacktestResult:
    raise NotImplementedError
