"""Motor de backtest de un par, sin información futura (secciones 7, 8 y 9).

CONVENCIÓN DE EJECUCIÓN (8.2): la señal se calcula con el cierre del día t
usando solo información hasta t; se ejecuta al cierre del día t+1 (siguiente
observación disponible).

PnL (sección 9), nunca como diferencia del spread:
    ΔΠ_t = q_y,t−1·ΔP_y,t + q_x,t−1·ΔP_x,t − C_t

CONTRATO — bucle por observación i = 0..n−1 del segmento:
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
    - capital ≤ 0 con posición → salida forzada (evento "capital agotado"); en
      el último cierre se registra el evento, sin orden.
 6. Fin del segmento: la posición abierta queda NO realizada y la operación se
    marca ``open_at_end=True`` (no se fuerza el cierre).

Precisiones de la implementación:
- La reducción por apalancamiento se decide en i con un factor
  f = max_leverage·equity_i / exposición_i aplicado a AMBAS cantidades (redondeo
  hacia cero si ``integer_units``); se ejecuta en i+1 con costes. Solo se dispara
  si el apalancamiento supera el máximo en más de ``LEVERAGE_TOLERANCE`` (evita
  reducciones minúsculas causadas por los costes de la propia entrada). Si con
  unidades enteras una pierna quedaría en 0, se cierra la posición
  (``ExitReason.FORCED``).
- Una entrada que al ejecutarse no se puede dimensionar (p. ej. una pierna queda
  en 0 unidades enteras al precio de t+1) se registra como "senal_no_ejecutable".
- Las señales bloqueadas (enfriamiento, rearme, sin cortos, capital agotado) se
  anotan en la columna ``event`` de cada cierre; en ``events`` se registra un
  evento por episodio (cuando empieza o cambia el motivo o el lado).
- ``trade_id`` de cada fila es la operación a la que se atribuyen el PnL y los
  costes de esa fila (la mantenida en (i−1, i] o la que entra en i).
- ``Trade.q_y``/``q_x`` son las cantidades de ENTRADA (una reducción posterior
  queda en los eventos y en ``daily``). ``Trade.bars_held`` = observaciones desde
  la ejecución de la entrada hasta la de la salida (o hasta el último cierre).
- Avisos del resultado (``BacktestResult.warnings``): «capital agotado» si
  ALGÚN cierre de ``daily`` tiene capital ≤ 0 (también el último, donde ya no
  se crean órdenes), y ``CRYPTO_SHORT_WARNING`` si alguna operación deja corta
  una pierna de clase cripto (el corto de cripto spot requiere derivados, fuera
  de la v1; el préstamo cobrado es una aproximación).

Contabilidad obligatoria (se prueba): Σ pnl_gross − Σ costes = Σ pnl_net;
capital_final = capital_inicial + Σ pnl_net; realizado + no realizado = total.
Realizado = suma del PnL neto de operaciones cerradas; no realizado = PnL neto
acumulado (bruto − costes ya incurridos) de la operación abierta.

Columnas de ``daily`` (en este orden): date, z, state, q_y, q_x, price_y,
price_x, pnl_gross, cost_commission, cost_slippage, cost_borrow,
cost_financing, cost_total, pnl_net, equity, gross_exposure, net_exposure,
leverage, realized_cum, unrealized, trade_id (o None), event (texto o "").
``leverage`` es NaN si hay exposición y el capital es ≤ 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mrlab.backtest.costs import carry_costs, transaction_costs
from mrlab.backtest.metrics import Metrics, compute_metrics
from mrlab.data.calendar import periods_per_year
from mrlab.domain.errors import ConfigError, DataError
from mrlab.domain.models import AssetClass, CalendarType, DataOrigin, Segment, SpreadType
from mrlab.strategy.params import CostParams, StrategyParams
from mrlab.strategy.sizing import PAIRS_NEED_BOTH_LEGS, requires_short, short_legs, size_position
from mrlab.strategy.state_machine import (
    EXIT_REASON_LABELS,
    REARM_REASONS,
    SIDE_LABELS,
    STATE_FOR_SIDE,
    STOP_REASONS,
    ExitReason,
    PositionState,
    Side,
    active_exit_conditions,
    entry_signal,
    rearm_cleared,
    select_exit,
)

EXECUTION_CONVENTION = (
    "La señal se calcula con el cierre del día t usando solo información disponible hasta t, "
    "y se ejecuta al cierre del día t+1 (siguiente observación disponible). Con datos diarios no se "
    "afirma ninguna ejecución intradía."
)

#: Margen relativo sobre max_leverage antes de ordenar una reducción.
LEVERAGE_TOLERANCE = 0.01

#: Aviso cuando la simulación vende en corto una pierna cripto ({symbols}: símbolos afectados).
CRYPTO_SHORT_WARNING = (
    "Se simuló la venta en corto de cripto spot ({symbols}). En la práctica, ponerse corto en "
    "cripto spot requiere derivados (perpetuos con funding), que están fuera del alcance de la "
    "v1; el coste de préstamo de cortos aplicado es solo una aproximación."
)

CAPITAL_EXHAUSTED_WARNING = "El capital del par se agotó durante el segmento."

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


def _fmt(x: float, decimals: int = 2) -> str:
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


@dataclass
class _Order:
    kind: Literal["entrada", "salida", "reduccion"]
    signal_idx: int
    side: Side | None = None  # entrada
    reason: ExitReason | None = None  # salida
    q_y: float = 0.0  # reducción: cantidades objetivo
    q_x: float = 0.0


def _validate_input(
    inp: BacktestInput,
) -> tuple[pd.DatetimeIndex, list[float], list[float], list[float]]:
    prices = inp.prices
    if not {"y", "x"} <= set(prices.columns):
        raise ConfigError(
            "La entrada del backtest debe tener las columnas de precios ejecutables «y» y «x».",
            code="ENTRADA_BACKTEST_INVALIDA",
        )
    if len(prices) == 0:
        raise DataError(
            "El segmento no tiene observaciones: no hay nada que simular.", code="SEGMENTO_VACIO"
        )
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ConfigError(
            "Los precios del backtest deben estar indexados por fecha.",
            code="ENTRADA_BACKTEST_INVALIDA",
        )
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise DataError(
            "Las fechas del segmento deben estar en orden creciente y sin duplicados.",
            code="FECHAS_INVALIDAS",
        )
    if not inp.z.index.equals(prices.index):
        raise ConfigError(
            "La serie z debe tener exactamente las mismas fechas que los precios del segmento.",
            code="ENTRADA_BACKTEST_INVALIDA",
        )
    if not math.isfinite(inp.beta):
        raise ConfigError(
            "El hedge ratio β no es un número finito.", code="ENTRADA_BACKTEST_INVALIDA"
        )
    py = prices["y"].to_numpy(dtype=float)
    px = prices["x"].to_numpy(dtype=float)
    bad = ~(np.isfinite(py) & np.isfinite(px) & (py > 0) & (px > 0))
    if bad.any():
        first = prices.index[int(np.argmax(bad))].date().isoformat()
        raise DataError(
            "Hay precios faltantes o no positivos en el segmento (primera fecha afectada: "
            f"{first}). El motor no rellena precios: revisa la validación de datos.",
            code="PRECIOS_INVALIDOS",
        )
    return prices.index, py.tolist(), px.tolist(), inp.z.to_numpy(dtype=float).tolist()


class _Simulator:
    """Estado mutable de una simulación. Cada paso usa solo datos hasta el cierre actual."""

    def __init__(self, inp: BacktestInput, params: StrategyParams, costs: CostParams):
        self.inp = inp
        self.p = params
        self.c = costs
        self.index, self.py, self.px, self.z = _validate_input(inp)
        self.n = len(self.py)
        self.max_bars = params.max_holding_bars(inp.half_life)
        self.capital0 = params.initial_capital

        self.equity = params.initial_capital
        self.q_y = 0.0
        self.q_x = 0.0
        self.state = PositionState.FLAT
        self.pending: _Order | None = None
        self.trade: Trade | None = None
        self.entry_idx = -1
        self.realized = 0.0
        self.cooldown_until = -1  # primer cierre en que se puede volver a evaluar una entrada
        self.rearm_blocked: set[Side] = set()
        self.last_blocked: tuple[str, Side] | None = None

        self.trades: list[Trade] = []
        self.events: list[Event] = []
        self.rows: dict[str, list] = {col: [] for col in DAILY_COLUMNS if col != "date"}

    # ------------------------------------------------------------------ bucle
    def run(self) -> tuple[pd.DataFrame, list[Trade], list[Event]]:
        for i in range(self.n):
            self._step(i)
        if self.trade is not None:
            self.trade.open_at_end = True
        columns = {col: self.rows[col] for col in DAILY_COLUMNS if col != "date"}
        columns["trade_id"] = pd.Series(self.rows["trade_id"], dtype=object)
        daily = pd.DataFrame({"date": self.index.to_numpy(), **columns}, columns=DAILY_COLUMNS)
        return daily, self.trades, self.events

    def _step(self, i: int) -> None:
        notes: list[str] = []
        equity_prev = self.equity
        held = self.trade  # operación mantenida durante (i−1, i]
        gross = borrow = financing = 0.0
        if held is not None:
            py0, px0 = self.py[i - 1], self.px[i - 1]
            gross = self.q_y * (self.py[i] - py0) + self.q_x * (self.px[i] - px0)
            short_notional = sum(abs(q * p) for q, p in ((self.q_y, py0), (self.q_x, px0)) if q < 0)
            gross_prev = abs(self.q_y * py0) + abs(self.q_x * px0)
            days = (self.index[i] - self.index[i - 1]).days
            borrow, financing = carry_costs(short_notional, gross_prev, equity_prev, days, self.c)
            self._charge(held, gross=gross, borrow=borrow, financing=financing)

        commission = slippage = 0.0
        if self.pending is not None:
            commission, slippage = self._execute(i, equity_prev, notes)

        cost_total = commission + slippage + borrow + financing
        pnl_net = gross - cost_total
        self.equity = equity_prev + pnl_net
        if self.trade is not None:
            self.trade.bars_held = i - self.entry_idx

        price_y, price_x = self.py[i], self.px[i]
        gross_exposure = abs(self.q_y * price_y) + abs(self.q_x * price_x)
        net_exposure = self.q_y * price_y + self.q_x * price_x
        if gross_exposure == 0:
            leverage = 0.0
        elif self.equity > 0:
            leverage = gross_exposure / self.equity
        else:
            leverage = math.nan
        unrealized = self.trade.net_pnl if self.trade is not None else 0.0
        row_trade = held if held is not None else self.trade

        self._decide(i, gross_exposure, leverage, notes)

        row = self.rows
        row["z"].append(self.z[i])
        row["state"].append(self.state.value)
        row["q_y"].append(self.q_y)
        row["q_x"].append(self.q_x)
        row["price_y"].append(price_y)
        row["price_x"].append(price_x)
        row["pnl_gross"].append(gross)
        row["cost_commission"].append(commission)
        row["cost_slippage"].append(slippage)
        row["cost_borrow"].append(borrow)
        row["cost_financing"].append(financing)
        row["cost_total"].append(cost_total)
        row["pnl_net"].append(pnl_net)
        row["equity"].append(self.equity)
        row["gross_exposure"].append(gross_exposure)
        row["net_exposure"].append(net_exposure)
        row["leverage"].append(leverage)
        row["realized_cum"].append(self.realized)
        row["unrealized"].append(unrealized)
        row["trade_id"].append(row_trade.trade_id if row_trade is not None else None)
        row["event"].append(" | ".join(notes))

    # ------------------------------------------------------------- ejecución
    def _execute(self, i: int, equity_prev: float, notes: list[str]) -> tuple[float, float]:
        order, self.pending = self.pending, None
        assert order is not None
        if order.kind == "entrada":
            return self._execute_entry(i, order, equity_prev, notes)
        if order.kind == "salida":
            return self._execute_exit(i, order, notes)
        return self._execute_reduction(i, order, notes)

    def _execute_entry(
        self, i: int, order: _Order, equity_prev: float, notes: list[str]
    ) -> tuple[float, float]:
        side = order.side
        assert side is not None
        label = SIDE_LABELS[side]
        # Nocional decidido con el capital del cierre de la señal; cantidades al precio de ejecución.
        sizing = size_position(
            side, self.inp.beta, self.inp.spread_type, self.py[i], self.px[i], equity_prev, self.p
        )
        if not sizing.executable:
            self._event(
                i,
                "senal_no_ejecutable",
                f"La entrada en {label} decidida el {self._date(order.signal_idx).isoformat()} no se "
                f"ejecutó: {' '.join(sizing.reasons)}",
                {"lado": side.value, "z_senal": self.z[order.signal_idx]},
            )
            notes.append(f"Entrada no ejecutada ({label})")
            return 0.0, 0.0

        trade = Trade(
            trade_id=len(self.trades) + 1,
            side=side,
            entry_signal_date=self._date(order.signal_idx),
            entry_date=self._date(i),
            entry_z=self.z[order.signal_idx],
            q_y=sizing.q_y,
            q_x=sizing.q_x,
            price_y_entry=self.py[i],
            price_x_entry=self.px[i],
        )
        self.trades.append(trade)
        self.trade = trade
        self.entry_idx = i
        self.state = STATE_FOR_SIDE[side]
        costs = self._trade_legs(i, sizing.q_y, sizing.q_x, trade)
        self.q_y, self.q_x = sizing.q_y, sizing.q_x
        if sizing.reduced:
            self._event(
                i,
                "tamano_reducido",
                f"Operación {trade.trade_id}: " + " ".join(sizing.reasons),
                {"objetivo": sizing.target_gross, "exposicion_bruta": sizing.gross_exposure},
            )
        notes.append(f"Entrada ejecutada: {label} (operación {trade.trade_id})")
        return costs

    def _execute_exit(self, i: int, order: _Order, notes: list[str]) -> tuple[float, float]:
        trade = self.trade
        reason = order.reason
        assert trade is not None and reason is not None
        costs = self._trade_legs(i, self.q_y, self.q_x, trade)
        trade.exit_date = self._date(i)
        trade.price_y_exit = self.py[i]
        trade.price_x_exit = self.px[i]
        trade.bars_held = i - self.entry_idx
        self.q_y = self.q_x = 0.0
        self.realized += trade.net_pnl
        self.trade = None
        if reason in STOP_REASONS and self.p.cooldown_bars > 0:
            self.state = PositionState.COOLDOWN
            self.cooldown_until = i + self.p.cooldown_bars
        else:
            self.state = PositionState.FLAT
        if self.p.rearm and reason in REARM_REASONS:
            self.rearm_blocked.add(trade.side)
        notes.append(f"Salida ejecutada: {EXIT_REASON_LABELS[reason]} (operación {trade.trade_id})")
        return costs

    def _execute_reduction(self, i: int, order: _Order, notes: list[str]) -> tuple[float, float]:
        trade = self.trade
        assert trade is not None
        costs = self._trade_legs(i, self.q_y - order.q_y, self.q_x - order.q_x, trade)
        self.q_y, self.q_x = order.q_y, order.q_x
        notes.append(f"Reducción por apalancamiento ejecutada (operación {trade.trade_id})")
        return costs

    def _trade_legs(self, i: int, dq_y: float, dq_x: float, trade: Trade) -> tuple[float, float]:
        """Costes de transacción de operar |Δq| en cada pierna al cierre i (atribuidos a ``trade``)."""
        notional_y = abs(dq_y) * self.py[i]
        notional_x = abs(dq_x) * self.px[i]
        com_y, slip_y = transaction_costs(notional_y, self.inp.leg_y.asset_class, self.c)
        com_x, slip_x = transaction_costs(notional_x, self.inp.leg_x.asset_class, self.c)
        trade.notional_traded += notional_y + notional_x
        self._charge(trade, commission=com_y + com_x, slippage=slip_y + slip_x)
        return com_y + com_x, slip_y + slip_x

    def _charge(
        self,
        trade: Trade,
        *,
        gross: float = 0.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        borrow: float = 0.0,
        financing: float = 0.0,
    ) -> None:
        trade.gross_pnl += gross
        trade.commission += commission
        trade.slippage += slippage
        trade.borrow += borrow
        trade.financing += financing
        trade.total_costs = trade.commission + trade.slippage + trade.borrow + trade.financing
        trade.net_pnl = trade.gross_pnl - trade.total_costs
        trade.return_on_capital = trade.net_pnl / self.capital0

    # -------------------------------------------------------------- decisión
    def _decide(self, i: int, gross_exposure: float, leverage: float, notes: list[str]) -> None:
        z = self.z[i]
        for side in sorted(self.rearm_blocked):
            if rearm_cleared(side, z, self.p):
                self.rearm_blocked.discard(side)
                notes.append(f"Rearme liberado: {SIDE_LABELS[side]}")
        if self.state == PositionState.COOLDOWN and i >= self.cooldown_until:
            self.state = PositionState.FLAT
        if i == self.n - 1:
            # En el último cierre no se crean órdenes, pero un capital agotado se registra.
            if self.trade is not None and self.equity <= 0:
                self._event(
                    i,
                    "capital_agotado",
                    f"Operación {self.trade.trade_id}: el capital del par llegó a "
                    f"{_fmt(self.equity)} en el último cierre del segmento con la posición "
                    "abierta; ya no queda un cierre siguiente para ejecutar la salida forzada.",
                    {"capital": self.equity},
                )
            return
        if self.trade is not None:
            self.last_blocked = None
            self._decide_with_position(i, z, gross_exposure, leverage, notes)
        else:
            self._decide_entry(i, z, notes)

    def _decide_with_position(
        self, i: int, z: float, gross_exposure: float, leverage: float, notes: list[str]
    ) -> None:
        trade = self.trade
        assert trade is not None
        p = self.p
        active = active_exit_conditions(
            trade.side, z, trade.net_pnl, i - self.entry_idx, self.max_bars, p
        )
        if self.equity <= 0:
            active = [ExitReason.FORCED, *active]
            self._event(
                i,
                "capital_agotado",
                f"Operación {trade.trade_id}: el capital del par llegó a {_fmt(self.equity)} con la posición "
                "abierta; se fuerza la salida al cierre siguiente y no se abrirán nuevas posiciones.",
                {"capital": self.equity},
            )
        elif not active and leverage > p.max_leverage * (1 + LEVERAGE_TOLERANCE):
            factor = p.max_leverage * self.equity / gross_exposure
            new_q_y, new_q_x = self.q_y * factor, self.q_x * factor
            if p.integer_units:
                new_q_y, new_q_x = float(math.trunc(new_q_y)), float(math.trunc(new_q_x))
            details = {"apalancamiento": leverage, "maximo": p.max_leverage, "factor": factor}
            if new_q_y == 0 or new_q_x == 0:
                active = [ExitReason.FORCED]
                self._event(
                    i,
                    "reduccion_apalancamiento",
                    f"Operación {trade.trade_id}: el apalancamiento ({_fmt(leverage)}×) supera el máximo "
                    f"({_fmt(p.max_leverage)}×) y reducirlo dejaría una pierna en 0 unidades enteras; "
                    "se cierra la posición al cierre siguiente.",
                    details,
                )
            else:
                self.pending = _Order("reduccion", i, q_y=new_q_y, q_x=new_q_x)
                self._event(
                    i,
                    "reduccion_apalancamiento",
                    f"Operación {trade.trade_id}: el apalancamiento ({_fmt(leverage)}×) supera el máximo "
                    f"({_fmt(p.max_leverage)}×); ambas piernas se reducen un {_fmt((1 - factor) * 100, 1)} % "
                    "al cierre siguiente, con costes.",
                    details,
                )
                notes.append(
                    f"Apalancamiento {_fmt(leverage)}× > {_fmt(p.max_leverage)}×: reducción pendiente"
                )
                return

        reason = select_exit(active)
        if reason is None:
            return
        self.pending = _Order("salida", i, reason=reason)
        self.state = PositionState.CLOSING
        trade.exit_signal_date = self._date(i)
        trade.exit_z = z
        trade.active_exit_conditions = list(active)
        trade.exit_reason = reason
        note = f"Señal de salida: {EXIT_REASON_LABELS[reason]}"
        if len(active) > 1:
            note += f" (condiciones activas: {', '.join(EXIT_REASON_LABELS[r] for r in active)})"
        notes.append(note)

    def _decide_entry(self, i: int, z: float, notes: list[str]) -> None:
        side = entry_signal(z, self.p)
        if side is None:
            self.last_blocked = None
            return
        blocked = self._entry_block(i, side, z)
        if blocked is not None:
            kind, message, note = blocked
            notes.append(note)
            if self.last_blocked != (kind, side):
                self._event(i, kind, message, {"lado": side.value, "z": z})
            self.last_blocked = (kind, side)
            return
        self.last_blocked = None
        self.pending = _Order("entrada", i, side=side)
        notes.append(
            f"Señal de entrada: {SIDE_LABELS[side]} (z = {_fmt(z)}); se ejecuta al cierre siguiente"
        )

    def _entry_block(self, i: int, side: Side, z: float) -> tuple[str, str, str] | None:
        """(tipo de evento, mensaje, nota corta) si la señal de entrada no puede generar orden."""
        label = SIDE_LABELS[side]
        p = self.p
        if self.state == PositionState.COOLDOWN:
            remaining = self.cooldown_until - i
            return (
                "senal_ignorada_enfriamiento",
                f"Señal de {label} (z = {_fmt(z)}) ignorada: enfriamiento tras un stop. Las señales "
                f"de entrada se ignoran durante {remaining} cierre(s) más, incluido este.",
                "Señal ignorada: enfriamiento",
            )
        if self.equity <= 0:
            return (
                "capital_agotado",
                f"Señal de {label} ignorada: el capital del par es {_fmt(self.equity)}; no se abren "
                "nuevas posiciones.",
                "Señal ignorada: capital agotado",
            )
        if side in self.rearm_blocked:
            return (
                "rearme_bloqueado",
                f"Señal de {label} (z = {_fmt(z)}) ignorada por la regla de rearme: tras una salida por "
                f"stop o por tiempo máximo de este lado, z debe volver dentro de ±{_fmt(p.entry_z)} antes "
                "de entrar otra vez por el mismo lado.",
                "Señal bloqueada por rearme",
            )
        if not p.allow_short and requires_short(side, self.inp.beta):
            short_y, short_x = short_legs(side, self.inp.beta)
            legs = [
                leg.symbol
                for leg, short in ((self.inp.leg_y, short_y), (self.inp.leg_x, short_x))
                if short
            ]
            return (
                "senal_no_ejecutable",
                f"Señal de {label} (z = {_fmt(z)}) no ejecutable: habría que vender en corto "
                f"{' y '.join(legs)} y las ventas en corto están desactivadas. {PAIRS_NEED_BOTH_LEGS}",
                "Señal no ejecutable (sin ventas en corto)",
            )
        return None

    # ------------------------------------------------------------- utilidades
    def _date(self, i: int) -> date:
        return self.index[i].date()

    def _event(self, i: int, kind: str, message: str, details: dict) -> None:
        self.events.append(Event(date=self._date(i), kind=kind, message=message, details=details))


def _crypto_short_symbols(inp: BacktestInput, trades: list[Trade]) -> list[str]:
    """Piernas cripto que alguna operación dejó en corto (cantidad de entrada < 0)."""
    symbols: list[str] = []
    for leg, short in (
        (inp.leg_y, any(t.q_y < 0 for t in trades)),
        (inp.leg_x, any(t.q_x < 0 for t in trades)),
    ):
        if short and leg.asset_class == AssetClass.CRYPTO and leg.symbol not in symbols:
            symbols.append(leg.symbol)
    return symbols


def _result_warnings(
    inp: BacktestInput,
    params: StrategyParams,
    trades: list[Trade],
    events: list[Event],
    daily: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []
    if inp.segment == Segment.TRAIN:
        warnings.append(
            "Formación: resultado DENTRO DE MUESTRA (α, β y el OU se estimaron con estos mismos datos). "
            "Es solo ilustrativo y tiende a verse mejor de lo que sería fuera de muestra."
        )
    if inp.origin == DataOrigin.SYNTHETIC:
        warnings.append("Datos SINTÉTICOS: no son precios históricos.")
    limit = params.stop_loss_pct * params.initial_capital
    for t in trades:
        if t.exit_reason == ExitReason.STOP_MONEY and t.net_pnl < -limit:
            warnings.append(
                f"Operación {t.trade_id}: el stop monetario se activó, pero la pérdida final "
                f"({_fmt(-t.net_pnl)}) superó el límite ({_fmt(limit)}) porque la salida se ejecuta al "
                "cierre siguiente; con saltos de precio la pérdida real puede ser mayor que el límite."
            )
        if t.open_at_end:
            warnings.append(
                f"La operación {t.trade_id} quedó abierta al final del segmento: su resultado figura "
                "como no realizado (no se fuerza el cierre)."
            )
    n_not_exec = sum(e.kind == "senal_no_ejecutable" for e in events)
    if n_not_exec:
        warnings.append(f"Se registraron {n_not_exec} señales no ejecutables (ver eventos).")
    exhausted = bool((daily["equity"] <= 0).any()) if len(daily) else False
    if exhausted or any(e.kind == "capital_agotado" for e in events):
        warnings.append(CAPITAL_EXHAUSTED_WARNING)
    crypto_shorts = _crypto_short_symbols(inp, trades)
    if crypto_shorts:
        warnings.append(CRYPTO_SHORT_WARNING.format(symbols=", ".join(crypto_shorts)))
    return warnings


def run_backtest(inp: BacktestInput, params: StrategyParams, costs: CostParams) -> BacktestResult:
    sim = _Simulator(inp, params, costs)
    daily, trades, events = sim.run()
    metrics = compute_metrics(
        daily, trades, params.initial_capital, periods_per_year(inp.calendar), inp.calendar
    )
    return BacktestResult(
        segment=inp.segment,
        daily=daily,
        trades=trades,
        events=events,
        metrics=metrics,
        params=params,
        costs=costs,
        max_holding_bars=sim.max_bars,
        leg_y=inp.leg_y,
        leg_x=inp.leg_x,
        beta=inp.beta,
        spread_type=inp.spread_type,
        calendar=inp.calendar,
        origin=inp.origin,
        warnings=_result_warnings(inp, params, trades, events, daily),
    )
