"""Métricas de resultados (sección 10), cada una con explicación de una línea.

CONTRATO ``compute_metrics(daily, trades, initial_capital, periods_per_year, calendar)``:
- Rendimientos diarios netos r_t = equity_t/equity_{t−1} − 1 (incluye días sin
  posición). Sharpe = media/desv(ddof=1)·√periodos, tasa libre de riesgo 0
  (declarada en ``sharpe_convention``, p. ej. "rf = 0; anualizado con 252").
  None si desviación 0 o < 2 rendimientos.
- Retorno total = equity_final/capital − 1; anualizado = (1+R)^(periodos/n_rend) − 1.
- Máximo drawdown (fracción, ≤ 0) y su duración en observaciones (desde el
  pico hasta la recuperación, o hasta el final si no se recupera) + fechas.
- Operaciones CERRADAS: número, % aciertos (net_pnl > 0), ganancia media,
  pérdida media, factor de beneficio (None si no hay pérdidas → mostrar
  "no definido"), duración media en observaciones. Las abiertas al final se
  cuentan aparte (``n_open_at_end``).
- Rotación = nocional operado total / capital inicial; también anualizada.
- Exposición bruta media / capital; apalancamiento máximo usado.
- Costes totales y por tipo; % del PnL bruto (None si PnL bruto ≤ 0).
- ``warnings``: si n_trades < 10 → "Muestra insuficiente para conclusiones
  estadísticas (menos de 10 operaciones)."; si costes > PnL bruto positivo →
  aviso de que los costes se comen el resultado.
- ``METRIC_HELP``: tooltip de una línea en español por cada métrica.

Precisiones de la implementación:
- La fila 0 de ``daily`` es el punto de partida (en el motor, equity_0 = capital
  inicial): hay n − 1 rendimientos y el tiempo transcurrido es n − 1 periodos.
- Las fracciones (retornos, drawdown, % aciertos, costes / bruto, rotación) se
  devuelven como fracción (0,25 = 25 %); la interfaz les da formato.
- ``avg_gross_exposure`` = media de la exposición bruta diaria / capital inicial
  (incluye días sin posición).
- ``costs_by_type`` usa las claves ``COST_TYPE_KEYS`` (en español).
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mrlab.domain.models import CalendarType

MIN_TRADES_WARNING = 10

INSUFFICIENT_SAMPLE_WARNING = (
    "Muestra insuficiente para conclusiones estadísticas (menos de 10 operaciones)."
)

#: Columna de ``daily`` → clave en ``Metrics.costs_by_type``.
COST_TYPE_KEYS: dict[str, str] = {
    "cost_commission": "comision",
    "cost_slippage": "slippage",
    "cost_borrow": "prestamo_cortos",
    "cost_financing": "financiacion",
}

#: Desviación típica por debajo de la cual el Sharpe no está definido.
_STD_EPS = 1e-14

METRIC_HELP: dict[str, str] = {
    "initial_capital": "Capital asignado al par al inicio del segmento.",
    "final_equity": "Capital al último cierre del segmento, incluido el resultado no realizado.",
    "net_pnl": "Ganancia o pérdida total después de todos los costes (realizada + no realizada).",
    "gross_pnl": "Ganancia o pérdida por movimiento de precios, antes de costes.",
    "total_return": "Variación total del capital en el segmento: capital final / capital inicial − 1.",
    "annualized_return": "Retorno total expresado como tasa anual compuesta equivalente.",
    "max_drawdown": "Mayor caída del capital desde un máximo previo, en fracción del máximo.",
    "max_drawdown_duration": "Observaciones desde el máximo previo hasta recuperarlo (o hasta el final si no se recupera).",
    "max_drawdown_start": "Fecha del máximo previo a la peor caída.",
    "max_drawdown_end": "Fecha en que se recuperó el máximo, o el último cierre si no se recuperó.",
    "max_drawdown_recovered": "Indica si el capital volvió a su máximo previo dentro del segmento.",
    "sharpe": "Rendimiento medio diario neto dividido por su volatilidad, anualizado (tasa libre de riesgo 0).",
    "sharpe_convention": "Convención usada en el Sharpe: tasa libre de riesgo y periodos por año.",
    "periods_per_year": "Observaciones por año para anualizar: 252 (acciones, ETF, mixtos) o 365 (solo cripto).",
    "n_observations": "Número de cierres diarios simulados en el segmento.",
    "n_trades": "Operaciones cerradas en el segmento (no incluye la que quede abierta al final).",
    "n_open_at_end": "Operaciones que siguen abiertas al último cierre; su resultado es no realizado.",
    "win_rate": "Fracción de operaciones cerradas con PnL neto positivo.",
    "avg_win": "PnL neto medio de las operaciones cerradas con resultado positivo.",
    "avg_loss": "PnL neto medio de las operaciones cerradas con resultado negativo (valor negativo).",
    "profit_factor": "Suma de ganancias / suma de pérdidas de las operaciones cerradas; no definido si no hubo pérdidas.",
    "avg_holding_bars": "Duración media de las operaciones cerradas, en observaciones.",
    "turnover": "Nocional total operado (todas las piernas y ajustes) dividido por el capital inicial.",
    "turnover_annual": "Rotación expresada por año.",
    "avg_gross_exposure": "Exposición bruta media diaria (suma de ambas piernas) / capital inicial, incluidos días sin posición.",
    "max_leverage_used": "Mayor apalancamiento observado al cierre: exposición bruta / capital.",
    "total_costs": "Suma de comisiones, slippage, préstamo de cortos y financiación.",
    "costs_by_type": "Costes desglosados: comisión, slippage, préstamo de cortos y financiación.",
    "costs_pct_of_gross": "Costes totales como fracción del PnL bruto; no aplica si el PnL bruto es ≤ 0.",
    "realized_pnl": "PnL neto de las operaciones ya cerradas.",
    "unrealized_pnl": "PnL neto (marcado a mercado, menos costes ya pagados) de la operación abierta al final.",
    "warnings": "Avisos sobre la fiabilidad de las métricas (p. ej. pocas operaciones).",
}


def _fmt(x: float, decimals: int = 2) -> str:
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


class Metrics(BaseModel):
    initial_capital: float
    final_equity: float
    net_pnl: float
    gross_pnl: float
    total_return: float
    annualized_return: float | None
    max_drawdown: float
    max_drawdown_duration: int
    max_drawdown_start: date | None = None
    max_drawdown_end: date | None = None
    max_drawdown_recovered: bool | None = None
    sharpe: float | None
    sharpe_convention: str
    periods_per_year: int
    n_observations: int
    n_trades: int
    n_open_at_end: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    avg_holding_bars: float | None
    turnover: float
    turnover_annual: float | None
    avg_gross_exposure: float
    max_leverage_used: float
    total_costs: float
    costs_by_type: dict[str, float]
    costs_pct_of_gross: float | None
    realized_pnl: float
    unrealized_pnl: float
    warnings: list[str] = []


def sharpe_convention(periods_per_year: int, calendar: CalendarType) -> str:
    basis = (
        "solo cripto, calendario de 7 días"
        if calendar == CalendarType.CRYPTO_7D
        else "acciones, ETF y pares mixtos"
    )
    return (
        f"rf = 0; anualizado con {periods_per_year} ({basis}); rendimientos diarios netos "
        "del capital, incluidos los días sin posición"
    )


def daily_returns(equity: np.ndarray) -> np.ndarray:
    """r_t = equity_t/equity_{t−1} − 1; se omiten los tramos con capital previo ≤ 0."""
    if equity.size < 2:
        return np.empty(0)
    prev, cur = equity[:-1], equity[1:]
    ok = prev > 0
    return cur[ok] / prev[ok] - 1.0


def sharpe_ratio(returns: np.ndarray, periods_per_year: int) -> float | None:
    if returns.size < 2:
        return None
    sd = float(np.std(returns, ddof=1))
    if not math.isfinite(sd) or sd <= _STD_EPS:
        return None
    return float(np.mean(returns)) / sd * math.sqrt(periods_per_year)


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int | None, int | None, bool | None]:
    """(drawdown mínimo ≤ 0, duración, índice del pico, índice de fin, recuperado).

    El episodio empieza en el último cierre en máximos antes del valle y termina en
    el primer cierre posterior que alcanza ese máximo (o en el último cierre).
    """
    if equity.size == 0:
        return 0.0, 0, None, None, None
    peak = np.maximum.accumulate(equity)
    dd = np.zeros_like(equity)
    np.divide(equity, peak, out=dd, where=peak > 0)
    dd = np.where(peak > 0, dd - 1.0, 0.0)
    trough = int(np.argmin(dd))
    worst = float(dd[trough])
    if worst >= 0:
        return 0.0, 0, None, None, None
    start = int(np.flatnonzero(dd[: trough + 1] >= 0)[-1])
    after = np.flatnonzero(equity[trough + 1 :] >= peak[trough])
    recovered = after.size > 0
    end = trough + 1 + int(after[0]) if recovered else equity.size - 1
    return worst, end - start, start, end, recovered


def _as_date(value) -> date:
    return pd.Timestamp(value).date()


def compute_metrics(
    daily: pd.DataFrame,
    trades: list,
    initial_capital: float,
    periods_per_year: int,
    calendar: CalendarType,
) -> Metrics:
    n = len(daily)
    equity = daily["equity"].to_numpy(dtype=float) if n else np.empty(0)
    final_equity = float(equity[-1]) if n else float(initial_capital)
    elapsed = max(n - 1, 0)

    gross_pnl = float(daily["pnl_gross"].sum()) if n else 0.0
    net_pnl = float(daily["pnl_net"].sum()) if n else 0.0
    costs_by_type = {
        key: float(daily[col].sum()) if n else 0.0 for col, key in COST_TYPE_KEYS.items()
    }
    total_costs = float(daily["cost_total"].sum()) if n else 0.0

    total_return = final_equity / initial_capital - 1.0
    annualized = None
    if elapsed > 0 and 1.0 + total_return > 0:
        annualized = (1.0 + total_return) ** (periods_per_year / elapsed) - 1.0

    returns = daily_returns(equity)
    dd, dd_duration, dd_start, dd_end, dd_recovered = max_drawdown(equity)
    dates = daily["date"].to_numpy() if n else []

    closed = [t for t in trades if not t.open_at_end]
    n_open = len(trades) - len(closed)
    pnls = [float(t.net_pnl) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    turnover = sum(float(t.notional_traded) for t in trades) / initial_capital
    leverage = daily["leverage"].to_numpy(dtype=float) if n else np.empty(0)
    finite_lev = leverage[np.isfinite(leverage)]

    warnings: list[str] = []
    if len(closed) < MIN_TRADES_WARNING:
        warnings.append(INSUFFICIENT_SAMPLE_WARNING)
    if gross_pnl > 0 and total_costs > gross_pnl:
        warnings.append(
            f"Los costes ({_fmt(total_costs)}) superan el PnL bruto ({_fmt(gross_pnl)}): los costes "
            "se comen todo el resultado de la estrategia."
        )

    return Metrics(
        initial_capital=float(initial_capital),
        final_equity=final_equity,
        net_pnl=net_pnl,
        gross_pnl=gross_pnl,
        total_return=total_return,
        annualized_return=annualized,
        max_drawdown=dd,
        max_drawdown_duration=dd_duration,
        max_drawdown_start=_as_date(dates[dd_start]) if dd_start is not None else None,
        max_drawdown_end=_as_date(dates[dd_end]) if dd_end is not None else None,
        max_drawdown_recovered=dd_recovered,
        sharpe=sharpe_ratio(returns, periods_per_year),
        sharpe_convention=sharpe_convention(periods_per_year, calendar),
        periods_per_year=int(periods_per_year),
        n_observations=n,
        n_trades=len(closed),
        n_open_at_end=n_open,
        win_rate=len(wins) / len(closed) if closed else None,
        avg_win=float(np.mean(wins)) if wins else None,
        avg_loss=float(np.mean(losses)) if losses else None,
        profit_factor=sum(wins) / abs(sum(losses)) if losses else None,
        avg_holding_bars=float(np.mean([t.bars_held for t in closed])) if closed else None,
        turnover=turnover,
        turnover_annual=turnover * periods_per_year / elapsed if elapsed > 0 else None,
        avg_gross_exposure=float(daily["gross_exposure"].mean()) / initial_capital if n else 0.0,
        max_leverage_used=float(finite_lev.max()) if finite_lev.size else 0.0,
        total_costs=total_costs,
        costs_by_type=costs_by_type,
        costs_pct_of_gross=total_costs / gross_pnl if gross_pnl > 0 else None,
        realized_pnl=float(daily["realized_cum"].iloc[-1]) if n else 0.0,
        unrealized_pnl=float(daily["unrealized"].iloc[-1]) if n else 0.0,
        warnings=warnings,
    )
