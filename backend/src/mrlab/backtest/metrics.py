"""Métricas de resultados (sección 10), cada una con explicación de una línea.

CONTRATO (a implementar) ``compute_metrics(daily, trades, initial_capital, periods_per_year, calendar)``:
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
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel

from mrlab.domain.models import CalendarType

MIN_TRADES_WARNING = 10

METRIC_HELP: dict[str, str] = {}


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


def compute_metrics(
    daily: pd.DataFrame,
    trades: list,
    initial_capital: float,
    periods_per_year: int,
    calendar: CalendarType,
) -> Metrics:
    raise NotImplementedError
