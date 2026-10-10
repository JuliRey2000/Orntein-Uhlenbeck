"""Utilidades de las pruebas de estrategia y backtest (z construido a mano, sin model/ ni stats/)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from conftest import cointegrated_log_prices

from mrlab.backtest.engine import BacktestInput, LegSpec
from mrlab.domain.models import AssetClass, CalendarType, Segment, SpreadType
from mrlab.strategy.params import CostParams

NO_COSTS = CostParams(
    commission_bps=0.0,
    slippage_bps_equity=0.0,
    slippage_bps_crypto=0.0,
    borrow_rate_annual=0.0,
    financing_rate_annual=0.0,
)


def bdays(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    """n días hábiles (lunes a viernes) desde ``start`` (2024-01-01 es lunes)."""
    return pd.date_range(start=start, periods=n, freq="B", name="date")


def make_input(
    z: Sequence[float] | np.ndarray,
    price_y: float | Sequence[float] = 100.0,
    price_x: float | Sequence[float] = 50.0,
    *,
    beta: float = 1.0,
    dates: pd.DatetimeIndex | None = None,
    spread_type: SpreadType = SpreadType.LOG,
    half_life: float | None = 10.0,
    class_y: AssetClass = AssetClass.EQUITY,
    class_x: AssetClass = AssetClass.EQUITY,
    calendar: CalendarType = CalendarType.EXCHANGE,
    segment: Segment = Segment.VALIDATION,
) -> BacktestInput:
    z = np.asarray(z, dtype=float)
    idx = dates if dates is not None else bdays(len(z))
    py = np.broadcast_to(np.asarray(price_y, dtype=float), z.shape).copy()
    px = np.broadcast_to(np.asarray(price_x, dtype=float), z.shape).copy()
    return BacktestInput(
        prices=pd.DataFrame({"y": py, "x": px}, index=idx),
        z=pd.Series(z, index=idx, name="z"),
        beta=beta,
        spread_type=spread_type,
        half_life=half_life,
        leg_y=LegSpec("AAA", class_y),
        leg_x=LegSpec("BBB", class_x),
        calendar=calendar,
        segment=segment,
    )


class SimPair:
    """Par simulado con parámetros VERDADEROS fijos: z = (log y − α − β·log x − μ)/σ_eq."""

    def __init__(
        self,
        n: int = 600,
        seed: int = 7,
        beta: float = 1.0,
        alpha: float = 0.0,
        lam: float = 0.1,
        sigma: float = 0.015,
        b_vol: float = 0.012,
    ):
        log_y, log_x = cointegrated_log_prices(
            n=n, seed=seed, beta=beta, alpha=alpha, lam=lam, sigma=sigma, b_vol=b_vol
        )
        self.alpha, self.beta, self.mu = alpha, beta, 0.0
        self.sigma_eq = sigma / np.sqrt(2 * lam)
        self.half_life = float(np.log(2) / lam)
        self.index = bdays(n, start="2018-01-01")
        self.prices = pd.DataFrame({"y": np.exp(log_y), "x": np.exp(log_x)}, index=self.index)

    def z_of(self, prices: pd.DataFrame) -> pd.Series:
        spread = np.log(prices["y"]) - self.alpha - self.beta * np.log(prices["x"])
        return ((spread - self.mu) / self.sigma_eq).rename("z")

    def input_for(self, prices: pd.DataFrame | None = None, **overrides) -> BacktestInput:
        prices = self.prices if prices is None else prices
        kwargs = dict(
            prices=prices,
            z=self.z_of(prices),
            beta=self.beta,
            spread_type=SpreadType.LOG,
            half_life=self.half_life,
            leg_y=LegSpec("SIM-A", AssetClass.EQUITY),
            leg_x=LegSpec("SIM-B", AssetClass.EQUITY),
            calendar=CalendarType.EXCHANGE,
            segment=Segment.VALIDATION,
        )
        kwargs.update(overrides)
        return BacktestInput(**kwargs)
