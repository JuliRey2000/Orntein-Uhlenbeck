"""Costes de transacción y de mantenimiento (sección 9).

CONTRATO (a implementar):
- ``slippage_bps(asset_class, costs)``: cripto → slippage_bps_crypto; resto →
  slippage_bps_equity.
- ``transaction_costs(traded_notional_abs, asset_class, costs)`` →
  (comisión, slippage) = nocional × pb / 10.000 cada uno. Se aplica a cada
  pierna y a cada ajuste de cantidades (rebalanceo).
- ``carry_costs(short_notional, gross_exposure, equity, days, costs)`` →
  (préstamo, financiación), devengados por días NATURALES / day_count:
  préstamo = borrow_rate × |nocional corto| × días/365;
  financiación = financing_rate × max(0, exposición bruta − capital) × días/365.
  Los montos se miden al cierre ANTERIOR (posición mantenida en el intervalo).
- Dividendos: ya incluidos en el precio ajustado; nunca se suman aparte.
"""

from __future__ import annotations

from mrlab.domain.models import AssetClass
from mrlab.strategy.params import CostParams

DIVIDENDS_NOTE = (
    "Los precios de Yahoo están ajustados por splits y dividendos (auto_adjust=True); por eso el "
    "PnL no suma dividendos por separado (evita el doble conteo)."
)


def slippage_bps(asset_class: AssetClass, costs: CostParams) -> float:
    raise NotImplementedError


def transaction_costs(traded_notional_abs: float, asset_class: AssetClass, costs: CostParams) -> tuple[float, float]:
    raise NotImplementedError


def carry_costs(
    short_notional: float, gross_exposure: float, equity: float, days: int, costs: CostParams
) -> tuple[float, float]:
    raise NotImplementedError
