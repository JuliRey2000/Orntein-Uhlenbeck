"""Costes de transacción y de mantenimiento (sección 9).

CONTRATO:
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
  Sin exposición no hay financiación (el efectivo no se remunera ni se cobra).
- Dividendos: ya incluidos en el precio ajustado; nunca se suman aparte.
"""

from __future__ import annotations

from mrlab.domain.models import AssetClass
from mrlab.strategy.params import CostParams

DIVIDENDS_NOTE = (
    "Los precios de Yahoo están ajustados por splits y dividendos (auto_adjust=True); por eso el "
    "PnL no suma dividendos por separado (evita el doble conteo)."
)

BPS = 10_000.0


def slippage_bps(asset_class: AssetClass, costs: CostParams) -> float:
    if asset_class == AssetClass.CRYPTO:
        return costs.slippage_bps_crypto
    return costs.slippage_bps_equity


def transaction_costs(
    traded_notional_abs: float, asset_class: AssetClass, costs: CostParams
) -> tuple[float, float]:
    notional = abs(traded_notional_abs)
    commission = notional * costs.commission_bps / BPS
    slippage = notional * slippage_bps(asset_class, costs) / BPS
    return commission, slippage


def carry_costs(
    short_notional: float, gross_exposure: float, equity: float, days: int, costs: CostParams
) -> tuple[float, float]:
    if days <= 0 or gross_exposure <= 0:
        return 0.0, 0.0
    year_fraction = days / costs.day_count
    borrow = costs.borrow_rate_annual * abs(short_notional) * year_fraction
    financing = costs.financing_rate_annual * max(0.0, gross_exposure - equity) * year_fraction
    return borrow, financing
