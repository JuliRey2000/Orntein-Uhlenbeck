"""Parámetros de estrategia y costes (secciones 7.1 y 9). Valores por defecto de la especificación.

Precisiones de la implementación:
- ``CostParams`` valida rangos razonables con mensajes en español (ConfigError
  ``COSTES_INVALIDOS``, con la ruta del campo, p. ej. ``costs.day_count``, y el
  valor recibido): pb de comisión y slippage en [0, 1.000] (hasta 10 % por
  pierna); tasas anuales de préstamo y financiación en [0, 1] (fracción anual:
  0,03 = 3 %); ``day_count`` en (0, 366] (convenciones habituales: 365 y 360).
  Un ``run_config.json`` editado a mano no puede así producir una división por
  cero ni convertir costes en ingresos. Los ``Field`` llevan las mismas cotas
  (esquema y defensa adicional).
"""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field, model_validator

from mrlab.domain.errors import ConfigError


class StrategyParams(BaseModel):
    entry_z: float = Field(2.0, gt=0, description="Entrada cuando |z| supera este umbral")
    exit_z: float = Field(0.5, ge=0, description="Salida normal al volver a este nivel")
    stop_z: float = Field(3.5, gt=0, description="Stop-loss en z")
    stop_loss_pct: float = Field(
        0.05, gt=0, le=1, description="Stop monetario: fracción del capital asignado al par"
    )
    max_holding_mult: float = Field(3.0, gt=0, description="Tiempo máximo = mult × vida media")
    max_holding_min: int = Field(5, ge=1, description="Mínimo del tiempo máximo, en observaciones")
    max_holding_override: int | None = Field(
        None, ge=1, description="Fija el tiempo máximo (ignora la vida media)"
    )
    cooldown_bars: int = Field(5, ge=0, description="Enfriamiento tras un stop, en observaciones")
    rearm: bool = Field(
        True, description="Tras stop o tiempo máximo, exigir que z vuelva dentro de ±entrada"
    )
    initial_capital: float = Field(10_000.0, gt=0)
    gross_exposure_target: float = Field(
        1.0, gt=0, description="Exposición bruta objetivo / capital"
    )
    max_leverage: float = Field(2.0, gt=0, description="Exposición bruta máxima / capital")
    allow_short: bool = True
    integer_units: bool = Field(False, description="Redondear cantidades a enteros (hacia cero)")
    zscore_mode: Literal["fijo", "movil"] = "fijo"
    rolling_window: int = Field(63, ge=5)

    @model_validator(mode="after")
    def _check_thresholds(self) -> StrategyParams:
        if not (self.exit_z < self.entry_z < self.stop_z):
            raise ConfigError(
                "Los umbrales deben cumplir salida < entrada < stop "
                f"(ahora: salida={self.exit_z}, entrada={self.entry_z}, stop={self.stop_z}).",
                code="UMBRALES_INCONSISTENTES",
            )
        return self

    def max_holding_bars(self, half_life: float | None) -> int:
        """3 × vida media redondeado, mínimo 5 (o el valor fijado por el usuario)."""
        if self.max_holding_override is not None:
            return self.max_holding_override
        if half_life is None:
            return self.max_holding_min
        return max(self.max_holding_min, int(round(self.max_holding_mult * half_life)))


MAX_COST_BPS = 1_000.0  # 10 % del nocional por pierna
MAX_ANNUAL_RATE = 1.0  # 100 % anual
MAX_DAY_COUNT = 366


class _Range(NamedTuple):
    low: float
    low_included: bool
    high: float
    label: str
    unit: str


_BPS = "puntos básicos"
_COST_RANGES: dict[str, _Range] = {
    "commission_bps": _Range(0.0, True, MAX_COST_BPS, "comisión por pierna", _BPS),
    "slippage_bps_equity": _Range(0.0, True, MAX_COST_BPS, "slippage en acciones y ETF", _BPS),
    "slippage_bps_crypto": _Range(0.0, True, MAX_COST_BPS, "slippage en cripto", _BPS),
    "borrow_rate_annual": _Range(
        0.0, True, MAX_ANNUAL_RATE, "tasa anual de préstamo de cortos", "fracción: 0,03 = 3 %"
    ),
    "financing_rate_annual": _Range(
        0.0, True, MAX_ANNUAL_RATE, "tasa anual de financiación", "fracción: 0,05 = 5 %"
    ),
    "day_count": _Range(0.0, False, MAX_DAY_COUNT, "base de días para devengos", "días"),
}


def _num(x: float) -> str:
    """Número para un mensaje: coma decimal, sin ceros sobrantes."""
    if not math.isfinite(x):
        return str(x)
    return format(x, ".10g").replace(".", ",")


class CostParams(BaseModel):
    # Las cotas de ``Field`` repiten las de ``_COST_RANGES`` (esquema y defensa adicional).
    commission_bps: float = Field(
        5.0, ge=0, le=MAX_COST_BPS, description="Comisión por pierna, pb del nocional operado"
    )
    slippage_bps_equity: float = Field(
        5.0, ge=0, le=MAX_COST_BPS, description="Slippage por pierna, pb (acciones y ETF)"
    )
    slippage_bps_crypto: float = Field(
        10.0, ge=0, le=MAX_COST_BPS, description="Slippage por pierna, pb (cripto)"
    )
    borrow_rate_annual: float = Field(
        0.03, ge=0, le=MAX_ANNUAL_RATE, description="Préstamo de cortos, anual sobre el corto"
    )
    financing_rate_annual: float = Field(
        0.05, ge=0, le=MAX_ANNUAL_RATE, description="Financiación del exceso sobre 1×, anual"
    )
    day_count: int = Field(
        365, gt=0, le=MAX_DAY_COUNT, description="Base de días naturales para devengos (C2: 365)"
    )

    @model_validator(mode="before")
    @classmethod
    def _check_ranges(cls, data: Any) -> Any:
        """Rangos con mensaje en español (antes de las cotas de ``Field``, que darían inglés)."""
        if not isinstance(data, dict):
            return data
        for name, (low, low_included, high, label, unit) in _COST_RANGES.items():
            value = data.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue  # ausente o de otro tipo: lo valida pydantic
            x = float(value)
            above_low = x >= low if low_included else x > low
            if math.isfinite(x) and above_low and x <= high:
                continue
            bound = "mayor o igual que" if low_included else "mayor que"
            message = (
                f"Costes: «costs.{name}» ({label}) debe ser {bound} {_num(low)} y como máximo "
                f"{_num(high)} ({unit}); recibido: {_num(x)}."
            )
            if name.endswith("_rate_annual") and 1 < x <= 100:
                message += f" Si quisiste escribir {_num(x)} %, usa {_num(x / 100)}."
            if name == "day_count":
                message += " Las convenciones habituales son 365 y 360."
            raise ConfigError(
                message,
                code="COSTES_INVALIDOS",
                details={"campo": f"costs.{name}", "valor": x, "minimo": low, "maximo": high},
            )
        return data
