"""Parámetros de estrategia y costes (secciones 7.1 y 9). Valores por defecto de la especificación."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from mrlab.domain.errors import ConfigError


class StrategyParams(BaseModel):
    entry_z: float = Field(2.0, gt=0, description="Entrada cuando |z| supera este umbral")
    exit_z: float = Field(0.5, ge=0, description="Salida normal al volver a este nivel")
    stop_z: float = Field(3.5, gt=0, description="Stop-loss en z")
    stop_loss_pct: float = Field(0.05, gt=0, le=1, description="Stop monetario: fracción del capital asignado al par")
    max_holding_mult: float = Field(3.0, gt=0, description="Tiempo máximo = mult × vida media")
    max_holding_min: int = Field(5, ge=1, description="Mínimo del tiempo máximo, en observaciones")
    max_holding_override: int | None = Field(None, ge=1, description="Fija el tiempo máximo (ignora la vida media)")
    cooldown_bars: int = Field(5, ge=0, description="Enfriamiento tras un stop, en observaciones")
    rearm: bool = Field(True, description="Tras stop o tiempo máximo, exigir que z vuelva dentro de ±entrada")
    initial_capital: float = Field(10_000.0, gt=0)
    gross_exposure_target: float = Field(1.0, gt=0, description="Exposición bruta objetivo / capital")
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


class CostParams(BaseModel):
    commission_bps: float = Field(5.0, ge=0, description="Comisión por pierna, pb del nocional operado")
    slippage_bps_equity: float = Field(5.0, ge=0, description="Spread bid-ask/slippage por pierna (acciones y ETF)")
    slippage_bps_crypto: float = Field(10.0, ge=0, description="Spread bid-ask/slippage por pierna (cripto)")
    borrow_rate_annual: float = Field(0.03, ge=0, description="Préstamo de cortos, anual sobre el nocional corto")
    financing_rate_annual: float = Field(0.05, ge=0, description="Financiación del exceso sobre 1×, anual")
    day_count: int = Field(365, description="Base de días naturales para devengos")
