"""Proveedor SINTÉTICO: pares generados con semilla fija. Siempre etiquetados.

CONTRATO:
- ``generate_pair(params)`` → (serie A, serie B), días hábiles lunes–viernes
  (``pd.bdate_range``, sin festivos) desde ``params.start``, ``n_obs`` puntos.
  RNG: ``np.random.default_rng(params.seed)``. Precios = exp(log-precio).
  Escenarios:
  * COINTEGRATED: log B = log(b_start_price) + paseo aleatorio (drift b_drift,
    vol b_vol); X = OU exacto (``mrlab.model.ou.simulate_ou_exact``) con
    (ou_lambda, ou_mu, ou_sigma, x0 o μ); log A = α + β·log B + X.
  * COINTEGRATED_LOW_CORRELATION: igual pero b_vol pequeño respecto a ou_sigma
    (valores por defecto del escenario: b_vol=0.004, ou_sigma=0.015, λ=0.1) →
    rendimientos poco correlacionados pero cointegrados.
  * CORRELATED_NOT_COINTEGRATED: dos paseos aleatorios con incrementos
    correlacionados ``rho`` → alta correlación, sin cointegración.
  * INDEPENDENT: dos paseos aleatorios independientes (se ignora ``rho``).
  Los valores por defecto específicos de cada escenario solo se aplican a los
  campos que el usuario no fijó explícitamente (usar ``model_fields_set``).
  Parámetros fuera de rango → ConfigError en español.
- Metadatos: origin=SINTETICO, price_type=SYNTHETIC, currency "USD",
  asset_class=EQUITY (calendario y costes de acciones), provider "synthetic",
  nombre que empiece por "SINTÉTICO — ", ``retrieved_at=None``,
  ``source_note`` con escenario, parámetros y semilla.
- Símbolos deterministas: presets fijos ("SINT-COI-A"/"SINT-COI-B",
  "SINT-CNC-A"/"SINT-CNC-B" correlado no cointegrado, "SINT-CBC-A"/"SINT-CBC-B"
  cointegrado baja correlación, "SINT-IND-A"/"SINT-IND-B" independientes; todos
  con semilla 7) y pares de usuario "SINT-U<hash8>-A/B" donde hash8 son los 8
  primeros hex del sha256 del JSON canónico de los parámetros EFECTIVOS (con los
  valores por defecto del escenario ya aplicados).
- ``SyntheticProvider.register(params)`` → (símbolo A, símbolo B) en un
  registro en memoria (los presets siempre existen). ``get_history`` genera la
  serie completa y recorta [start, end] inclusive (la serie no depende del
  rango pedido). Símbolo desconocido o rango sin datos →
  ProviderError(code="SIMBOLO_SIN_DATOS"). Los símbolos no distinguen mayúsculas.
- ``search``: subcadena sobre presets y registrados (consulta vacía → todos).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mrlab.data.calendar import to_date_index
from mrlab.data.providers.base import check_interval
from mrlab.domain.errors import ConfigError, ProviderError
from mrlab.domain.models import (
    AssetClass,
    AssetInfo,
    DataOrigin,
    PriceSeries,
    PriceType,
    SeriesMeta,
    coverage_of,
)
from mrlab.model.ou import simulate_ou_exact

SYNTHETIC_NAME_PREFIX = "SINTÉTICO — "
SYNTHETIC_NOTE = "Datos SINTÉTICOS generados por el laboratorio: no son precios históricos."
PRESET_SEED = 7
MAX_OBS = 20_000


class SyntheticScenario(StrEnum):
    COINTEGRATED = "cointegrado"
    COINTEGRATED_LOW_CORRELATION = "cointegrado_baja_correlacion"
    CORRELATED_NOT_COINTEGRATED = "correlado_no_cointegrado"
    INDEPENDENT = "independientes"


class SyntheticPairParams(BaseModel):
    scenario: SyntheticScenario = SyntheticScenario.COINTEGRATED
    n_obs: int = 1500
    start: date = date(2018, 1, 1)
    seed: int = 7
    alpha: float = 0.0
    beta: float = 1.0
    ou_lambda: float = 0.1
    ou_mu: float = 0.0
    ou_sigma: float = 0.01
    x0: float | None = None
    b_drift: float = 0.0002
    b_vol: float = 0.01
    a_drift: float = 0.0002  # solo escenarios no cointegrados
    a_vol: float = 0.01  # solo escenarios no cointegrados
    rho: float = 0.9
    b_start_price: float = 100.0
    a_start_price: float = 100.0  # solo escenarios no cointegrados


PRESET_SYMBOLS: dict[str, tuple[SyntheticScenario, str]] = {
    "SINT-COI": (SyntheticScenario.COINTEGRATED, "Par cointegrado"),
    "SINT-CBC": (
        SyntheticScenario.COINTEGRATED_LOW_CORRELATION,
        "Cointegrado con baja correlación",
    ),
    "SINT-CNC": (
        SyntheticScenario.CORRELATED_NOT_COINTEGRATED,
        "Correlacionado pero no cointegrado",
    ),
    "SINT-IND": (SyntheticScenario.INDEPENDENT, "Paseos aleatorios independientes"),
}

SCENARIO_LABELS: dict[SyntheticScenario, str] = {
    scenario: label for scenario, label in PRESET_SYMBOLS.values()
}

SCENARIO_DEFAULTS: dict[SyntheticScenario, dict[str, float]] = {
    SyntheticScenario.COINTEGRATED_LOW_CORRELATION: {
        "b_vol": 0.004,
        "ou_sigma": 0.015,
        "ou_lambda": 0.1,
    },
}

_COINTEGRATED = (SyntheticScenario.COINTEGRATED, SyntheticScenario.COINTEGRATED_LOW_CORRELATION)


def effective_params(params: SyntheticPairParams) -> SyntheticPairParams:
    """Aplica los valores por defecto del escenario a los campos que el usuario no fijó."""
    defaults = SCENARIO_DEFAULTS.get(params.scenario, {})
    update = {k: v for k, v in defaults.items() if k not in params.model_fields_set}
    return params.model_copy(update=update) if update else params


def params_hash(params: SyntheticPairParams) -> str:
    """sha256 del JSON canónico de los parámetros efectivos."""
    payload = json.dumps(
        effective_params(params).model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def preset_params(prefix: str) -> SyntheticPairParams:
    scenario, _ = PRESET_SYMBOLS[prefix]
    return SyntheticPairParams(scenario=scenario, seed=PRESET_SEED)


def user_prefix(params: SyntheticPairParams) -> str:
    return f"SINT-U{params_hash(params)[:8].upper()}"


def _check(p: SyntheticPairParams) -> None:
    problems: list[str] = []
    if not 2 <= p.n_obs <= MAX_OBS:
        problems.append(f"el número de observaciones debe estar entre 2 y {MAX_OBS}")
    if p.b_vol < 0 or p.a_vol < 0:
        problems.append("las volatilidades no pueden ser negativas")
    if p.b_start_price <= 0 or p.a_start_price <= 0:
        problems.append("los precios iniciales deben ser positivos")
    if p.scenario == SyntheticScenario.CORRELATED_NOT_COINTEGRATED and not -1 <= p.rho <= 1:
        problems.append("la correlación ρ debe estar entre −1 y 1")
    if p.scenario in _COINTEGRATED and (p.ou_lambda <= 0 or p.ou_sigma <= 0):
        problems.append("λ y σ del OU deben ser positivos")
    values = [v for v in p.model_dump().values() if isinstance(v, float)]
    if not all(math.isfinite(v) for v in values):
        problems.append("todos los parámetros deben ser números finitos")
    if problems:
        raise ConfigError(
            "Parámetros del generador sintético no válidos: " + "; ".join(problems) + ".",
            code="PARAMETROS_SINTETICOS_INVALIDOS",
            details={"params": p.model_dump(mode="json")},
        )


def _random_walk(start_price: float, drift: float, vol: float, shocks: np.ndarray) -> np.ndarray:
    """Log-precio: arranca exactamente en log(start_price); ``shocks`` tiene n − 1 normales."""
    steps = drift + vol * shocks
    return math.log(start_price) + np.concatenate([[0.0], np.cumsum(steps)])


def _log_prices(p: SyntheticPairParams) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(p.seed)
    n = p.n_obs
    if p.scenario in _COINTEGRATED:
        log_b = _random_walk(p.b_start_price, p.b_drift, p.b_vol, rng.standard_normal(n - 1))
        ou_seed = int(rng.integers(0, 2**63 - 1))
        x0 = p.ou_mu if p.x0 is None else p.x0
        x = simulate_ou_exact(p.ou_lambda, p.ou_mu, p.ou_sigma, x0, n_steps=n - 1, seed=ou_seed)[0]
        return p.alpha + p.beta * log_b + x, log_b
    rho = p.rho if p.scenario == SyntheticScenario.CORRELATED_NOT_COINTEGRATED else 0.0
    z_b = rng.standard_normal(n - 1)
    z_other = rng.standard_normal(n - 1)
    z_a = rho * z_b + math.sqrt(1.0 - rho * rho) * z_other
    log_b = _random_walk(p.b_start_price, p.b_drift, p.b_vol, z_b)
    log_a = _random_walk(p.a_start_price, p.a_drift, p.a_vol, z_a)
    return log_a, log_b


def _source_note(p: SyntheticPairParams) -> str:
    label = SCENARIO_LABELS[p.scenario]
    if p.scenario in _COINTEGRATED:
        x0 = p.ou_mu if p.x0 is None else p.x0
        model = (
            f"log A = {p.alpha:g} + {p.beta:g}·log B + X, con X un OU exacto (λ={p.ou_lambda:g}, "
            f"μ={p.ou_mu:g}, σ={p.ou_sigma:g}, X₀={x0:g}); log B es un paseo aleatorio "
            f"(deriva {p.b_drift:g}, volatilidad {p.b_vol:g})."
        )
    else:
        rho = p.rho if p.scenario == SyntheticScenario.CORRELATED_NOT_COINTEGRATED else 0.0
        model = (
            f"Dos paseos aleatorios en log-precio con incrementos de correlación ρ={rho:g} "
            f"(A: deriva {p.a_drift:g}, volatilidad {p.a_vol:g}; B: deriva {p.b_drift:g}, "
            f"volatilidad {p.b_vol:g})."
        )
    params = json.dumps(p.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return f"SINTÉTICO — escenario «{label}», semilla {p.seed}. {model} Parámetros: {params}"


def _asset(symbol: str, leg: str, label: str) -> AssetInfo:
    return AssetInfo(
        symbol=symbol,
        name=f"{SYNTHETIC_NAME_PREFIX}{label}, serie {leg}",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        provider="synthetic",
        origin=DataOrigin.SYNTHETIC,
        executable=True,
        notes=(SYNTHETIC_NOTE,),
    )


def _label(prefix: str, params: SyntheticPairParams) -> str:
    if prefix in PRESET_SYMBOLS:
        return PRESET_SYMBOLS[prefix][1]
    return f"{SCENARIO_LABELS[params.scenario]} (usuario {prefix.removeprefix('SINT-U')})"


def generate_pair(
    params: SyntheticPairParams, symbol_prefix: str | None = None
) -> tuple[PriceSeries, PriceSeries]:
    p = effective_params(params)
    _check(p)
    prefix = symbol_prefix or user_prefix(p)
    log_a, log_b = _log_prices(p)
    index = to_date_index(pd.bdate_range(start=p.start, periods=p.n_obs))
    note = _source_note(p)
    label = _label(prefix, p)

    def build(leg: str, log_price: np.ndarray) -> PriceSeries:
        data = pd.DataFrame({"close": np.exp(log_price)}, index=index)
        cov_start, cov_end, n_obs = coverage_of(data)
        meta = SeriesMeta(
            asset=_asset(f"{prefix}-{leg}", leg, label),
            price_type=PriceType.SYNTHETIC,
            retrieved_at=None,
            coverage_start=cov_start,
            coverage_end=cov_end,
            n_obs=n_obs,
            source_note=note,
        )
        return PriceSeries(meta=meta, data=data)

    return build("A", log_a), build("B", log_b)


def _split_symbol(symbol: str) -> tuple[str, str]:
    sym = (symbol or "").strip().upper()
    if sym.endswith(("-A", "-B")):
        return sym[:-2], sym[-1]
    return sym, ""


class SyntheticProvider:
    name = "synthetic"
    description = "Datos SINTÉTICOS generados con semilla fija (no son históricos)."

    def __init__(self) -> None:
        self._registry: dict[str, SyntheticPairParams] = {}

    def register(self, params: SyntheticPairParams) -> tuple[str, str]:
        p = effective_params(params)
        _check(p)
        prefix = user_prefix(p)
        self._registry[prefix] = p
        return f"{prefix}-A", f"{prefix}-B"

    def search(self, query: str) -> list[AssetInfo]:
        q = query.strip().lower()
        found: list[AssetInfo] = []
        for prefix, params in self._pairs().items():
            for leg in ("A", "B"):
                asset = _asset(f"{prefix}-{leg}", leg, _label(prefix, params))
                if not q or q in asset.symbol.lower() or q in asset.name.lower():
                    found.append(asset)
        return found

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d") -> PriceSeries:
        check_interval(interval)
        prefix, leg = self._resolve(symbol)
        series_a, series_b = generate_pair(self._pairs()[prefix], symbol_prefix=prefix)
        series = series_a if leg == "A" else series_b
        idx = series.data.index
        data = series.data[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        if data.empty:
            raise ProviderError(
                f"La serie sintética «{series.symbol}» no tiene datos entre {start.isoformat()} y "
                f"{end.isoformat()}: cubre de {series.meta.coverage_start} a "
                f"{series.meta.coverage_end}.",
                code="SIMBOLO_SIN_DATOS",
                details={"symbol": series.symbol},
            )
        cov_start, cov_end, n_obs = coverage_of(data)
        return series.with_data(data).with_meta(
            requested_start=start,
            requested_end=end,
            coverage_start=cov_start,
            coverage_end=cov_end,
            n_obs=n_obs,
        )

    def get_metadata(self, symbol: str) -> AssetInfo:
        prefix, leg = self._resolve(symbol)
        return _asset(f"{prefix}-{leg}", leg, _label(prefix, self._pairs()[prefix]))

    # ------------------------------------------------------------------ internos

    def _pairs(self) -> dict[str, SyntheticPairParams]:
        return {**{prefix: preset_params(prefix) for prefix in PRESET_SYMBOLS}, **self._registry}

    def _resolve(self, symbol: str) -> tuple[str, str]:
        prefix, leg = _split_symbol(symbol)
        if not leg or prefix not in self._pairs():
            available = ", ".join(f"{p}-A/B" for p in self._pairs())
            raise ProviderError(
                f"No existe la serie sintética «{symbol}». Disponibles: {available}. Los pares "
                "generados con parámetros propios hay que registrarlos de nuevo tras reiniciar.",
                code="SIMBOLO_SIN_DATOS",
                details={"symbol": symbol},
            )
        return prefix, leg
