"""Proveedor SINTÉTICO: pares generados con semilla fija. Siempre etiquetados.

CONTRATO (a implementar):
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
  * INDEPENDENT: dos paseos aleatorios independientes.
  Los valores por defecto específicos de cada escenario solo se aplican a los
  campos que el usuario no fijó explícitamente (usar ``model_fields_set``).
- Metadatos: origin=SINTETICO, price_type=SYNTHETIC, currency "USD",
  asset_class=EQUITY (calendario y costes de acciones), provider "synthetic",
  nombre que empiece por "SINTÉTICO — ", ``retrieved_at=None``,
  ``source_note`` con escenario, parámetros y semilla.
- Símbolos deterministas: presets fijos ("SINT-COI-A"/"SINT-COI-B",
  "SINT-CNC-A"/"SINT-CNC-B" correlado no cointegrado, "SINT-CBC-A"/"SINT-CBC-B"
  cointegrado baja correlación, "SINT-IND-A"/"SINT-IND-B" independientes; todos
  con semilla 7) y pares de usuario "SINT-U<hash8>-A/B" donde hash8 son los 8
  primeros hex del sha256 del JSON canónico de los parámetros.
- ``SyntheticProvider.register(params)`` → (símbolo A, símbolo B) en un
  registro en memoria (los presets siempre existen). ``get_history`` genera la
  serie completa y recorta [start, end] inclusive (la serie no depende del
  rango pedido). Símbolo desconocido → ProviderError(code="SIMBOLO_SIN_DATOS").
- ``search``: subcadena sobre presets y registrados.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel

from mrlab.domain.models import AssetInfo, PriceSeries


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
    "SINT-CBC": (SyntheticScenario.COINTEGRATED_LOW_CORRELATION, "Cointegrado con baja correlación"),
    "SINT-CNC": (SyntheticScenario.CORRELATED_NOT_COINTEGRATED, "Correlacionado pero no cointegrado"),
    "SINT-IND": (SyntheticScenario.INDEPENDENT, "Paseos aleatorios independientes"),
}


def generate_pair(params: SyntheticPairParams, symbol_prefix: str | None = None) -> tuple[PriceSeries, PriceSeries]:
    raise NotImplementedError


class SyntheticProvider:
    name = "synthetic"
    description = "Datos SINTÉTICOS generados con semilla fija (no son históricos)."

    def __init__(self) -> None:
        self._registry: dict[str, SyntheticPairParams] = {}

    def register(self, params: SyntheticPairParams) -> tuple[str, str]:
        raise NotImplementedError

    def search(self, query: str) -> list[AssetInfo]:
        raise NotImplementedError

    def get_history(self, symbol: str, start: date, end: date, interval: str = "1d") -> PriceSeries:
        raise NotImplementedError

    def get_metadata(self, symbol: str) -> AssetInfo:
        raise NotImplementedError
