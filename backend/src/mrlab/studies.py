"""Estudios: configuración congelada y registro persistente de la prueba (sección 8.1, decisión S10).

Un estudio es la ÚNICA vía de la API para ejecutar el segmento de prueba. Se guarda en
``<studies_dir>/<id>.json`` y sobrevive a reinicios (a diferencia de ``RunConfig.test_history``,
que viaja dentro del archivo; ver S10a/S10b).

Reglas (CONTRATO, a implementar):
- ``create(config, data, name)``: ejecuta ``run_pair(config, data, include_test=False)`` para
  fijar instantáneas (``snapshot_id``) y el split congelado; guarda el RunConfig devuelto (con
  ``include_test=False`` y ``test_history`` vacío). Rechaza (ConfigError en español) un config con
  ``include_test=True``: la prueba se ejecuta solo con ``run_test``.
- ``update_config(id, strategy=None, costs=None, analysis=None)``: solo cambian parámetros; los
  datos (pair, snapshots, frozen_split) NO cambian (para otro par se crea otro estudio). Cada
  cambio queda en ``changes`` (qué campos, hash antes/después, fecha). Si la prueba ya se vio
  (``test_runs`` no vacío) y el hash cambia → ``contaminated = True`` PERMANENTE con
  ``contamination_reason`` en español ("se modificaron umbrales después de ver la prueba").
  Cambiar ``analysis`` (p. ej. α, ventana) también cuenta.
- ``run_test(id, data)``: ejecuta ``run_pair`` con la configuración actual e ``include_test=True``
  y ``test_history`` = hashes de ``test_runs``; registra la ejecución (hash, fecha); si el hash
  difiere de ejecuciones anteriores → contaminado (el runner ya lo detecta con test_history;
  combina ambas señales). Devuelve (Study, RunResult) con ``result.test_contaminated`` igual al
  estado del estudio (True si contaminado por cualquiera de las dos vías).
- ``result(id, data, include_test)``: re-ejecuta (determinista, desde caché) para exportar o
  mostrar; ``include_test`` solo permitido si ya se ejecutó la prueba (si no → ConfigError
  "ejecuta primero la prueba"). El RunResult refleja ``contaminated``.
- ``get``, ``list`` (más recientes primero), ``delete``. Id = uuid4 hex; ids con otro formato →
  DataError(code="ESTUDIO_NO_ENCONTRADO") (evita rutas peligrosas). Escritura atómica
  (temporal + os.replace) y lock por proceso. Marcas de tiempo UTC ISO.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from mrlab.analysis.pair_pipeline import AnalysisConfig
from mrlab.analysis.runner import RunResult
from mrlab.data.service import DataService
from mrlab.reporting.run_config import RunConfig
from mrlab.strategy.params import CostParams, StrategyParams


class StudyChange(BaseModel):
    at: datetime
    fields: list[str]
    hash_before: str
    hash_after: str
    after_test: bool


class StudyTestRun(BaseModel):
    at: datetime
    config_hash: str


class Study(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    config: RunConfig
    test_runs: list[StudyTestRun] = Field(default_factory=list)
    changes: list[StudyChange] = Field(default_factory=list)
    contaminated: bool = False
    contamination_reason: str | None = None

    @property
    def test_viewed(self) -> bool:
        return bool(self.test_runs)


class StudyStore:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def create(self, config: RunConfig, data: DataService, name: str = "") -> Study:
        raise NotImplementedError

    def get(self, study_id: str) -> Study:
        raise NotImplementedError

    def list(self) -> list[Study]:
        raise NotImplementedError

    def delete(self, study_id: str) -> None:
        raise NotImplementedError

    def update_config(
        self,
        study_id: str,
        *,
        strategy: StrategyParams | None = None,
        costs: CostParams | None = None,
        analysis: AnalysisConfig | None = None,
    ) -> Study:
        raise NotImplementedError

    def run_test(self, study_id: str, data: DataService) -> tuple[Study, RunResult]:
        raise NotImplementedError

    def result(self, study_id: str, data: DataService, include_test: bool = False) -> RunResult:
        raise NotImplementedError
