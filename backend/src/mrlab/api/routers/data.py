"""Datos: descarga (con caché), CSV, sintéticos, caché, series y validación.

CONTRATO (a implementar):
- POST /data/load  LoadRequest {provider, symbols: list[str] (1..30, sin duplicados),
  start: date, end: date, refresh: bool=False} → LoadResponse {series: list[SeriesMeta],
  errors: list[{symbol, error: ErrorBody}]} (fallos parciales por símbolo con su error en
  español; si fallan TODOS → el error del primero con su HTTP). ``synthetic`` en la respuesta.
- POST /data/csv  multipart: file (UploadFile) + campos de formulario symbol, name, currency,
  asset_class, price_type, origin (opcional), seed (opcional) → SeriesMeta del CSV registrado y
  cargado en caché (CSVProvider.register(content, meta, filename=file.filename), luego
  DataService.load con el rango completo). Límite de tamaño settings.max_upload_mb → 413 en
  español. Metadatos inválidos → 422 CSV_METADATOS_INVALIDOS (build_metadata).
- POST /data/synthetic  SyntheticPairParams → SyntheticOut {a: SeriesMeta, b: SeriesMeta,
  label: "SINTÉTICO", note}: registra en el SyntheticProvider y carga ambas series por
  DataService (quedan en caché, reproducibles).
- GET /data/cache?provider=&symbol= → list[SeriesMeta] de la caché.
- GET /data/series?snapshot_id=&start=&end= → SeriesDataOut {meta: SeriesMeta, close:
  SeriesOut} (para gráficos; solo caché, nunca descarga).
- POST /data/validate  ValidateRequest {a: SeriesRef, b: SeriesRef, spread_type,
  min_train_obs, train_fraction} → ValidationReport (+ synthetic).
SeriesRef = mrlab.reporting.run_config.SeriesRef (snapshot_id o provider+symbol+start+end).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["datos"])
