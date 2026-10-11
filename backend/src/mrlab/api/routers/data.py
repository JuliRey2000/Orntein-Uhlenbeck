"""Datos: descarga (con caché), CSV, sintéticos, caché, series y validación.

CONTRATO:
- POST /data/load  LoadRequest {provider, symbols: list[str] (1..30, sin duplicados),
  start: date, end: date, refresh: bool=False} → LoadResponse {series: list[SeriesMeta],
  errors: list[{symbol, status, error: ErrorBody}], cached, synthetic, disclaimer} (fallos
  parciales por símbolo con su error en español; si fallan TODOS → el error del primero con su
  HTTP: ERROR_RED 503, LIMITE_PETICIONES 429 con Retry-After, símbolo inexistente 404). Usa la
  caché (``DataService``); ``refresh`` fuerza la descarga.
- POST /data/csv  multipart: file (UploadFile) + campos de formulario symbol, name, currency,
  asset_class, price_type, origin (opcional), seed (opcional) → CSVUploadOut {series: SeriesMeta,
  synthetic, synthetic_forced, notices, disclaimer} del CSV registrado y cargado en caché
  (CSVProvider.register(content, meta, filename=file.filename), luego DataService.load con el
  rango completo). Límite de tamaño settings.max_upload_mb → 413 ARCHIVO_DEMASIADO_GRANDE en
  español: se rechaza por la cabecera Content-Length antes de leer el cuerpo y, si no, al leer el
  archivo en bloques (nunca se carga entero en memoria si supera el límite). Metadatos inválidos →
  422 CSV_METADATOS_INVALIDOS (build_metadata). Un CSV con la columna tipo_datos = SINTETICO o con
  nombre SINTETICO_* queda SINTÉTICO aunque el formulario diga histórico, y la respuesta lo avisa.
- POST /data/synthetic  SyntheticRequest → SyntheticOut {a, b, label: "SINTÉTICO", note, ...}:
  registra en el SyntheticProvider y carga ambas series por DataService (quedan en caché,
  reproducibles con su snapshot_id).
- GET /data/cache?provider=&symbol= → list[SeriesMeta] de la caché (nunca descarga).
- GET /data/series?snapshot_id=&start=&end= → SeriesDataOut {meta, close: SeriesOut} (para
  gráficos; solo caché, nunca descarga; nunca rellena fechas).
- POST /data/validate  ValidateRequest {a: SeriesRef, b: SeriesRef, spread_type,
  min_train_obs, train_fraction} → ValidationOut (ValidationReport + series + synthetic).
SeriesRef: snapshot_id (solo caché) o provider + symbol + start + end.
"""

from __future__ import annotations

import threading
from datetime import date
from typing import Any, BinaryIO

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.responses import Response

from mrlab import DISCLAIMER
from mrlab.api.errors import RequestValueError, json_safe, lab_error_status
from mrlab.api.openapi_models import EducationalOut, SeriesMetaOut, ValidationReportOut
from mrlab.api.schemas import ERROR_RESPONSES, ErrorBody, ErrorResponse, SeriesOut, series_out
from mrlab.api.state import AppState, get_state
from mrlab.config import Settings
from mrlab.data.providers.csv_provider import CSVProvider, build_metadata
from mrlab.data.providers.synthetic import (
    MAX_OBS,
    SCENARIO_LABELS,
    SYNTHETIC_NOTE,
    SyntheticPairParams,
    SyntheticProvider,
    SyntheticScenario,
    effective_params,
)
from mrlab.data.service import DataService
from mrlab.data.validation import MIN_TRAIN_OBS, validate_pair
from mrlab.domain.errors import ConfigError, LabError
from mrlab.domain.models import AssetClass, DataOrigin, PriceSeries, PriceType, SpreadType

OK = "Respuesta correcta."
MAX_SYMBOLS = 30
MAX_SYMBOL_CHARS = 40
CHUNK_BYTES = 64 * 1024
#: Margen sobre el límite del archivo para la cabecera Content-Length (campos y separadores).
MULTIPART_OVERHEAD = 64 * 1024
SYNTHETIC_LABEL = "SINTÉTICO"

_csv_lock = threading.Lock()

DATA_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    429: {"model": ErrorResponse, "description": "El proveedor limitó las peticiones"},
    503: {"model": ErrorResponse, "description": "Sin conexión con el proveedor de datos"},
}
UPLOAD_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    413: {"model": ErrorResponse, "description": "El archivo supera el tamaño máximo"},
}


def _values(enum: Any) -> str:
    return ", ".join(m.value for m in enum)


# --------------------------------------------------------------------------- límites de subida


def max_upload_bytes(settings: Settings) -> int:
    return max(0, int(settings.max_upload_mb * 1024 * 1024))


def _mb(settings: Settings) -> str:
    return f"{settings.max_upload_mb:g}".replace(".", ",")


def too_large(settings: Settings) -> HTTPException:
    limit = max_upload_bytes(settings)
    return HTTPException(
        status_code=413,
        detail={
            "code": "ARCHIVO_DEMASIADO_GRANDE",
            "message": (
                f"El archivo supera el tamaño máximo permitido ({_mb(settings)} MB). Reduce el "
                "rango de fechas o elimina columnas que no se usan (basta con date y close). El "
                "límite lo configura quien administra el servidor (variable MRLAB_MAX_UPLOAD_MB)."
            ),
            "details": {"max_mb": settings.max_upload_mb, "max_bytes": limit},
        },
    )


def read_limited(stream: BinaryIO, limit: int, settings: Settings) -> bytes:
    """Lee en bloques; si supera ``limit`` bytes lanza 413 sin seguir leyendo."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise too_large(settings)
        chunks.append(chunk)
    return b"".join(chunks)


class UploadLimitRoute(APIRoute):
    """Rechaza con 413 por Content-Length ANTES de que se lea y procese el cuerpo multipart."""

    def get_route_handler(self):  # type: ignore[no-untyped-def]
        handler = super().get_route_handler()

        async def limited(request: Request) -> Response:
            settings: Settings = request.app.state.lab.settings
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > max_upload_bytes(settings) + MULTIPART_OVERHEAD:
                raise too_large(settings)
            return await handler(request)

        return limited


router = APIRouter(tags=["datos"])
_upload_router = APIRouter(route_class=UploadLimitRoute)


# --------------------------------------------------------------------------- modelos


class SeriesRefIn(BaseModel):
    """Referencia a una serie: por instantánea de la caché o por proveedor + rango."""

    provider: str = Field(
        min_length=1, max_length=40, description="Proveedor (p. ej. «yahoo», «csv», «synthetic»)."
    )
    symbol: str = Field(min_length=1, max_length=MAX_SYMBOL_CHARS, description="Símbolo.")
    snapshot_id: str | None = Field(
        None,
        max_length=200,
        description=(
            "Instantánea de la caché. Si se indica, se usa SOLO la caché (reproducción exacta, "
            "nunca descarga)."
        ),
    )
    start: date | None = Field(
        None, description="Inicio (AAAA-MM-DD). Obligatorio sin snapshot_id; con él, recorta."
    )
    end: date | None = Field(
        None, description="Fin (AAAA-MM-DD). Obligatorio sin snapshot_id; con él, recorta."
    )

    @model_validator(mode="after")
    def _check_range(self) -> SeriesRefIn:
        if not self.snapshot_id and (self.start is None or self.end is None):
            raise RequestValueError(
                "sin «snapshot_id» hay que indicar «start» y «end» (formato AAAA-MM-DD) para "
                "cargar la serie del proveedor"
            )
        if self.start is not None and self.end is not None and self.start > self.end:
            raise RequestValueError(
                f"la fecha de inicio ({self.start.isoformat()}) es posterior a la de fin "
                f"({self.end.isoformat()})"
            )
        return self


def load_ref(data: DataService, ref: SeriesRefIn, role: str) -> PriceSeries:
    """Carga una referencia: con ``snapshot_id`` solo desde la caché; si no, caché o proveedor."""
    if ref.snapshot_id:
        series = data.load_snapshot(ref.snapshot_id, ref.start, ref.end)
        if (
            series.meta.provider != ref.provider
            or series.symbol.strip().upper() != ref.symbol.strip().upper()
        ):
            raise ConfigError(
                f"La instantánea «{ref.snapshot_id}» contiene «{series.symbol}» del proveedor "
                f"«{series.meta.provider}», pero la serie {role} pide «{ref.symbol}» de "
                f"«{ref.provider}». Revisa la referencia.",
                code="REFERENCIA_INCONSISTENTE",
                details={"serie": role, "snapshot_id": ref.snapshot_id},
            )
        return series
    assert ref.start is not None and ref.end is not None  # garantizado por el validador
    return data.load(ref.provider, ref.symbol.strip(), ref.start, ref.end)


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(
        None,
        max_length=40,
        description="Proveedor (p. ej. «yahoo»). Si se omite, el proveedor por defecto.",
    )
    symbols: list[str] = Field(
        min_length=1,
        max_length=MAX_SYMBOLS,
        description="Símbolos a cargar (de 1 a 30, sin repetir), p. ej. [\"KO\", \"PEP\"].",
    )
    start: date = Field(description="Fecha de inicio (AAAA-MM-DD), inclusive.")
    end: date = Field(description="Fecha de fin (AAAA-MM-DD), inclusive.")
    refresh: bool = Field(
        False,
        description=(
            "True para forzar una descarga nueva aunque haya datos en caché. Una descarga nueva "
            "puede traer precios distintos (otro snapshot_id): los análisis anteriores siguen "
            "reproduciéndose con su instantánea."
        ),
    )

    @field_validator("symbols")
    @classmethod
    def _check_symbols(cls, symbols: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: dict[str, str] = {}
        for pos, raw in enumerate(symbols, start=1):
            sym = raw.strip()
            if not sym:
                raise RequestValueError(f"el símbolo en la posición {pos} está vacío")
            if len(sym) > MAX_SYMBOL_CHARS:
                raise RequestValueError(
                    f"el símbolo en la posición {pos} es demasiado largo (máximo "
                    f"{MAX_SYMBOL_CHARS} caracteres)"
                )
            key = sym.upper()
            if key in seen:
                raise RequestValueError(
                    f"«{sym}» está repetido (ya aparece como «{seen[key]}»): cada símbolo debe "
                    "aparecer una sola vez"
                )
            seen[key] = sym
            cleaned.append(sym)
        return cleaned

    @model_validator(mode="after")
    def _check_range(self) -> LoadRequest:
        if self.start > self.end:
            raise RequestValueError(
                f"la fecha de inicio ({self.start.isoformat()}) es posterior a la de fin "
                f"({self.end.isoformat()})"
            )
        return self


class SymbolErrorOut(BaseModel):
    symbol: str = Field(description="Símbolo que no se pudo cargar.")
    status: int = Field(description="Código HTTP que tendría este error por sí solo.")
    error: ErrorBody = Field(description="Error en español (code, message, details).")


class LoadResponse(EducationalOut):
    series: list[SeriesMetaOut] = Field(
        description="Series cargadas (en el orden pedido), con su snapshot_id."
    )
    errors: list[SymbolErrorOut] = Field(
        default_factory=list, description="Símbolos que fallaron, cada uno con su error."
    )
    cached: list[str] = Field(
        default_factory=list,
        description="Símbolos servidos desde la caché local, sin descargar.",
    )


class CSVUploadOut(EducationalOut):
    series: SeriesMetaOut = Field(description="Serie registrada y guardada en la caché.")
    synthetic_forced: bool = Field(
        False,
        description=(
            "True si el archivo se trata como SINTÉTICO aunque el formulario declaraba datos "
            "históricos (columna tipo_datos o nombre SINTETICO_…)."
        ),
    )
    notices: list[str] = Field(default_factory=list, description="Avisos en español.")


class SyntheticRequest(BaseModel):
    """Parámetros del generador SINTÉTICO (los no indicados toman el valor del escenario)."""

    model_config = ConfigDict(extra="forbid")

    scenario: SyntheticScenario = Field(
        SyntheticScenario.COINTEGRATED,
        description=f"Escenario: {_values(SyntheticScenario)}.",
    )
    n_obs: int = Field(1500, ge=2, le=MAX_OBS, description="Observaciones (días hábiles).")
    start: date = Field(date(2018, 1, 1), description="Primera fecha (lunes a viernes).")
    seed: int = Field(7, ge=0, le=2**63 - 1, description="Semilla (entero ≥ 0): reproducible.")
    alpha: float = Field(0.0, allow_inf_nan=False, description="α de la relación log A = α + β·log B + X.")
    beta: float = Field(1.0, allow_inf_nan=False, description="β de la relación de cointegración.")
    ou_lambda: float = Field(
        0.1, gt=0, allow_inf_nan=False, description="λ del spread OU (1/día); vida media ln2/λ."
    )
    ou_mu: float = Field(0.0, allow_inf_nan=False, description="μ del spread OU.")
    ou_sigma: float = Field(0.01, gt=0, allow_inf_nan=False, description="σ del spread OU.")
    x0: float | None = Field(
        None, allow_inf_nan=False, description="Valor inicial del spread (por defecto μ)."
    )
    b_drift: float = Field(0.0002, allow_inf_nan=False, description="Deriva diaria de log B.")
    b_vol: float = Field(0.01, ge=0, allow_inf_nan=False, description="Volatilidad diaria de B.")
    a_drift: float = Field(
        0.0002, allow_inf_nan=False, description="Deriva de log A (escenarios no cointegrados)."
    )
    a_vol: float = Field(
        0.01, ge=0, allow_inf_nan=False, description="Volatilidad de A (no cointegrados)."
    )
    rho: float = Field(
        0.9, ge=-1, le=1, allow_inf_nan=False, description="Correlación de los incrementos (correlado no cointegrado)."
    )
    b_start_price: float = Field(100.0, gt=0, allow_inf_nan=False, description="Precio inicial de B.")
    a_start_price: float = Field(
        100.0, gt=0, allow_inf_nan=False, description="Precio inicial de A (no cointegrados)."
    )


class SyntheticOut(EducationalOut):
    synthetic: bool = Field(True, description="Siempre True: datos SINTÉTICOS.")
    label: str = Field(SYNTHETIC_LABEL, description="Etiqueta permanente: «SINTÉTICO».")
    a: SeriesMetaOut = Field(description="Serie A (con snapshot_id en la caché).")
    b: SeriesMetaOut = Field(description="Serie B (con snapshot_id en la caché).")
    scenario: SyntheticScenario = Field(description="Escenario generado.")
    scenario_label: str = Field(description="Nombre del escenario en español.")
    params: dict[str, Any] = Field(description="Parámetros efectivos (con los del escenario).")
    note: str = Field(description="Explicación en español.")


class SeriesDataOut(EducationalOut):
    meta: SeriesMetaOut = Field(description="Metadatos de la serie.")
    close: SeriesOut = Field(
        description=(
            "Precios de cierre tal como están en la caché (sin rellenar fechas; valores "
            "faltantes como null)."
        )
    )


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: SeriesRefIn = Field(description="Serie A.")
    b: SeriesRefIn = Field(description="Serie B.")
    spread_type: SpreadType = Field(
        SpreadType.LOG, description="Tipo de spread: «log» (por defecto) o «precio»."
    )
    min_train_obs: int = Field(
        MIN_TRAIN_OBS,
        ge=30,
        le=100_000,
        description="Mínimo de observaciones en formación (por defecto 500).",
    )
    train_fraction: float = Field(
        0.6, gt=0, le=1, allow_inf_nan=False, description="Fracción de formación (por defecto 0,6)."
    )


class ValidationOut(ValidationReportOut):
    """Informe de validación del par, con las series usadas y el aviso educativo."""

    disclaimer: str = Field(DISCLAIMER, description="Aviso educativo obligatorio.")
    synthetic: bool = Field(False, description="True si alguna serie es SINTÉTICA.")
    series_meta: list[SeriesMetaOut] = Field(
        default_factory=list,
        description="Metadatos de las series validadas (A y B), con su snapshot_id.",
    )


# --------------------------------------------------------------------------- endpoints


@router.post(
    "/data/load",
    response_model=LoadResponse,
    summary="Cargar series de un proveedor",
    description=(
        "Carga uno o varios símbolos para un rango de fechas. Usa la caché local si una "
        "instantánea cubre el rango (sin red); si no, descarga y guarda una instantánea "
        "inmutable. Si algunos símbolos fallan, la respuesta trae las series cargadas y un error "
        "en español por cada símbolo fallido; si fallan todos, responde con el error del primero "
        "(404 símbolo sin datos, 429 límite de peticiones, 503 sin conexión)."
    ),
    response_description=OK,
    responses=DATA_RESPONSES,
)
def load_data(req: LoadRequest, state: AppState = Depends(get_state)) -> LoadResponse:
    name = req.provider or state.registry.default_name or ""
    source = state.registry.get(name)
    always_reload = bool(getattr(source, "always_reload", False))
    loaded = []
    errors: list[SymbolErrorOut] = []
    cached: list[str] = []
    first_error: LabError | None = None
    for sym in req.symbols:
        try:
            hit = (
                not req.refresh
                and not always_reload
                and state.data.find_covering(name, sym, req.start, req.end) is not None
            )
            series = state.data.load(name, sym, req.start, req.end, refresh=req.refresh)
        except LabError as exc:
            first_error = first_error or exc
            errors.append(
                SymbolErrorOut(
                    symbol=sym,
                    status=lab_error_status(exc),
                    error=ErrorBody(code=exc.code, message=exc.message, details=json_safe(exc.details)),
                )
            )
            continue
        loaded.append(series.meta)
        if hit:
            cached.append(sym)
    if not loaded and first_error is not None:
        raise first_error
    return LoadResponse(
        series=loaded,  # type: ignore[arg-type]
        errors=errors,
        cached=cached,
        synthetic=any(m.is_synthetic for m in loaded),
    )


def _csv_provider(state: AppState) -> tuple[str, CSVProvider]:
    for name in ["csv", *state.registry.names()]:
        if name in state.registry.names():
            provider = state.registry.get(name)
            if isinstance(provider, CSVProvider):
                return name, provider
    raise ConfigError(
        "La carga de archivos CSV no está habilitada en este servidor (proveedor «csv» en "
        "providers.toml).",
        code="PROVEEDOR_DESCONOCIDO",
        details={"proveedor": "csv", "disponibles": state.registry.names()},
    )


def _blank_to_none(value: str | None) -> str | None:
    return value.strip() if value is not None and value.strip() else None


@_upload_router.post(
    "/data/csv",
    response_model=CSVUploadOut,
    summary="Cargar un archivo CSV",
    description=(
        "Sube un CSV con columnas date y close (opcional open, high, low, volume; se aceptan "
        "«fecha» y «cierre», separador coma, punto y coma o tabulador). Hay que declarar la "
        "moneda, la clase de activo y el tipo de precio. La serie se valida, se registra y queda "
        "en la caché con su snapshot_id. Si el archivo tiene la columna tipo_datos = SINTETICO o "
        "su nombre empieza por SINTETICO_, la serie se trata como SINTÉTICA aunque el formulario "
        "diga otra cosa, y la respuesta lo avisa. El tamaño máximo lo fija el servidor (413 si "
        "se supera)."
    ),
    response_description=OK,
    responses=UPLOAD_RESPONSES,
)
def upload_csv(
    file: UploadFile = File(
        ..., description="Archivo CSV (UTF-8 o Windows-1252; fechas AAAA-MM-DD)."
    ),
    symbol: str = Form(
        ..., min_length=1, max_length=MAX_SYMBOL_CHARS, description="Símbolo de la serie (p. ej. «KO»)."
    ),
    currency: str = Form(..., description="Moneda: código ISO de 3 letras (p. ej. USD, EUR, MXN)."),
    asset_class: str = Form(..., description=f"Clase de activo: {_values(AssetClass)}."),
    price_type: str = Form(
        ...,
        description=(
            f"Tipo de precio: {_values(PriceType)} («cierre_ajustado» si incluye splits y "
            "dividendos)."
        ),
    ),
    name: str = Form("", max_length=200, description="Nombre (si se omite, el símbolo)."),
    origin: str | None = Form(
        None,
        description=(
            f"Tipo de datos: {_values(DataOrigin)} (opcional). El nombre del archivo o la columna "
            "tipo_datos pueden forzar SINTETICO; nunca al revés."
        ),
    ),
    seed: str | None = Form(
        None, description="Semilla del generador (opcional; implica datos SINTÉTICOS)."
    ),
    state: AppState = Depends(get_state),
) -> CSVUploadOut:
    settings = state.settings
    limit = max_upload_bytes(settings)
    if file.size is not None and file.size > limit:
        raise too_large(settings)
    file.file.seek(0)
    content = read_limited(file.file, limit, settings)

    fields: dict[str, Any] = {
        "symbol": symbol,
        "name": name,
        "currency": currency,
        "asset_class": asset_class,
        "price_type": price_type,
    }
    if _blank_to_none(origin) is not None:
        fields["origin"] = _blank_to_none(origin)
    if _blank_to_none(seed) is not None:
        fields["seed"] = _blank_to_none(seed)
    meta = build_metadata(**fields)
    declared_synthetic = meta.is_synthetic

    provider_name, provider = _csv_provider(state)
    with _csv_lock:  # registrar y cachear como una sola operación (subidas concurrentes)
        registered = provider.register(content, meta, filename=file.filename)
        series = state.data.load(
            provider_name,
            registered.symbol,
            registered.meta.coverage_start,  # type: ignore[arg-type]
            registered.meta.coverage_end,  # type: ignore[arg-type]
        )

    synthetic = series.meta.is_synthetic
    forced = synthetic and not declared_synthetic
    notices: list[str] = []
    if forced:
        notices.append(
            "Los datos se tratan como SINTÉTICOS aunque el formulario declaraba datos históricos: "
            "nunca se presentan datos sintéticos como históricos."
        )
        notices.extend(n for n in series.meta.asset.notes if n != SYNTHETIC_NOTE)
    elif synthetic:
        notices.append(SYNTHETIC_NOTE)
    return CSVUploadOut(
        series=series.meta,  # type: ignore[arg-type]
        synthetic=synthetic,
        synthetic_forced=forced,
        notices=notices,
    )


def _synthetic_provider(state: AppState) -> tuple[str, SyntheticProvider]:
    for name in ["synthetic", *state.registry.names()]:
        if name in state.registry.names():
            provider = state.registry.get(name)
            if isinstance(provider, SyntheticProvider):
                return name, provider
    raise ConfigError(
        "El generador de datos SINTÉTICOS no está habilitado en este servidor (proveedor "
        "«synthetic» en providers.toml).",
        code="PROVEEDOR_DESCONOCIDO",
        details={"proveedor": "synthetic", "disponibles": state.registry.names()},
    )


@router.post(
    "/data/synthetic",
    response_model=SyntheticOut,
    summary="Generar un par SINTÉTICO",
    description=(
        "Genera un par de series SINTÉTICAS con semilla fija (cointegrado, cointegrado con baja "
        "correlación, correlacionado no cointegrado o independientes), lo registra y guarda ambas "
        "series en la caché con su snapshot_id (reproducible). Los datos llevan siempre la "
        "etiqueta SINTÉTICO y nunca modifican datos históricos."
    ),
    response_description=OK,
    responses=ERROR_RESPONSES,
)
def generate_synthetic(req: SyntheticRequest, state: AppState = Depends(get_state)) -> SyntheticOut:
    # Solo los campos indicados: el escenario aplica sus valores por defecto al resto.
    params = SyntheticPairParams(**req.model_dump(exclude_unset=True))
    effective = effective_params(params)
    name, provider = _synthetic_provider(state)
    sym_a, sym_b = provider.register(params)
    dates = pd.bdate_range(start=effective.start, periods=effective.n_obs)
    start, end = dates[0].date(), dates[-1].date()
    series_a = state.data.load(name, sym_a, start, end)
    series_b = state.data.load(name, sym_b, start, end)
    note = (
        f"Datos SINTÉTICOS generados con la semilla {effective.seed}: no son precios históricos. "
        "Ambas series quedan en la caché con su snapshot_id, así que el análisis se puede "
        "reproducir exactamente. Los mismos parámetros generan siempre los mismos símbolos y "
        "los mismos precios."
    )
    return SyntheticOut(
        a=series_a.meta,  # type: ignore[arg-type]
        b=series_b.meta,  # type: ignore[arg-type]
        scenario=effective.scenario,
        scenario_label=SCENARIO_LABELS[effective.scenario],
        params=effective.model_dump(mode="json"),
        note=note,
    )


@router.get(
    "/data/cache",
    response_model=list[SeriesMetaOut],
    summary="Instantáneas en la caché",
    description=(
        "Lista las instantáneas guardadas en la caché local (más recientes primero), con filtro "
        "opcional por proveedor y símbolo. Nunca descarga."
    ),
    response_description=OK,
    responses=ERROR_RESPONSES,
)
def list_cache(
    provider: str | None = Query(None, max_length=40, description="Filtrar por proveedor."),
    symbol: str | None = Query(
        None, max_length=MAX_SYMBOL_CHARS, description="Filtrar por símbolo (sin distinguir mayúsculas)."
    ),
    state: AppState = Depends(get_state),
) -> list[Any]:
    return state.cache.list(provider or None, (symbol or "").strip() or None)


@router.get(
    "/data/series",
    response_model=SeriesDataOut,
    summary="Precios de una instantánea",
    description=(
        "Devuelve los precios de cierre de una instantánea de la caché (para gráficos), con "
        "recorte opcional por fechas. Solo usa la caché: nunca descarga ni rellena fechas. Si la "
        "instantánea no existe → 404 SNAPSHOT_NO_ENCONTRADO."
    ),
    response_description=OK,
    responses=ERROR_RESPONSES,
)
def series_data(
    snapshot_id: str = Query(..., min_length=1, max_length=200, description="Instantánea."),
    start: date | None = Query(None, description="Inicio opcional (AAAA-MM-DD)."),
    end: date | None = Query(None, description="Fin opcional (AAAA-MM-DD)."),
    state: AppState = Depends(get_state),
) -> SeriesDataOut:
    series = state.data.load_snapshot(snapshot_id.strip(), start, end)
    return SeriesDataOut(
        meta=series.meta,  # type: ignore[arg-type]
        close=series_out(series.close),  # type: ignore[arg-type]
        synthetic=series.is_synthetic,
    )


@router.post(
    "/data/validate",
    response_model=ValidationOut,
    summary="Validar los datos de un par",
    description=(
        "Informe de validación previo a cualquier análisis (sección 2.3): orden y duplicados, "
        "faltantes (nunca se rellenan), precios ≤ 0, historial común mínimo, monedas distintas, "
        "saltos sospechosos y calendario (siempre intersección de fechas). Un par bloqueado no es "
        "un error HTTP: el informe lo indica con «blocked» y sus causas."
    ),
    response_description=OK,
    responses=DATA_RESPONSES,
)
def validate_data(req: ValidateRequest, state: AppState = Depends(get_state)) -> ValidationOut:
    a = load_ref(state.data, req.a, "A")
    b = load_ref(state.data, req.b, "B")
    report, _ = validate_pair(a, b, req.spread_type, req.min_train_obs, req.train_fraction)
    return ValidationOut(
        **report.model_dump(),
        series_meta=[a.meta, b.meta],  # type: ignore[list-item]
        synthetic=a.is_synthetic or b.is_synthetic,
    )


router.include_router(_upload_router)
