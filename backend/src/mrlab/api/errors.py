"""Manejo de errores: todo error sale como ``{"error": {"code", "message", "details"}}`` en español.

- ``LabError`` y subclases → HTTP según clase y código (``lab_error_status``).
- Errores de validación de la petición → 422 ``PETICION_INVALIDA`` con un resumen en español
  (nunca el texto de pydantic en inglés).
- Errores HTTP de Starlette (ruta inexistente, método no permitido, cuerpo demasiado grande…)
  → mismo formato, en español.
- Cualquier otra excepción → 500 ``ERROR_INTERNO`` sin traza ni texto interno (se registra en el
  log del servidor).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mrlab.domain.errors import (
    AnalysisBlocked,
    ConfigError,
    DataError,
    LabError,
    ModelError,
    ProviderError,
)

logger = logging.getLogger("mrlab.api")

#: Códigos de LabError que significan «no existe» → 404.
NOT_FOUND_CODES = frozenset(
    {
        "SNAPSHOT_NO_ENCONTRADO",
        "ESTUDIO_NO_ENCONTRADO",
        "TRABAJO_NO_ENCONTRADO",
        "SIMBOLO_SIN_DATOS",
        "RUN_CONFIG_NO_ENCONTRADO",
        "PROVEEDOR_DESCONOCIDO",
    }
)

#: Segundos sugeridos antes de reintentar cuando el proveedor limita peticiones.
RETRY_AFTER_SECONDS = 60

_MAX_RECEIVED_CHARS = 60
_MAX_FIELD_ERRORS = 8


def lab_error_status(exc: LabError) -> int:
    """Código HTTP de un error del laboratorio."""
    if exc.code in NOT_FOUND_CODES:
        return 404
    if isinstance(exc, ProviderError):
        if exc.code == "LIMITE_PETICIONES":
            return 429
        if exc.code == "ERROR_RED":
            return 503
        return 502
    if isinstance(exc, ConfigError):
        return 400
    if isinstance(exc, (DataError, AnalysisBlocked, ModelError)):
        return 422
    return 400


def error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def _json_safe(value: Any) -> Any:
    """Convierte ``details`` en algo serializable (fechas, enums, numpy…)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


# --------------------------------------------------------------------------- validación

_PYDANTIC_REASONS: dict[str, str] = {
    "missing": "falta este campo obligatorio",
    "extra_forbidden": "campo no permitido",
    "int_parsing": "debe ser un número entero",
    "int_type": "debe ser un número entero",
    "int_from_float": "debe ser un número entero (sin decimales)",
    "float_parsing": "debe ser un número",
    "float_type": "debe ser un número",
    "finite_number": "debe ser un número finito",
    "bool_parsing": "debe ser verdadero o falso (true/false)",
    "bool_type": "debe ser verdadero o falso (true/false)",
    "string_type": "debe ser un texto",
    "string_too_short": "el texto es demasiado corto",
    "string_too_long": "el texto es demasiado largo",
    "date_parsing": "fecha inválida: usa el formato AAAA-MM-DD",
    "date_from_datetime_parsing": "fecha inválida: usa el formato AAAA-MM-DD",
    "date_type": "fecha inválida: usa el formato AAAA-MM-DD",
    "datetime_parsing": "fecha y hora inválidas (formato ISO)",
    "list_type": "debe ser una lista",
    "dict_type": "debe ser un objeto",
    "model_type": "debe ser un objeto",
    "json_invalid": "el cuerpo no es un JSON válido",
    "json_type": "el cuerpo no es un JSON válido",
    "too_short": "la lista tiene demasiado pocos elementos",
    "too_long": "la lista tiene demasiados elementos",
    "value_error": "valor no válido",
    "assertion_error": "valor no válido",
    "uuid_parsing": "identificador no válido",
}


def _field_path(loc: tuple[Any, ...] | list[Any]) -> str:
    parts = [str(p) for p in loc if p not in ("body", "query", "path", "header", "form", "cookie")]
    return ".".join(parts) if parts else "(cuerpo de la petición)"


def _reason(err: dict[str, Any]) -> str:
    etype = str(err.get("type", ""))
    ctx = err.get("ctx") or {}
    if etype in ("greater_than", "greater_than_equal", "less_than", "less_than_equal"):
        op = {
            "greater_than": "mayor que",
            "greater_than_equal": "mayor o igual que",
            "less_than": "menor que",
            "less_than_equal": "menor o igual que",
        }[etype]
        limit = next(iter(ctx.values()), "")
        return f"debe ser {op} {limit}"
    if etype in ("enum", "literal_error"):
        expected = ctx.get("expected")
        return (
            f"opción no permitida; valores válidos: {expected}"
            if expected
            else "opción no permitida"
        )
    if etype == "too_short" and "min_length" in ctx:
        return f"debe tener al menos {ctx['min_length']} elemento(s)"
    if etype == "too_long" and "max_length" in ctx:
        return f"admite como máximo {ctx['max_length']} elemento(s)"
    if etype in ("value_error", "assertion_error"):
        # Los validadores del laboratorio lanzan LabError (en español) envuelto en ValueError.
        original = ctx.get("error")
        if isinstance(original, LabError):
            return original.message
        if original is not None and getattr(original, "args", None):
            text = str(original.args[0])
            if text and not text.isascii():  # probablemente ya en español
                return text
    return _PYDANTIC_REASONS.get(etype, "valor no válido")


def _received(err: dict[str, Any]) -> Any:
    if "input" not in err or err.get("type") == "missing":
        return None
    value = err["input"]
    if isinstance(value, (dict, list)):
        return None
    text = str(value)
    return text if len(text) <= _MAX_RECEIVED_CHARS else text[: _MAX_RECEIVED_CHARS - 1] + "…"


def validation_summary(errors: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """(mensaje en español, detalles por campo) a partir de los errores de pydantic."""
    items = [
        {
            "campo": _field_path(err.get("loc", ())),
            "motivo": _reason(err),
            "recibido": _received(err),
        }
        for err in errors
    ]
    shown = items[:_MAX_FIELD_ERRORS]
    parts = [f"{it['campo']}: {it['motivo']}" for it in shown]
    extra = len(items) - len(shown)
    if extra > 0:
        parts.append(f"y {extra} problema(s) más")
    message = "La petición tiene datos no válidos. " + "; ".join(parts) + "."
    return message, items


# --------------------------------------------------------------------------- HTTP

_HTTP_MESSAGES: dict[int, tuple[str, str]] = {
    400: ("PETICION_INVALIDA", "La petición no es válida."),
    401: ("NO_AUTORIZADO", "No autorizado."),
    403: ("PROHIBIDO", "No tienes permiso para esta operación."),
    404: (
        "RUTA_NO_ENCONTRADA",
        "La ruta solicitada no existe. Consulta la documentación en /docs.",
    ),
    405: ("METODO_NO_PERMITIDO", "Método HTTP no permitido para esta ruta. Consulta /docs."),
    409: ("CONFLICTO", "La operación no es posible en el estado actual del recurso."),
    413: ("ARCHIVO_DEMASIADO_GRANDE", "El archivo o el cuerpo de la petición es demasiado grande."),
    415: ("TIPO_NO_SOPORTADO", "Tipo de contenido no soportado."),
    429: (
        "DEMASIADAS_PETICIONES",
        "Demasiadas peticiones. Espera un momento y vuelve a intentarlo.",
    ),
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LabError)
    async def _lab_error(_: Request, exc: LabError) -> JSONResponse:
        status = lab_error_status(exc)
        headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if status == 429 else None
        return JSONResponse(
            status_code=status,
            content=error_payload(exc.code, exc.message, _json_safe(exc.details)),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        message, items = validation_summary(list(exc.errors()))
        return JSONResponse(
            status_code=422,
            content=error_payload("PETICION_INVALIDA", message, {"campos": _json_safe(items)}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            # Un router puede lanzar HTTPException(detail={"code", "message", "details"}).
            payload = error_payload(
                detail["code"], detail["message"], _json_safe(detail.get("details"))
            )
        else:
            code, message = _HTTP_MESSAGES.get(
                exc.status_code, ("ERROR_HTTP", f"Error HTTP {exc.status_code}.")
            )
            payload = error_payload(code, message)
        return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Error no controlado en %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "ERROR_INTERNO",
                "Ocurrió un error interno inesperado. Si se repite, revisa los registros del "
                "servidor e informa los pasos para reproducirlo.",
            ),
        )
