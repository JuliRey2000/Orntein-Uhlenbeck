"""Manejo de errores: todo error sale como ``{"error": {"code", "message", "details"}}`` en español.

- ``LabError`` y subclases → HTTP según clase y código (``lab_error_status``). Las rutas de las
  carpetas locales del servidor (caché, estudios…) se sustituyen por una etiqueta
  (``redact_paths``): son información interna.
- Errores de validación de la petición → 422 ``PETICION_INVALIDA`` con un resumen en español
  (nunca el texto de pydantic en inglés).
- Errores HTTP de Starlette (ruta inexistente, método no permitido, cuerpo demasiado grande…)
  → mismo formato, en español.
- Cualquier otra excepción → 500 ``ERROR_INTERNO`` sin traza ni texto interno (se registra en el
  log del servidor).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
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


class RequestValueError(ValueError):
    """``ValueError`` cuyo mensaje (en español) se muestra tal cual al usuario.

    Los validadores de los modelos de petición de la API lo lanzan para que el resumen del 422
    muestre su texto (en lugar de un genérico «valor no válido»)."""


def json_safe(value: Any) -> Any:
    """Convierte ``details`` en algo serializable (fechas, enums, numpy…); NaN/±inf → None."""
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
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


# --------------------------------------------------------------------------- rutas del servidor

#: Etiqueta con la que se sustituye cada carpeta local en los errores que llegan al cliente.
_PATH_LABELS = (
    ("cache_dir", "[caché del servidor]"),
    ("studies_dir", "[estudios del servidor]"),
    ("uploads_dir", "[archivos subidos al servidor]"),
    ("data_dir", "[datos del servidor]"),
)


def server_path_replacements(settings: Any) -> list[tuple[str, str]]:
    """(ruta absoluta, etiqueta) de las carpetas locales del servidor, las más largas primero.

    Los mensajes del núcleo incluyen a veces la ruta de la caché (útil en la línea de comandos);
    en la API esa ruta es información interna del servidor y se sustituye por una etiqueta."""
    from mrlab.config import REPO_ROOT

    pairs: set[tuple[str, str]] = set()
    sources = [(getattr(settings, attr, None), label) for attr, label in _PATH_LABELS]
    sources.append((REPO_ROOT, "[carpeta del laboratorio]"))
    for raw, label in sources:
        if raw is None:
            continue
        path = Path(raw)
        # Solo rutas ABSOLUTAS: una relativa («data») sustituiría cualquier palabra igual en
        # el mensaje (y no revela nada del servidor).
        candidates = [path.resolve()] + ([path] if path.is_absolute() else [])
        for candidate in candidates:
            for form in (str(candidate), candidate.as_posix()):
                if len(Path(form).parts) > 1:  # nunca la raíz del sistema («/» o una unidad)
                    pairs.add((form, label))
    return sorted(pairs, key=lambda item: len(item[0]), reverse=True)


def redact_paths(value: Any, replacements: list[tuple[str, str]]) -> Any:
    """Sustituye las rutas del servidor en textos, listas y diccionarios (para el cliente)."""
    if isinstance(value, str):
        for path, label in replacements:
            if path in value:
                value = value.replace(path, label)
        return value
    if isinstance(value, dict):
        return {k: redact_paths(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_paths(v, replacements) for v in value]
    return value


def _replacements_for(request: Request) -> list[tuple[str, str]]:
    lab = getattr(request.app.state, "lab", None)
    return server_path_replacements(lab.settings) if lab is not None else []


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
    "uuid_type": "identificador no válido",
    "model_attributes_type": "debe ser un objeto JSON con campos ({…})",
    "string_pattern_mismatch": "el texto no tiene el formato esperado",
    "string_unicode": "el texto no es Unicode válido",
    "bytes_type": "debe ser un texto o un archivo",
    "int_too_big": "el número es demasiado grande",
    "multiple_of": "no es un múltiplo válido",
    "date_from_datetime_inexact": "fecha inválida: indica solo el día (AAAA-MM-DD), sin hora",
    "datetime_type": "fecha y hora inválidas (formato ISO)",
    "datetime_from_date_parsing": "fecha y hora inválidas (formato ISO)",
    "time_parsing": "hora inválida (formato HH:MM)",
    "timedelta_parsing": "duración inválida",
    "date_past": "la fecha debe estar en el pasado",
    "date_future": "la fecha debe estar en el futuro",
    "tuple_type": "debe ser una lista",
    "set_type": "debe ser una lista",
    "frozen_set_type": "debe ser una lista",
    "iterable_type": "debe ser una lista",
    "mapping_type": "debe ser un objeto",
    "none_required": "debe ser null",
    "decimal_parsing": "debe ser un número",
    "decimal_type": "debe ser un número",
    "url_parsing": "dirección (URL) no válida",
    "url_type": "dirección (URL) no válida",
    "union_tag_invalid": "tipo de objeto no reconocido",
    "union_tag_not_found": "falta el campo que indica el tipo de objeto",
    "frozen_field": "este campo no se puede modificar",
    "is_instance_of": "tipo de valor no válido",
}

#: Partes del parámetro ``loc`` que no son nombres de campo.
_LOC_SOURCES = ("body", "query", "path", "header", "form", "cookie")


BODY_LABEL = "(cuerpo de la petición)"


def _field_path(loc: tuple[Any, ...] | list[Any]) -> str:
    parts = [str(p) for p in loc if p not in _LOC_SOURCES]
    return ".".join(parts) if parts else BODY_LABEL


def _expected_es(expected: Any) -> str:
    """«'log' or 'precio'» (texto de pydantic) → «'log' o 'precio'»."""
    return str(expected).replace(" or ", " o ")


def format_number(value: Any) -> str:
    """Número en formato español del laboratorio: ``2520`` → «2.520», ``0.5`` → «0,5».

    Agrupa siempre los miles con punto (también con 4 cifras, como el resto de la interfaz) y
    usa coma decimal. Lo que no es un número se devuelve como texto."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value.is_integer() and abs(value) < 1e15:
            value = int(value)
        else:
            return repr(value).replace(".", ",")
    sign = "-" if value < 0 else ""
    return sign + f"{abs(value):,}".replace(",", ".")


def _is_body_root(err: dict[str, Any]) -> bool:
    loc = tuple(err.get("loc", ()))
    return loc in ((), ("body",))


def _reason(err: dict[str, Any]) -> str:
    etype = str(err.get("type", ""))
    ctx = err.get("ctx") or {}
    if etype == "missing" and _is_body_root(err):
        return "falta el cuerpo de la petición: envía un objeto JSON con los campos requeridos"
    if etype == "model_attributes_type" and isinstance(err.get("input"), (bytes, bytearray)):
        return (
            "el cuerpo debe ser un objeto JSON enviado con la cabecera "
            "«Content-Type: application/json»"
        )
    if etype == "date_from_datetime_inexact" and not isinstance(err.get("input"), str):
        return "fecha inválida: usa el formato AAAA-MM-DD (como texto)"
    if etype in ("greater_than", "greater_than_equal", "less_than", "less_than_equal"):
        op = {
            "greater_than": "mayor que",
            "greater_than_equal": "mayor o igual que",
            "less_than": "menor que",
            "less_than_equal": "menor o igual que",
        }[etype]
        limit = next(iter(ctx.values()), "")
        return f"debe ser {op} {format_number(limit)}"
    if etype in ("enum", "literal_error"):
        expected = ctx.get("expected")
        return (
            f"opción no permitida; valores válidos: {_expected_es(expected)}"
            if expected
            else "opción no permitida"
        )
    if etype == "json_invalid":
        where = err.get("loc", ())
        pos = where[1] if len(where) > 1 and isinstance(where[1], int) else None
        near = f" (error cerca del carácter {pos})" if pos is not None else ""
        return f"el cuerpo no es un JSON válido{near}"
    if etype == "too_short" and "min_length" in ctx:
        return f"debe tener al menos {format_number(ctx['min_length'])} elemento(s)"
    if etype == "too_long" and "max_length" in ctx:
        return f"admite como máximo {format_number(ctx['max_length'])} elemento(s)"
    if etype == "string_too_short" and "min_length" in ctx:
        return f"el texto debe tener al menos {format_number(ctx['min_length'])} carácter(es)"
    if etype == "string_too_long" and "max_length" in ctx:
        return f"el texto admite como máximo {format_number(ctx['max_length'])} carácter(es)"
    if etype in ("value_error", "assertion_error"):
        # Los validadores lanzan RequestValueError o LabError (en español), a veces envuelto en
        # un ValueError.
        original = ctx.get("error")
        if isinstance(original, LabError):
            return original.message
        args = getattr(original, "args", None) or ()
        if args and isinstance(args[0], LabError):
            return args[0].message
        if isinstance(original, RequestValueError) and args:
            return str(args[0])
        if args:
            text = str(args[0])
            if text and not text.isascii():  # probablemente ya en español
                return text
    return _PYDANTIC_REASONS.get(etype, "valor no válido")


def _received(err: dict[str, Any]) -> Any:
    if "input" not in err or err.get("type") == "missing":
        return None
    value = err["input"]
    # Ni objetos ni el cuerpo crudo: solo valores simples, acotados.
    if value is None or isinstance(value, (dict, list, tuple, bytes, bytearray)):
        return None
    text = str(value)
    return text if len(text) <= _MAX_RECEIVED_CHARS else text[: _MAX_RECEIVED_CHARS - 1] + "…"


def _error_field(err: dict[str, Any]) -> str:
    if err.get("type") == "json_invalid":
        return BODY_LABEL  # loc = ("body", posición): la posición no es un campo
    return _field_path(err.get("loc", ()))


def validation_summary(errors: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """(mensaje en español, detalles por campo) a partir de los errores de pydantic."""
    items = [
        {
            "campo": _error_field(err),
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
    400: (
        "PETICION_INVALIDA",
        "La petición no es válida: no se pudo interpretar el cuerpo. Revisa el formato (JSON o "
        "formulario multipart) y vuelve a intentarlo.",
    ),
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
    async def _lab_error(request: Request, exc: LabError) -> JSONResponse:
        status = lab_error_status(exc)
        headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if status == 429 else None
        payload = error_payload(exc.code, exc.message, _json_safe(exc.details))
        return JSONResponse(
            status_code=status,
            content=redact_paths(payload, _replacements_for(request)),
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
