"""Manejo de errores: todo error sale como ``{"error": {"code", "message", "details"}}`` en español.

CONTRATO (a implementar) ``register_exception_handlers(app)``:
- ``LabError`` y subclases → HTTP según clase y código:
  ConfigError → 400; DataError → 422, salvo códigos de "no encontrado" (SNAPSHOT_NO_ENCONTRADO,
  ESTUDIO_NO_ENCONTRADO, SIMBOLO_SIN_DATOS, ...) → 404; AnalysisBlocked y ModelError → 422;
  ProviderError → 502, LIMITE_PETICIONES → 429 (con cabecera Retry-After razonable),
  ERROR_RED → 503. El cuerpo es ``exc.to_dict()``.
- ``RequestValidationError`` (pydantic/FastAPI) → 422, code "PETICION_INVALIDA", message en
  español que resume los campos con problema (ruta del campo con puntos, p. ej.
  "strategy.entry_z", y un motivo en español para los tipos de error más comunes: falta el
  campo, tipo incorrecto, valor fuera de rango, opción no permitida (listar las válidas), JSON
  mal formado); details = lista {campo, motivo, recibido} (recibido truncado). NUNCA texto de
  pydantic en inglés en ``message``.
- ``StarletteHTTPException`` → mismo formato: 404 "RUTA_NO_ENCONTRADA", 405
  "METODO_NO_PERMITIDO", 413 "ARCHIVO_DEMASIADO_GRANDE", otros con mensaje en español.
- ``Exception`` genérica → 500 "ERROR_INTERNO", mensaje en español genérico, SIN traza ni texto
  interno en la respuesta (se registra en el log del servidor).
- ``lab_error_status(exc) -> int`` público (lo usan las pruebas y el gestor de trabajos).
"""

from __future__ import annotations

from fastapi import FastAPI

from mrlab.domain.errors import LabError


def lab_error_status(exc: LabError) -> int:
    raise NotImplementedError


def register_exception_handlers(app: FastAPI) -> None:
    raise NotImplementedError
