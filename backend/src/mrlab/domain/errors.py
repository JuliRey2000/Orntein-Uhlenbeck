"""Excepciones del laboratorio. Todas llevan un código estable y un mensaje en español."""

from __future__ import annotations

from typing import Any


class LabError(Exception):
    """Error base. `code` es estable (para la API); `message` está en español."""

    default_code = "ERROR"

    def __init__(self, message: str, *, code: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code or self.default_code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class DataError(LabError):
    """Datos inválidos, mal formados o no disponibles."""

    default_code = "DATOS_INVALIDOS"


class ProviderError(LabError):
    """Fallo del proveedor externo (red, límite de peticiones, símbolo inexistente)."""

    default_code = "ERROR_PROVEEDOR"


class AnalysisBlocked(LabError):
    """El análisis no puede continuar (p. ej. monedas distintas, precios ≤ 0 con spread log)."""

    default_code = "ANALISIS_BLOQUEADO"


class ModelError(LabError):
    """Se pidió un cálculo que el modelo no permite (p. ej. vida media de un OU inválido)."""

    default_code = "MODELO_NO_VALIDO"


class ConfigError(LabError):
    """Configuración inconsistente."""

    default_code = "CONFIGURACION_INVALIDA"
