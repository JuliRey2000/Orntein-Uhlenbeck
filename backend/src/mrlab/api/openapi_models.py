"""Modelos de respuesta con descripciones en español para la documentación (/docs).

Envuelven modelos del núcleo (``AssetInfo``, ``SeriesMeta``, ``ValidationReport``…) SIN cambiar su
forma JSON: mismas claves, mismos tipos y mismos validadores; solo añaden descripciones en español
al esquema OpenAPI. Un endpoint puede devolver directamente el modelo del núcleo y declarar como
``response_model`` la versión de este módulo: FastAPI lo convierte al serializar.

Todos aceptan además una instancia del modelo del núcleo como valor de un campo
(``from_attributes=True``): ``LoadResponse(series=[serie.meta])`` funciona sin convertir a mano
(pydantic, por defecto, solo acepta en un campo la clase declarada o sus subclases, no la clase
madre).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mrlab import DISCLAIMER
from mrlab.data.presets import PresetPair
from mrlab.data.proxies import ProxySuggestion
from mrlab.data.validation import SeriesValidation, ValidationIssue, ValidationReport
from mrlab.domain.models import AssetClass, AssetInfo, DataOrigin, PriceType, SeriesMeta


def _values(enum: Any) -> str:
    return ", ".join(f"«{m.value}»" for m in enum)


def out_config(descriptions: dict[str, str]) -> ConfigDict:
    """Configuración de un modelo «Out»: descripciones en español y ``from_attributes``."""
    return ConfigDict(from_attributes=True, json_schema_extra=describe(descriptions))


def describe(descriptions: dict[str, str]) -> Callable[[dict[str, Any], type], None]:
    """``json_schema_extra`` que añade descripciones (en español) a las propiedades del esquema."""

    def extra(schema: dict[str, Any], _model: type) -> None:
        props = schema.get("properties", {})
        for name, text in descriptions.items():
            if name in props:
                props[name]["description"] = text

    return extra


# --------------------------------------------------------------------------- activos

ASSET_FIELDS = {
    "symbol": "Símbolo de la serie en el proveedor (p. ej. «^GSPC» en Yahoo).",
    "name": "Nombre económico (p. ej. «S&P 500»). Los datos SINTÉTICOS empiezan por «SINTÉTICO —».",
    "asset_class": f"Clase de activo: {_values(AssetClass)}.",
    "currency": (
        "Moneda (código ISO de 3 letras). Null si el proveedor no la informa: en ese caso el "
        "análisis queda bloqueado hasta conocerla."
    ),
    "provider": "Proveedor de datos (p. ej. «yahoo», «csv», «synthetic»).",
    "exchange": "Bolsa o mercado, si el proveedor lo informa.",
    "timezone": "Zona horaria del mercado, si el proveedor la informa.",
    "origin": (
        f"Tipo de datos: {_values(DataOrigin)}. Los datos SINTÉTICOS nunca se presentan como "
        "históricos."
    ),
    "executable": (
        "False para índices y futuros: no se pueden comprar ni vender directamente y la simulación "
        "exige elegir un instrumento ejecutable (proxy)."
    ),
    "suggested_proxy": (
        "Proxy ejecutable SUGERIDO para índices y futuros. Nunca se sustituye en silencio: el "
        "usuario debe confirmarlo."
    ),
    "notes": "Notas en español sobre el activo (proxy, cortos en cripto, datos sintéticos…).",
}


class AssetInfoOut(AssetInfo):
    """Identidad económica de un activo (sección 2.2)."""

    model_config = out_config(ASSET_FIELDS)


SERIES_META_FIELDS = {
    "asset": "Activo de la serie.",
    "price_type": (
        f"Tipo de precio: {_values(PriceType)} («cierre_ajustado» = ajustado por splits y "
        "dividendos; los dividendos no se suman aparte en el PnL)."
    ),
    "interval": "Frecuencia de los datos (en la v1 solo «1d», diaria).",
    "snapshot_id": (
        "Identificador de la instantánea inmutable en la caché local (direccionado por "
        "contenido). Con él se reproduce exactamente el análisis sin volver a descargar."
    ),
    "retrieved_at": (
        "Fecha y hora UTC de la descarga o de la lectura del archivo. Null en datos SINTÉTICOS "
        "(son deterministas)."
    ),
    "requested_start": "Inicio del rango pedido (AAAA-MM-DD).",
    "requested_end": "Fin del rango pedido (AAAA-MM-DD).",
    "coverage_start": "Primera fecha con datos.",
    "coverage_end": "Última fecha con datos.",
    "n_obs": "Número de observaciones (filas) de la serie.",
    "source_note": "Nota sobre la fuente (proveedor, ajuste, escenario sintético…).",
    "seed": "Semilla del generador (solo datos SINTÉTICOS).",
}


class SeriesMetaOut(SeriesMeta):
    """Metadatos de una serie descargada o cargada (sección 2.2)."""

    model_config = out_config(SERIES_META_FIELDS)

    asset: AssetInfoOut


class ProxySuggestionOut(ProxySuggestion):
    """Proxy ejecutable sugerido para un índice o futuro (nunca se sustituye en silencio)."""

    model_config = out_config(
        {
            "reference": "Símbolo de la referencia no ejecutable (p. ej. «^GSPC»).",
            "name": "Nombre de la referencia.",
            "proxy": "Instrumento ejecutable sugerido (p. ej. «SPY»).",
            "note": "Explicación del riesgo de base entre la referencia y el proxy.",
        }
    )


class ExampleOut(PresetPair):
    """Par de ejemplo precargado. No se asume que ninguno esté cointegrado."""

    model_config = out_config(
        {
            "key": "Identificador del ejemplo.",
            "provider": "Proveedor de los datos del ejemplo.",
            "symbol_a": "Símbolo de la pierna A.",
            "symbol_b": "Símbolo de la pierna B.",
            "title": "Título para mostrar.",
            "description": "Descripción pedagógica (no promete ningún resultado).",
            "synthetic": "True si el ejemplo usa datos SINTÉTICOS.",
        }
    )

    available: bool = Field(
        True, description="True si el proveedor del ejemplo está habilitado en este servidor."
    )


# --------------------------------------------------------------------------- validación


class ValidationIssueOut(ValidationIssue):
    """Incidencia detectada al validar los datos."""

    model_config = out_config(
        {
            "code": "Código estable de la incidencia (p. ej. «DUPLICADOS», «MONEDA_DISTINTA»).",
            "severity": "Severidad: «info», «advertencia» o «error».",
            "message": "Explicación en español.",
            "symbol": "Símbolo afectado (null si la incidencia es del par).",
            "dates": "Fechas afectadas (AAAA-MM-DD).",
            "details": "Datos adicionales de la incidencia.",
        }
    )


class SeriesValidationOut(SeriesValidation):
    """Resultado de la limpieza de una serie (nunca se rellenan fechas ni valores)."""

    model_config = out_config(
        {
            "symbol": "Símbolo de la serie.",
            "n_raw": "Observaciones recibidas.",
            "n_clean": "Observaciones tras la limpieza (sin duplicados ni faltantes).",
            "unsorted": "True si las fechas no estaban en orden cronológico.",
            "duplicates_removed": "Fechas duplicadas eliminadas (se conserva la última).",
            "duplicate_dates": "Fechas que estaban duplicadas.",
            "missing_values": "Precios faltantes (se eliminan; nunca se rellenan).",
            "missing_dates": "Fechas con precio faltante.",
            "non_positive": "Precios menores o iguales a cero.",
            "non_positive_dates": "Fechas con precio ≤ 0.",
            "suspicious_jumps": "Fechas con saltos sospechosos (posible error de ajuste).",
            "crypto_gaps": "Fechas que faltan en una serie cripto (calendario de 7 días).",
            "issues": "Incidencias de la serie.",
        }
    )

    issues: list[ValidationIssueOut] = []


VALIDATION_FIELDS = {
    "series": "Validación de cada serie.",
    "calendar": "Calendario del par: «bolsa», «cripto_7d» o «mixto» (siempre intersección).",
    "n_aligned": "Observaciones comunes a las dos series (intersección de fechas).",
    "n_train_estimate": "Observaciones estimadas en la muestra de formación.",
    "min_train_obs": "Mínimo de observaciones exigido en formación.",
    "only_in_a": "Fechas presentes solo en la serie A (no se rellenan).",
    "only_in_b": "Fechas presentes solo en la serie B (no se rellenan).",
    "currency_mismatch": "True si las monedas son distintas (bloquea el análisis en la v1).",
    "log_spread_allowed": "False si hay precios ≤ 0 (el spread logarítmico exige precios > 0).",
    "insufficient_data": "True si la formación no alcanza el mínimo de observaciones.",
    "blocked": "True si el análisis queda bloqueado («⛔ Datos insuficientes»).",
    "issues": "Todas las incidencias (de cada serie y del par).",
    "status_message": "Resumen en una frase, en español.",
}


class ValidationReportOut(ValidationReport):
    """Informe de validación de datos de un par (sección 2.3)."""

    model_config = out_config(VALIDATION_FIELDS)

    series: list[SeriesValidationOut]
    issues: list[ValidationIssueOut] = []


class EducationalOut(BaseModel):
    """Campos comunes a toda respuesta con resultados."""

    disclaimer: str = Field(DISCLAIMER, description="Aviso educativo obligatorio.")
    synthetic: bool = Field(False, description="True si algún dato usado es SINTÉTICO.")
