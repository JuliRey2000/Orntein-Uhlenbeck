"""Documentación en español del esquema OpenAPI para los modelos del núcleo.

Las respuestas de la API reutilizan modelos del núcleo (``Metrics``, ``Trade``, ``EGResult``…)
que no tienen descripciones en sus campos. En lugar de tocar el núcleo (librería pura, fase 1),
este módulo completa el esquema OpenAPI al generarlo:

1. Copia las descripciones de su versión documentada: el esquema ``X`` toma las de ``XIn`` o
   ``XOut`` si existen (``RunConfig`` ← ``RunConfigIn``, ``SeriesMeta`` ← ``SeriesMetaOut``…).
2. Añade las descripciones de ``CORE_FIELDS`` (y la ayuda de una línea de cada métrica,
   ``METRIC_HELP``), las de cada modelo (``CORE_SCHEMAS``) y las de cada enumeración
   (``ENUM_DESCRIPTIONS``).
3. Da nombres estables a los formularios multipart (``FORM_SCHEMAS``): FastAPI los llama
   «Body_<operación>», un nombre poco útil para generar los tipos TypeScript.

Solo rellena descripciones que faltan: nunca cambia tipos, nombres ni valores. ``install_openapi``
sustituye ``app.openapi`` por la versión completada (en caché, como hace FastAPI).
"""

from __future__ import annotations

import copy
from typing import Any

from fastapi import FastAPI

from mrlab.backtest.metrics import METRIC_HELP

#: Descripciones de los campos de los modelos del núcleo que aparecen en las respuestas.
CORE_FIELDS: dict[str, dict[str, str]] = {
    "Cause": {
        "code": "Código estable de la causa (p. ej. «EG_NO_RECHAZA», «OU_NO_VALIDO»).",
        "status": "Estado del par que provoca (null si solo es informativa o un descarte).",
        "message": "Explicación en español.",
    },
    "Trade": {
        "trade_id": "Número de la operación dentro del segmento.",
        "side": "«largo_spread» (largo A, corto β·B) o «corto_spread» (corto A, largo β·B).",
        "entry_signal_date": "Fecha de la señal de entrada (cierre de t).",
        "entry_date": "Fecha de ejecución de la entrada (cierre de t+1).",
        "entry_z": "z en el cierre de la señal de entrada.",
        "exit_signal_date": "Fecha de la señal de salida (null si sigue abierta).",
        "exit_date": "Fecha de ejecución de la salida (null si sigue abierta).",
        "exit_z": "z en el cierre de la señal de salida.",
        "exit_reason": "Condición de salida aplicada (según la prioridad de la sección 7.5).",
        "active_exit_conditions": "Todas las condiciones de salida activas en la señal.",
        "bars_held": "Días de negociación con la posición abierta.",
        "q_y": "Cantidad de la pierna A (positiva = larga, negativa = corta).",
        "q_x": "Cantidad de la pierna B (positiva = larga, negativa = corta).",
        "price_y_entry": "Precio de ejecución de la pierna A en la entrada.",
        "price_x_entry": "Precio de ejecución de la pierna B en la entrada.",
        "price_y_exit": "Precio de ejecución de la pierna A en la salida.",
        "price_x_exit": "Precio de ejecución de la pierna B en la salida.",
        "gross_pnl": "PnL bruto de la operación (antes de costes).",
        "commission": "Comisiones pagadas.",
        "slippage": "Coste de bid-ask / slippage.",
        "borrow": "Coste de préstamo de los cortos.",
        "financing": "Coste de financiación del apalancamiento.",
        "total_costs": "Costes totales de la operación.",
        "net_pnl": "PnL neto = bruto − costes.",
        "return_on_capital": "PnL neto / capital inicial del par.",
        "notional_traded": "Nocional operado (entrada, salida y ajustes).",
        "open_at_end": "True si la operación seguía abierta al final del segmento.",
    },
    "Event": {
        "date": "Fecha del evento.",
        "kind": (
            "Tipo de evento (p. ej. «senal_no_ejecutable», «reduccion_apalancamiento», "
            "«capital_agotado»)."
        ),
        "message": "Explicación en español.",
        "details": "Datos adicionales del evento.",
    },
    "HedgeFit": {
        "spread_type": "Tipo de spread: «log» o «precio».",
        "dependent": "Serie dependiente (la que se explica).",
        "independent": "Serie independiente.",
        "alpha": "α estimado por MCO (solo formación).",
        "beta": "β (hedge ratio) estimado por MCO (solo formación).",
        "alpha_se": "Error estándar MCO de α (orientativo: ver el aviso sobre la inferencia).",
        "beta_se": "Error estándar MCO de β (orientativo: ver el aviso sobre la inferencia).",
        "r_squared": "R² de la regresión de formación.",
        "nobs": "Observaciones usadas en la estimación.",
        "method": "Método de estimación.",
        "estimation_start": "Inicio del periodo de estimación (formación).",
        "estimation_end": "Fin del periodo de estimación (formación).",
        "units": "Unidades del spread.",
        "interpretation": "Interpretación de β en español.",
        "warnings": "Avisos en español.",
    },
    "UnitRootTest": {
        "test": "Prueba: «ADF» (H0: raíz unitaria) o «KPSS» (H0: estacionaria).",
        "series": "Serie probada: «niveles» o «diferencias».",
        "statistic": "Estadístico de la prueba.",
        "pvalue": "p-value.",
        "pvalue_bound": (
            "«exacto», o «menor_que»/«mayor_que» si el p-value de KPSS está fuera de la tabla "
            "(es una cota)."
        ),
        "lags": "Rezagos usados.",
        "nobs": "Observaciones usadas.",
        "regression": "Términos deterministas (c = constante).",
        "critical_values": "Valores críticos (1 %, 5 %, 10 %).",
        "rejects_null": "True si rechaza la hipótesis nula al nivel elegido.",
    },
    "IntegrationResult": {
        "symbol": "Símbolo de la serie.",
        "alpha": "Nivel de las pruebas.",
        "level_adf": "ADF sobre log-precios (niveles).",
        "level_kpss": "KPSS sobre log-precios (niveles).",
        "diff_adf": "ADF sobre las primeras diferencias.",
        "diff_kpss": "KPSS sobre las primeras diferencias.",
        "classification": (
            "Interpretación combinada: «compatible_I1», «posiblemente_estacionaria» o «ambiguo»."
        ),
        "explanation": "Explicación en español.",
    },
    "EGResult": {
        "direction": "Dirección: «a_sobre_b» (A dependiente) o «b_sobre_a».",
        "dependent": "Serie dependiente.",
        "independent": "Serie independiente.",
        "statistic": "Estadístico de Engle–Granger.",
        "pvalue": (
            "p-value de MacKinnon para residuos estimados (nunca el de un ADF ordinario sobre "
            "los residuos)."
        ),
        "critical_values": "Valores críticos (1 %, 5 %, 10 %).",
        "nobs": "Observaciones usadas.",
        "trend": "Términos deterministas (c = constante).",
        "lags": "Rezagos elegidos.",
        "maxlag": "Rezagos máximos considerados.",
        "autolag": "Criterio de selección de rezagos (AIC).",
        "alpha_ols": "α de la regresión de cointegración.",
        "beta_ols": "β de la regresión de cointegración.",
        "method": "Método y fuente de los p-values.",
        "warning": "Aviso (p. ej. series casi colineales: prueba no fiable).",
    },
    "EGDecision": {
        "direction": "Dirección elegida.",
        "direction_was_auto": "True si se eligió automáticamente (la de menor p-value).",
        "p_raw": "p-value original por dirección.",
        "p_adjusted": "p-value ajustado por pruebas múltiples, por dirección.",
        "evidence": "True si el p-value ajustado de la dirección elegida es menor que α.",
        "message": "Explicación en español.",
        "alpha_level": "Nivel α.",
        "fdr_method": "Corrección por pruebas múltiples («bh» o «by»).",
        "family_size": "Número de pruebas de la familia corregida.",
    },
    "EGWindow": {
        "start": "Inicio de la ventana.",
        "end": "Fin de la ventana.",
        "nobs": "Observaciones de la ventana.",
        "statistic": "Estadístico de Engle–Granger (null si la ventana no se pudo evaluar).",
        "pvalue": "p-value sin ajustar de la ventana.",
        "alpha": "α de la ventana.",
        "beta": "β de la ventana.",
        "passed": "True si la ventana rechaza la hipótesis nula (p < α).",
        "evaluable": "False si la ventana no se pudo evaluar.",
        "reason": "Por qué no se pudo evaluar (en español).",
    },
    "EGStability": {
        "direction": "Dirección evaluada.",
        "window": "Tamaño de ventana (observaciones).",
        "step": "Paso entre ventanas.",
        "alpha_level": "Nivel α por ventana.",
        "max_fail_frac": "Fracción máxima de ventanas que no rechazan.",
        "windows": "Resultado de cada ventana.",
        "frac_fail": "Fracción de ventanas evaluables que no rechazan.",
        "beta_sign_change": "True si β cambia de signo entre ventanas.",
        "unstable": "True si el resultado es inestable entre ventanas.",
        "reasons": "Motivos de la inestabilidad, en español.",
    },
    "LjungBoxResult": {
        "lag": "Rezago.",
        "statistic": "Estadístico de Ljung–Box.",
        "pvalue": "p-value.",
    },
    "ResidualDiagnostics": {
        "nlags": "Rezagos de la ACF.",
        "acf": "Autocorrelación de los residuos AR(1), rezagos 1..nlags.",
        "conf_band": "Banda de confianza aproximada de la ACF (±).",
        "ljung_box": "Pruebas de Ljung–Box.",
        "residual_mean": "Media de los residuos.",
        "residual_std": "Desviación estándar de los residuos.",
        "autocorrelation_detected": "True si Ljung–Box detecta autocorrelación.",
        "message": "Interpretación en español.",
    },
    "OUForecast": {
        "horizon": "Horizonte h (días de negociación).",
        "mean": "E[X_{t+h} | X_t].",
        "std": "Desviación estándar condicional.",
        "lower": "Límite inferior del intervalo predictivo.",
        "upper": "Límite superior del intervalo predictivo.",
        "level": "Nivel de confianza del intervalo.",
    },
    "OUWindow": {
        "start": "Inicio de la ventana.",
        "end": "Fin de la ventana.",
        "nobs": "Observaciones de la ventana.",
        "estimable": "True si 0 < b < 1 en la ventana (parámetros calculables).",
        "b": "Coeficiente AR(1) b.",
        "lam": "λ de la ventana.",
        "mu": "μ de la ventana.",
        "sigma": "σ de la ventana.",
        "sigma_eq": "σ_eq de la ventana.",
        "half_life": "Vida media de la ventana.",
        "unit_root_rejected": "True si la prueba ADF rechaza la raíz unitaria (informativa).",
        "df_stat": "Estadístico ADF de la ventana.",
        "df_pvalue": "p-value ADF de la ventana.",
        "df_lags": "Rezagos de la prueba ADF.",
    },
    "OUStability": {
        "window": "Tamaño de ventana (observaciones).",
        "step": "Paso entre ventanas.",
        "windows": "Parámetros por ventana.",
        "n_invalid": "Ventanas sin parámetros (b fuera de (0, 1) o datos insuficientes).",
        "cv_lambda": "Coeficiente de variación de λ entre ventanas.",
        "message": "Interpretación en español.",
        "unit_root_alpha": "Nivel de la prueba ADF por ventana.",
        "n_unit_root_tested": "Ventanas en las que se pudo calcular la prueba ADF.",
        "n_unit_root_rejected": "Ventanas en las que la prueba ADF rechaza la raíz unitaria.",
    },
    "SampleSplit": {
        "train_start": "Inicio de la formación.",
        "train_end": "Fin de la formación.",
        "validation_start": "Inicio de la validación.",
        "validation_end": "Fin de la validación.",
        "test_start": "Inicio de la prueba.",
        "test_end": "Fin de la prueba.",
        "n_train": "Observaciones de formación.",
        "n_validation": "Observaciones de validación.",
        "n_test": "Observaciones de prueba.",
    },
}

#: Descripción de cada modelo del núcleo que aparece en las respuestas.
CORE_SCHEMAS: dict[str, str] = {
    "Cause": "Causa del estado de un par o razón de descarte («¿Por qué se descartó?»).",
    "EGDecision": "Decisión de cointegración con el p-value ajustado por pruebas múltiples.",
    "EGResult": "Prueba de Engle–Granger en una dirección.",
    "EGStability": "Estabilidad de Engle–Granger por ventanas dentro de la formación.",
    "EGWindow": "Engle–Granger en una ventana de la formación.",
    "Event": "Evento del backtest (señal no ejecutable, reducción de apalancamiento…).",
    "HedgeFit": "Hedge ratio (α, β) estimado por MCO solo con la formación.",
    "IntegrationResult": "Orden de integración de una serie (ADF + KPSS).",
    "LjungBoxResult": "Prueba de Ljung–Box en un rezago.",
    "Metrics": "Métricas del segmento (cada una con su ayuda en metric_help).",
    "OUForecast": "Pronóstico del OU a un horizonte, con su intervalo predictivo.",
    "OUStability": "Estabilidad de los parámetros del OU por ventanas.",
    "OUWindow": "Parámetros del OU en una ventana.",
    "ResidualDiagnostics": "Diagnóstico de los residuos del AR(1) (ACF y Ljung–Box).",
    "SampleSplit": "Fechas de corte congeladas de formación, validación y prueba.",
    "Trade": "Operación del backtest (las dos piernas), con su PnL y sus costes.",
    "UnitRootTest": "Prueba de raíz unitaria (ADF) o de estacionariedad (KPSS).",
}

#: Nombres y descripciones de los cuerpos multipart (FastAPI los llama «Body_<operación>»).
FORM_SCHEMAS: dict[str, tuple[str, str]] = {
    "Body_upload_csv_api_data_csv_post": (
        "CSVUploadForm",
        "Formulario multipart: archivo CSV y metadatos de la serie.",
    ),
    "Body_replay_api_runs_replay_post": (
        "ReplayForm",
        "Formulario multipart: run_config.json y formato de los CSV.",
    ),
    "Body_replay_json_api_runs_replay_json_post": (
        "ReplayJsonForm",
        "Formulario multipart: run_config.json.",
    ),
}

#: Descripciones de las enumeraciones (sus valores ya están en español).
ENUM_DESCRIPTIONS: dict[str, str] = {
    "AssetClass": "Clase de activo.",
    "CalendarType": (
        "Calendario del par: «bolsa», «cripto_7d» o «mixto» (siempre intersección de fechas)."
    ),
    "DataOrigin": "Tipo de datos: HISTORICO o SINTETICO (nunca se presentan como históricos).",
    "Direction": "Dirección de Engle–Granger: «a_sobre_b» (A dependiente) o «b_sobre_a».",
    "ExitReason": (
        "Condición de salida. Prioridad: stop monetario > stop z > tiempo máximo > salida normal "
        "(«salida_forzada»: capital agotado o reducción imposible)."
    ),
    "PairStatus": (
        "Estado del par: «evidencia» (✅ evidencia de cointegración en la ventana evaluada), "
        "«evidencia_insuficiente», «supuestos_dudosos», «inestable» (⚠️) o "
        "«datos_insuficientes» (⛔)."
    ),
    "PriceType": "Tipo de precio.",
    "Segment": "Segmento de la muestra: formación, validación o prueba.",
    "Severity": "Severidad de una incidencia.",
    "Side": "Lado de la posición en el spread.",
    "SpreadType": "Tipo de spread: logarítmico o en precios.",
    "SyntheticScenario": "Escenario del generador SINTÉTICO.",
}


def _counterpart(name: str, schemas: dict[str, Any]) -> dict[str, Any] | None:
    for suffix in ("In", "Out"):
        other = schemas.get(name + suffix)
        if other is not None and other.get("properties"):
            return other
    return None


def _rename_refs(node: Any, renames: dict[str, str]) -> Any:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                prefix, _, name = value.rpartition("/")
                value = f"{prefix}/{renames.get(name, name)}"
            out[key] = _rename_refs(value, renames)
        return out
    if isinstance(node, list):
        return [_rename_refs(v, renames) for v in node]
    return node


def _rename_forms(spec: dict[str, Any]) -> dict[str, Any]:
    """«Body_<operación>» → nombre estable y legible (para los tipos TypeScript)."""
    schemas: dict[str, Any] = spec.get("components", {}).get("schemas", {})
    renames = {old: new for old, (new, _) in FORM_SCHEMAS.items() if old in schemas}
    if not renames:
        return spec
    spec = _rename_refs(spec, renames)
    schemas = spec["components"]["schemas"]
    for old, new in renames.items():
        schema = schemas.pop(old)
        schema["title"] = new
        schema.setdefault("description", FORM_SCHEMAS[old][1])
        schemas[new] = schema
    spec["components"]["schemas"] = dict(sorted(schemas.items()))
    return spec


def enrich_openapi(spec: dict[str, Any]) -> dict[str, Any]:
    """Devuelve una copia del esquema con las descripciones en español que faltaban."""
    spec = _rename_forms(copy.deepcopy(spec))
    schemas: dict[str, Any] = spec.get("components", {}).get("schemas", {})
    extra = {**CORE_FIELDS, "Metrics": dict(METRIC_HELP)}
    for name, schema in schemas.items():
        if not schema.get("description") and name in CORE_SCHEMAS:
            schema["description"] = CORE_SCHEMAS[name]
        if "enum" in schema and not schema.get("description") and name in ENUM_DESCRIPTIONS:
            schema["description"] = ENUM_DESCRIPTIONS[name]
        props: dict[str, Any] = schema.get("properties") or {}
        if not props:
            continue
        counterpart = _counterpart(name, schemas)
        if counterpart is not None:
            if not schema.get("description") and counterpart.get("description"):
                schema["description"] = counterpart["description"]
            for prop, other in counterpart["properties"].items():
                if prop in props and not props[prop].get("description"):
                    if other.get("description"):
                        props[prop]["description"] = other["description"]
        for prop, text in extra.get(name, {}).items():
            if prop in props and not props[prop].get("description"):
                props[prop]["description"] = text
    return spec


def install_openapi(app: FastAPI) -> None:
    """Sustituye ``app.openapi`` por la versión con las descripciones completadas."""
    generate = app.openapi  # método original de FastAPI (guarda el esquema en openapi_schema)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = enrich_openapi(generate())
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
