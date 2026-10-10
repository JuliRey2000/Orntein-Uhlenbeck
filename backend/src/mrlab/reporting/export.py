"""Exportación determinista (sección 15): misma configuración + mismos datos = mismos bytes.

CONTRATO:
- CSV estándar: separador ",", punto decimal, UTF-8 sin BOM, fin de línea LF,
  floats con ``format(x, ".12g")``, NaN/None → celda vacía. Opción
  ``excel_es=True``: separador ";" y coma decimal (para Excel en español).
- Encabezados en español. TODA fila lleva la columna ``tipo_datos``
  (HISTORICO | SINTETICO). Los archivos de datos sintéticos llevan prefijo
  ``SINTETICO_``.
- Archivos: ``operaciones.csv`` (una fila por operación y segmento),
  ``diario.csv`` (serie diaria de posiciones, PnL y capital por segmento),
  ``pruebas_estadisticas.csv`` (ADF/KPSS por serie, EG por dirección con p
  original y ajustado, estabilidad por ventana, OU), ``run_config.json``,
  ``LEEME.txt`` (convención de ejecución, aviso educativo, dividendos incluidos
  en el precio ajustado, si hay prueba contaminada).
- Sin marcas de tiempo de generación dentro de los archivos.
- ``export_zip_bytes``: ZIP con entradas en orden fijo y ``ZipInfo.date_time``
  fijo (1980, 1, 1, 0, 0, 0), compresión DEFLATED.

Precisiones de la implementación:
- ``tipo_datos`` es la PRIMERA columna de cada CSV. Con datos sintéticos TODOS
  los archivos exportados (también ``run_config.json`` y ``LEEME.txt``) llevan
  el prefijo ``SINTETICO_``, y el nombre base sugerido también.
- ``pruebas_estadisticas.csv`` está en formato largo (``seccion, elemento,
  campo, valor``) porque mezcla pruebas con columnas distintas; incluye también
  el estado y sus causas, los descartes y las métricas de cada segmento.
- Fracciones (retornos, drawdown, % aciertos, costes / bruto) como fracción
  (0,25 = 25 %); booleanos «sí»/«no»; fechas ISO; -0 se escribe 0.
- ``excel_es=True`` añade además el BOM de UTF-8 a los CSV: sin él, Excel en
  Windows no reconoce las tildes. ``LEEME.txt`` describe el formato usado.
- El ZIP fija también el sistema de origen y los permisos de cada entrada, para
  que los bytes no dependan del sistema operativo.
- «Prueba contaminada» dice solo lo que se verificó (``RunResult.test_contaminated``):
  «SÍ», «no (según el historial registrado en run_config.json…)» o «no aplica:
  la prueba no se ejecutó», con una nota sobre el alcance del historial (el
  registro persistente de estudios llega en la fase 2). ``pruebas_estadisticas.csv``
  lo repite en la sección ``prueba`` (solicitada, ejecutada, contaminada,
  configuraciones_en_historial).
- Un segmento que no se pudo simular (``RunResult.segment_errors``) figura en
  SEGMENTOS con su causa.
"""

from __future__ import annotations

import csv
import io
import math
import zipfile
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mrlab import DISCLAIMER
from mrlab.analysis.pair_pipeline import STATUS_LABELS, PairAnalysis
from mrlab.analysis.runner import RunResult
from mrlab.backtest.costs import DIVIDENDS_NOTE
from mrlab.backtest.engine import EXECUTION_CONVENTION, BacktestResult
from mrlab.data.cache import safe_symbol
from mrlab.domain.models import PRICE_TYPE_LABELS, PriceType, Segment
from mrlab.reporting.run_config import run_config_bytes
from mrlab.stats.integration import UnitRootTest

SYNTHETIC_PREFIX = "SINTETICO_"
DATA_TYPE_COLUMN = "tipo_datos"
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
UTF8_BOM = "\ufeff"

TRADES_FILE = "operaciones.csv"
DAILY_FILE = "diario.csv"
STATS_FILE = "pruebas_estadisticas.csv"
CONFIG_FILE = "run_config.json"
README_FILE = "LEEME.txt"
FILE_ORDER = (TRADES_FILE, DAILY_FILE, STATS_FILE, CONFIG_FILE, README_FILE)

SEGMENT_ORDER = (Segment.TRAIN, Segment.VALIDATION, Segment.TEST)
SEGMENT_LABELS = {
    Segment.TRAIN: "Formación",
    Segment.VALIDATION: "Validación",
    Segment.TEST: "Prueba",
}

TRADE_COLUMNS = [
    DATA_TYPE_COLUMN,
    "segmento",
    "operacion",
    "lado",
    "pierna_y",
    "pierna_x",
    "fecha_senal_entrada",
    "fecha_entrada",
    "z_entrada",
    "fecha_senal_salida",
    "fecha_salida",
    "z_salida",
    "motivo_salida",
    "condiciones_activas",
    "observaciones_mantenida",
    "cantidad_y",
    "cantidad_x",
    "precio_y_entrada",
    "precio_x_entrada",
    "precio_y_salida",
    "precio_x_salida",
    "pnl_bruto",
    "comision",
    "slippage",
    "prestamo_cortos",
    "financiacion",
    "costes_totales",
    "pnl_neto",
    "retorno_sobre_capital",
    "nocional_operado",
    "abierta_al_final",
]

#: Columna de ``BacktestResult.daily`` → encabezado exportado (mismo orden).
DAILY_HEADERS = {
    "date": "fecha",
    "z": "z",
    "state": "estado",
    "q_y": "cantidad_y",
    "q_x": "cantidad_x",
    "price_y": "precio_y",
    "price_x": "precio_x",
    "pnl_gross": "pnl_bruto",
    "cost_commission": "coste_comision",
    "cost_slippage": "coste_slippage",
    "cost_borrow": "coste_prestamo_cortos",
    "cost_financing": "coste_financiacion",
    "cost_total": "coste_total",
    "pnl_net": "pnl_neto",
    "equity": "capital",
    "gross_exposure": "exposicion_bruta",
    "net_exposure": "exposicion_neta",
    "leverage": "apalancamiento",
    "realized_cum": "realizado_acumulado",
    "unrealized": "no_realizado",
    "trade_id": "operacion",
    "event": "evento",
}

STATS_COLUMNS = [DATA_TYPE_COLUMN, "seccion", "elemento", "campo", "valor"]

#: «Prueba contaminada: …» según ``RunResult.test_contaminated`` (True / False / None).
TEST_CONTAMINATED_TEXT = {
    True: (
        "SÍ — según el historial registrado en run_config.json, la prueba ya se había ejecutado "
        "con otra configuración (p. ej. umbrales cambiados después de verla): sus resultados ya no "
        "son una evaluación fuera de muestra."
    ),
    False: (
        "no (según el historial registrado en run_config.json, la prueba solo se ha ejecutado "
        "con esta configuración)."
    ),
    None: "no aplica: la prueba no se ejecutó.",
}
TEST_HISTORY_NOTE = (
    "  Alcance: el historial de ejecuciones de la prueba viaja dentro de run_config.json; si se "
    "borra o se edita a mano, esta marca no puede detectarlo. El registro persistente de estudios "
    "llega en la fase 2."
)
#: Valor de ``contaminada`` en ``pruebas_estadisticas.csv``.
_CONTAMINATED_CELL = {True: "sí", False: "no", None: "no aplica"}

#: Nombres en español de los campos de los modelos exportados en ``pruebas_estadisticas.csv``.
FIELD_NAMES_ES = {
    # Hedge ratio
    "spread_type": "tipo_spread",
    "dependent": "dependiente",
    "independent": "independiente",
    "alpha": "alfa",
    "alpha_se": "ee_alfa",
    "beta_se": "ee_beta",
    "r_squared": "r2",
    "method": "metodo",
    "estimation_start": "inicio_estimacion",
    "estimation_end": "fin_estimacion",
    "units": "unidades",
    "interpretation": "interpretacion",
    "warnings": "avisos",
    # OU, pronósticos y ventanas
    "status": "estado",
    "valid": "valido",
    "a_se": "ee_a",
    "b_se": "ee_b",
    "lam": "lambda",
    "half_life": "vida_media",
    "mu_se": "ee_mu",
    "time_unit": "unidad_tiempo",
    "message": "mensaje",
    "df_stat": "estadistico_dickey_fuller",
    "df_crit": "valor_critico_dickey_fuller",
    "df_alpha": "nivel_dickey_fuller",
    "df_estimated_residual": "valores_criticos_residuo_estimado",
    "near_unit_root": "cerca_de_raiz_unitaria",
    "mean": "media",
    "std": "desv",
    "lower": "inferior",
    "upper": "superior",
    "level": "nivel",
    # Métricas
    "initial_capital": "capital_inicial",
    "final_equity": "capital_final",
    "net_pnl": "pnl_neto",
    "gross_pnl": "pnl_bruto",
    "total_return": "retorno_total",
    "annualized_return": "retorno_anualizado",
    "max_drawdown": "maximo_drawdown",
    "max_drawdown_duration": "duracion_maximo_drawdown",
    "max_drawdown_start": "inicio_maximo_drawdown",
    "max_drawdown_end": "fin_maximo_drawdown",
    "max_drawdown_recovered": "maximo_drawdown_recuperado",
    "sharpe_convention": "convencion_sharpe",
    "periods_per_year": "periodos_por_anio",
    "n_observations": "observaciones",
    "n_trades": "operaciones_cerradas",
    "n_open_at_end": "operaciones_abiertas_al_final",
    "win_rate": "fraccion_aciertos",
    "avg_win": "ganancia_media",
    "avg_loss": "perdida_media",
    "profit_factor": "factor_beneficio",
    "avg_holding_bars": "duracion_media_observaciones",
    "turnover": "rotacion",
    "turnover_annual": "rotacion_anual",
    "avg_gross_exposure": "exposicion_bruta_media",
    "max_leverage_used": "apalancamiento_maximo",
    "total_costs": "costes_totales",
    "costs_pct_of_gross": "costes_sobre_pnl_bruto",
    "realized_pnl": "pnl_realizado",
    "unrealized_pnl": "pnl_no_realizado",
}


# --------------------------------------------------------------------------- formato


def _cell(value: Any, excel_es: bool) -> str:
    """Texto de una celda: determinista y sin depender de la configuración regional."""
    if value is None:
        return ""
    if isinstance(value, bool | np.bool_):
        return "sí" if value else "no"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        x = float(value)
        if not math.isfinite(x):
            return ""
        text = format(0.0 if x == 0 else x, ".12g")
        return text.replace(".", ",") if excel_es else text
    if isinstance(value, pd.Timestamp | datetime):
        ts = pd.Timestamp(value)
        return ts.date().isoformat() if ts == ts.normalize() else ts.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, list | tuple):
        return " | ".join(_cell(v, excel_es) for v in value)
    return str(value)


def _csv(header: Sequence[str], rows: Iterable[Sequence[Any]], excel_es: bool) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";" if excel_es else ",", lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_cell(v, excel_es) for v in row])
    text = buf.getvalue()
    return UTF8_BOM + text if excel_es else text


def data_type(result: RunResult) -> str:
    """HISTORICO o SINTETICO (valor de la columna ``tipo_datos``)."""
    return result.data_origin.value


def export_basename(result: RunResult) -> str:
    """Nombre base sugerido (carpeta o ZIP): ``[SINTETICO_]<A>_<B>``."""
    prefix = SYNTHETIC_PREFIX if result.is_synthetic else ""
    return (
        f"{prefix}{safe_symbol(result.analysis.symbol_a)}_{safe_symbol(result.analysis.symbol_b)}"
    )


def file_name(result: RunResult, base: str) -> str:
    return f"{SYNTHETIC_PREFIX}{base}" if result.is_synthetic else base


def _backtests(result: RunResult) -> list[tuple[Segment, BacktestResult]]:
    return [(s, result.backtests[s]) for s in SEGMENT_ORDER if s in result.backtests]


# --------------------------------------------------------------------------- CSV


def trades_csv(result: RunResult, excel_es: bool = False) -> str:
    kind = data_type(result)
    rows = []
    for segment, bt in _backtests(result):
        for t in bt.trades:
            rows.append(
                [
                    kind,
                    segment,
                    t.trade_id,
                    t.side,
                    bt.leg_y.symbol,
                    bt.leg_x.symbol,
                    t.entry_signal_date,
                    t.entry_date,
                    t.entry_z,
                    t.exit_signal_date,
                    t.exit_date,
                    t.exit_z,
                    t.exit_reason,
                    "|".join(r.value for r in t.active_exit_conditions),
                    t.bars_held,
                    t.q_y,
                    t.q_x,
                    t.price_y_entry,
                    t.price_x_entry,
                    t.price_y_exit,
                    t.price_x_exit,
                    t.gross_pnl,
                    t.commission,
                    t.slippage,
                    t.borrow,
                    t.financing,
                    t.total_costs,
                    t.net_pnl,
                    t.return_on_capital,
                    t.notional_traded,
                    t.open_at_end,
                ]
            )
    return _csv(TRADE_COLUMNS, rows, excel_es)


def daily_csv(result: RunResult, excel_es: bool = False) -> str:
    kind = data_type(result)
    columns = list(DAILY_HEADERS)
    header = [DATA_TYPE_COLUMN, "segmento", *DAILY_HEADERS.values()]

    def rows() -> Iterable[list[Any]]:
        for segment, bt in _backtests(result):
            for values in bt.daily[columns].itertuples(index=False, name=None):
                yield [kind, segment, *values]

    return _csv(header, rows(), excel_es)


def _unit_root_rows(prefix: str, test: UnitRootTest) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = [
        (f"{prefix}_estadistico", test.statistic),
        (f"{prefix}_p_valor", test.pvalue),
        (f"{prefix}_cota_p_valor", test.pvalue_bound),
        (f"{prefix}_rezagos", test.lags),
        (f"{prefix}_nobs", test.nobs),
        (f"{prefix}_rechaza_h0", test.rejects_null),
    ]
    rows += [(f"{prefix}_valor_critico_{k}", v) for k, v in test.critical_values.items()]
    return rows


def _model_rows(model: Any, exclude: Iterable[str] = ()) -> list[tuple[str, Any]]:
    skip = set(exclude)
    return [(FIELD_NAMES_ES.get(k, k), v) for k, v in model.model_dump().items() if k not in skip]


def _stats_rows(result: RunResult) -> list[tuple[str, str, str, Any]]:
    an: PairAnalysis = result.analysis
    rows: list[tuple[str, str, str, Any]] = []

    def add(section: str, element: str, pairs: Iterable[tuple[str, Any]]) -> None:
        rows.extend((section, element, k, v) for k, v in pairs)

    rep = an.validation
    add(
        "validacion",
        "par",
        [
            ("resumen", rep.status_message),
            ("calendario", rep.calendar),
            ("observaciones_comunes", rep.n_aligned),
            ("observaciones_formacion_estimadas", rep.n_train_estimate),
            ("minimo_formacion", rep.min_train_obs),
            ("solo_en_a", rep.only_in_a),
            ("solo_en_b", rep.only_in_b),
            ("bloqueado", rep.blocked),
        ],
    )
    for issue in rep.issues:
        add("validacion", issue.symbol or "par", [(issue.code, issue.message)])

    add(
        "estado",
        "par",
        [
            ("estado", an.status),
            ("etiqueta", STATUS_LABELS[an.status]),
            ("estrategia_permitida", an.strategy_allowed),
            ("solo_ilustrativo", an.illustrative_only),
        ],
    )
    add("estado", "causa", [(c.code, c.message) for c in an.causes])
    add("descarte", "razon", [(c.code, c.message) for c in an.discard_reasons])
    add(
        "prueba",
        "par",
        [
            ("solicitada", result.test_included),
            ("ejecutada", Segment.TEST in result.backtests),
            ("contaminada", _CONTAMINATED_CELL[_contaminated(result)]),
            ("configuraciones_en_historial", len(result.config.test_history)),
        ],
    )

    for symbol, res in an.integration.items():
        pairs: list[tuple[str, Any]] = [("clasificacion", res.classification)]
        for test in (res.level_adf, res.level_kpss, res.diff_adf, res.diff_kpss):
            pairs += _unit_root_rows(f"{test.test.lower()}_{test.series}", test)
        pairs.append(("explicacion", res.explanation))
        add("integracion", symbol, pairs)

    for direction, eg in an.eg.items():
        pairs = [
            ("direccion", direction),
            ("dependiente", eg.dependent),
            ("independiente", eg.independent),
            ("estadistico", eg.statistic),
            ("p_valor", eg.pvalue),
            ("p_valor_ajustado", an.eg_p_adjusted.get(direction)),
            *[(f"valor_critico_{k}", v) for k, v in eg.critical_values.items()],
            ("nobs", eg.nobs),
            ("tendencia", eg.trend),
            ("rezagos", eg.lags),
            ("rezagos_maximos", eg.maxlag),
            ("criterio_rezagos", eg.autolag),
            ("alfa_mco", eg.alpha_ols),
            ("beta_mco", eg.beta_ols),
            ("elegida", direction == an.chosen_direction),
            ("metodo", eg.method),
            ("aviso", eg.warning),
        ]
        add("engle_granger", f"{eg.dependent} sobre {eg.independent}", pairs)
    if an.eg_decision is not None:
        dec = an.eg_decision
        add(
            "engle_granger",
            "decision",
            [
                ("direccion", dec.direction),
                ("direccion_automatica", dec.direction_was_auto),
                ("metodo_fdr", dec.fdr_method),
                ("pruebas_en_familia", dec.family_size),
                ("nivel", dec.alpha_level),
                ("evidencia", dec.evidence),
                ("mensaje", dec.message),
            ],
        )

    if an.stability is not None:
        st = an.stability
        add(
            "estabilidad_eg",
            "resumen",
            [
                ("direccion", st.direction),
                ("ventana", st.window),
                ("paso", st.step),
                ("nivel", st.alpha_level),
                ("fraccion_maxima_fallida", st.max_fail_frac),
                ("fraccion_fallida", st.frac_fail),
                ("cambio_signo_beta", st.beta_sign_change),
                ("inestable", st.unstable),
                *[(f"razon_{i}", r) for i, r in enumerate(st.reasons, start=1)],
            ],
        )
        for i, w in enumerate(st.windows, start=1):
            add(
                "estabilidad_eg",
                f"ventana {i} ({w.start.isoformat()} a {w.end.isoformat()})",
                [
                    ("nobs", w.nobs),
                    ("estadistico", w.statistic),
                    ("p_valor", w.pvalue),
                    ("alfa", w.alpha),
                    ("beta", w.beta),
                    ("pasa", w.passed),
                    ("evaluable", w.evaluable),
                    ("motivo", w.reason),
                ],
            )

    if an.hedge is not None:
        add("hedge_ratio", "mco", _model_rows(an.hedge))

    if an.ou is not None:
        add("ou", "formacion", _model_rows(an.ou))
    for fc in an.forecasts:
        add("pronostico_ou", f"h={fc.horizon}", _model_rows(fc, exclude=("horizon",)))
    if an.residuals is not None:
        res_diag = an.residuals
        pairs = [
            ("rezagos_acf", res_diag.nlags),
            ("banda_confianza_acf", res_diag.conf_band),
            ("media_residuos", res_diag.residual_mean),
            ("desv_residuos", res_diag.residual_std),
            ("autocorrelacion_detectada", res_diag.autocorrelation_detected),
            ("mensaje", res_diag.message),
        ]
        pairs += [(f"acf_{k}", v) for k, v in enumerate(res_diag.acf, start=1)]
        for lb in res_diag.ljung_box:
            pairs += [
                (f"ljung_box_{lb.lag}_estadistico", lb.statistic),
                (f"ljung_box_{lb.lag}_p_valor", lb.pvalue),
            ]
        add("residuos_ar1", "formacion", pairs)
    if an.ou_stability is not None:
        ous = an.ou_stability
        add(
            "estabilidad_ou",
            "resumen",
            [
                ("ventana", ous.window),
                ("paso", ous.step),
                ("ventanas_no_validas", ous.n_invalid),
                ("cv_lambda", ous.cv_lambda),
                ("mensaje", ous.message),
            ],
        )
        for i, w in enumerate(ous.windows, start=1):
            add(
                "estabilidad_ou",
                f"ventana {i} ({w.start.isoformat()} a {w.end.isoformat()})",
                _model_rows(w, exclude=("start", "end")),
            )
    add("correlacion", "formacion", [("pearson_rend_log", an.correlation_train)])

    for segment, bt in _backtests(result):
        pairs = _model_rows(bt.metrics, exclude=("costs_by_type", "warnings"))
        pairs += [(f"costes_{k}", v) for k, v in bt.metrics.costs_by_type.items()]
        pairs += [("tiempo_maximo_observaciones", bt.max_holding_bars)]
        pairs += [(f"aviso_{i}", w) for i, w in enumerate(bt.metrics.warnings, start=1)]
        add("metricas", segment.value, pairs)
    return rows


def stats_csv(result: RunResult, excel_es: bool = False) -> str:
    kind = data_type(result)
    return _csv(STATS_COLUMNS, ([kind, *row] for row in _stats_rows(result)), excel_es)


# --------------------------------------------------------------------------- LEEME


def _series_lines(result: RunResult) -> list[str]:
    lines: list[str] = []
    roles = {
        "a": "Serie A",
        "b": "Serie B",
        "exec_a": "Ejecutable de A",
        "exec_b": "Ejecutable de B",
    }
    for role, label in roles.items():
        series = result.series.get(role)
        if series is None:
            continue
        m, asset = series.meta, series.meta.asset
        retrieved = (
            m.retrieved_at.isoformat()
            if m.retrieved_at
            else "no aplica (datos generados o sin fecha)"
        )
        lines += [
            f"- {label}: {asset.symbol} — {asset.name}",
            f"    Proveedor: {asset.provider}; moneda: {asset.currency or 'desconocida'}; "
            f"clase de activo: {asset.asset_class.value}; tipo de datos: {asset.origin.value}",
            f"    Tipo de precio: {PRICE_TYPE_LABELS[m.price_type]}",
            f"    Cobertura usada: {m.coverage_start} a {m.coverage_end} ({m.n_obs} observaciones)",
            f"    Instantánea de la caché: {m.snapshot_id or 'sin instantánea'}",
            f"    Fecha de descarga o lectura (UTC): {retrieved}",
        ]
    return lines


def _segment_lines(result: RunResult) -> list[str]:
    split = result.analysis.split
    if split is None:
        return ["- No se dividió la muestra: el análisis se detuvo antes (ver estado y causas)."]
    counts = {
        Segment.TRAIN: split.n_train,
        Segment.VALIDATION: split.n_validation,
        Segment.TEST: split.n_test,
    }
    lines = []
    for segment in SEGMENT_ORDER:
        start, end = split.bounds(segment)
        span = f"{start} a {end} ({counts[segment]} observaciones)" if start else "vacío"
        if segment in result.backtests:
            note = "backtest incluido"
        elif segment in result.segment_errors:
            reason = result.segment_errors[segment].rstrip(". ")
            note = f"sin backtest; no se pudo simular: {_lower_first(reason)}"
        elif not result.analysis.strategy_allowed:
            note = "sin backtest (estrategia bloqueada)"
        elif segment == Segment.TEST and not result.test_included:
            note = "backtest NO ejecutado en esta exportación (la prueba se ejecuta una sola vez)"
        else:
            note = "sin backtest (segmento vacío)"
        if segment == Segment.TRAIN and segment in result.backtests:
            note += "; dentro de muestra, solo ilustrativo"
        lines.append(f"- {SEGMENT_LABELS[segment]}: {span} — {note}.")
    return lines


def _contaminated(result: RunResult) -> bool | None:
    flag = result.test_contaminated
    return None if flag is None else bool(flag)


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _price_note(result: RunResult) -> str:
    types = {s.meta.price_type for s in result.series.values()}
    notes = []
    if PriceType.ADJUSTED_CLOSE in types:
        notes.append(DIVIDENDS_NOTE)
    if PriceType.CLOSE in types:
        notes.append(
            "Hay series con precio de cierre SIN ajustar: el PnL no incluye dividendos y un split "
            "aparecería como un salto de precio."
        )
    if PriceType.SYNTHETIC in types:
        notes.append("Precios SINTÉTICOS: no hay dividendos ni splits.")
    return " ".join(notes)


def readme_txt(result: RunResult, excel_es: bool = False) -> str:
    an = result.analysis
    cfg = result.config
    synthetic = result.is_synthetic
    names = {base: file_name(result, base) for base in FILE_ORDER}
    lines: list[str] = [
        "LABORATORIO DE REVERSIÓN A LA MEDIA — EXPORTACIÓN DE UN ANÁLISIS DE PAR",
        "=" * 72,
        "",
    ]
    if synthetic:
        lines += [
            "*** DATOS SINTÉTICOS: generados por el laboratorio. NO son precios históricos. ***",
            "",
        ]
    lines += [
        f"AVISO: {DISCLAIMER}",
        "",
        f"Par analizado: {an.symbol_a} / {an.symbol_b}",
        f"Tipo de datos: {data_type(result)}",
        f"Estado del par: {STATUS_LABELS[an.status]}",
    ]
    if an.causes:
        lines.append("Causas del estado:")
        lines += [f"  - {c.message}" for c in an.causes]
    if an.discard_reasons:
        lines.append("¿Por qué se descartaría este par?")
        lines += [f"  - {c.message}" for c in an.discard_reasons]
    lines += [
        f"Hash de la configuración: {cfg.config_hash}",
        f"Versión del código: {cfg.code_version}",
        "",
        "SERIES",
        *_series_lines(result),
        "",
        "SEGMENTOS (fechas de corte congeladas en run_config.json)",
        *_segment_lines(result),
        f"Prueba contaminada: {TEST_CONTAMINATED_TEXT[_contaminated(result)]}",
        TEST_HISTORY_NOTE,
        "",
    ]
    bt = next(iter(result.backtests.values()), None)
    if bt is not None:
        lines += [
            "PIERNAS DEL BACKTEST",
            f"- y (variable dependiente del spread): {bt.leg_y.symbol}",
            f"- x: {bt.leg_x.symbol}",
            f"- β = {_cell(bt.beta, False)}; spread {bt.spread_type.value}. Las columnas "
            "cantidad_y/cantidad_x y precio_y/precio_x del diario se refieren a estas piernas.",
            "",
        ]
    lines += [
        "CONVENCIÓN DE EJECUCIÓN",
        EXECUTION_CONVENTION,
        "",
        "PRECIOS Y DIVIDENDOS",
        _price_note(result),
        "",
        "ARCHIVOS",
        f"- {names[TRADES_FILE]}: una fila por operación y segmento (entradas, salidas, motivo, "
        "costes y PnL).",
        f"- {names[DAILY_FILE]}: serie diaria por segmento: z, estado, cantidades, precios, PnL "
        "bruto, costes por tipo, PnL neto, capital, exposición y apalancamiento.",
        f"- {names[STATS_FILE]}: pruebas estadísticas en formato largo (sección, elemento, campo, "
        "valor): validación, estado y causas, ADF/KPSS, Engle–Granger con p original y "
        "ajustado, estabilidad por ventanas, hedge ratio, OU, pronósticos y métricas.",
        f"- {names[CONFIG_FILE]}: configuración completa con su hash. Cargarlo reproduce el "
        "análisis con las mismas instantáneas de la caché local (sin volver a descargar).",
        f"- {names[README_FILE]}: este archivo.",
        "",
        "FORMATO",
        (
            "- CSV para Excel en español: separador «;», coma decimal, UTF-8 con BOM, fin de "
            "línea LF."
            if excel_es
            else "- CSV: separador «,», punto decimal, UTF-8 sin BOM, fin de línea LF."
        ),
        "- Números con hasta 12 cifras significativas; una celda vacía significa «sin valor».",
        f"- Cada fila lleva la columna {DATA_TYPE_COLUMN} (HISTORICO o SINTETICO).",
        "- Retornos, drawdown, % de aciertos y costes / PnL bruto se expresan como fracción "
        "(0.25 = 25 %).",
        "- Sin marcas de tiempo de generación: la misma configuración con los mismos datos "
        "produce exactamente los mismos archivos.",
    ]
    warnings = list(an.warnings)
    for _, b in _backtests(result):
        warnings += b.warnings + b.metrics.warnings
    warnings = list(dict.fromkeys(warnings))
    if warnings:
        lines += ["", "AVISOS", *[f"- {w}" for w in warnings]]
    lines += ["", f"AVISO: {DISCLAIMER}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- archivos


def export_files(result: RunResult, excel_es: bool = False) -> dict[str, bytes]:
    """{nombre_archivo: contenido} en orden fijo."""
    contents = {
        TRADES_FILE: trades_csv(result, excel_es).encode("utf-8"),
        DAILY_FILE: daily_csv(result, excel_es).encode("utf-8"),
        STATS_FILE: stats_csv(result, excel_es).encode("utf-8"),
        CONFIG_FILE: run_config_bytes(result.config),
        README_FILE: readme_txt(result, excel_es).encode("utf-8"),
    }
    return {file_name(result, base): contents[base] for base in FILE_ORDER}


def export_run(result: RunResult, out_dir: Path | str, excel_es: bool = False) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in export_files(result, excel_es).items():
        path = out / name
        path.write_bytes(content)
        paths.append(path)
    return paths


def export_zip_bytes(result: RunResult, excel_es: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in export_files(result, excel_es).items():
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix, en cualquier sistema
            info.external_attr = 0o644 << 16
            zf.writestr(info, content)
    return buf.getvalue()
