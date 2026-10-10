"""Demo reproducible: análisis completo de un par desde CSV (fase 1).

Sin argumentos analiza el par SINTÉTICO de ``examples/data/`` (no necesita
internet). Con ``--csv-a`` y ``--csv-b`` analiza tus propios CSV (por ejemplo, KO
y PEP descargados con ``make fetch-example``). Imprime un informe en español y
exporta los resultados (CSV + run_config.json + LEEME.txt) a
``examples/output/<par>/``. La muestra de PRUEBA no se ejecuta salvo que se pida
con ``--incluir-prueba``: se ejecuta una sola vez, con la configuración congelada;
si después se vuelve a ejecutar con otra configuración, queda marcada como
contaminada.

Índices y futuros no se pueden comprar ni vender directamente: para simular la
estrategia hay que indicar el CSV del instrumento ejecutable (proxy, p. ej. un
ETF) con ``--exec-a`` / ``--exec-b``. Sin proxy, el análisis estadístico se
muestra completo, pero la simulación queda bloqueada (``PROXY_REQUERIDO``).

Metadatos de cada CSV: los de ``<archivo>.meta.json`` si existe; si no, los de la
línea de comandos. Un CSV con la columna ``tipo_datos = SINTETICO`` o cuyo nombre
empieza por ``SINTETICO_`` se trata siempre como datos SINTÉTICOS.

Ejemplos (desde la raíz del repositorio):
    python examples/demo_ko_pep.py
    python examples/demo_ko_pep.py --csv-a examples/data/real/KO.csv --csv-b examples/data/real/PEP.csv
    python examples/demo_ko_pep.py --csv-a a.csv --csv-b b.csv --currency USD \\
        --asset-class accion --price-type cierre_ajustado
    python examples/demo_ko_pep.py --csv-a SPX.csv --csv-b NDX.csv --currency USD \\
        --asset-class indice --price-type cierre --exec-a SPY.csv --exec-b QQQ.csv \\
        --asset-class-exec-a etf --asset-class-exec-b etf
    python examples/demo_ko_pep.py --reproducir examples/output/<par>/run_config.json

Errores: todos los mensajes son en español, sin trazas de Python; el programa
termina con código 2 (argumentos no válidos: también 2).
"""

from __future__ import annotations

import argparse
import shlex
import sys
import tempfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR / "data"
DEFAULT_OUTPUT = EXAMPLES_DIR / "output"
DEFAULT_SYMBOLS = ("SINT-COI-A", "SINT-COI-B")

try:
    import mrlab  # noqa: F401
except ImportError:  # permite ejecutar la demo sin instalar el paquete
    sys.path.insert(0, str(EXAMPLES_DIR.parent / "backend" / "src"))

from mrlab import DISCLAIMER  # noqa: E402
from mrlab.analysis.pair_pipeline import (  # noqa: E402
    ILLUSTRATIVE_BANNER,
    STATUS_LABELS,
    STRATEGY_BLOCKING_CODES,
    PairAnalysis,
)
from mrlab.analysis.runner import RunResult, run_pair  # noqa: E402
from mrlab.backtest.engine import EXECUTION_CONVENTION, BacktestResult  # noqa: E402
from mrlab.config import get_settings  # noqa: E402
from mrlab.data.cache import SnapshotCache  # noqa: E402
from mrlab.data.providers.csv_provider import (  # noqa: E402
    META_SUFFIX,
    SYNTHETIC_FILE_PREFIX,
    CSVMetadata,
    CSVProvider,
    build_metadata,
    origin_from_filename,
    parse_metadata,
)
from mrlab.data.registry import ProviderRegistry  # noqa: E402
from mrlab.data.service import DataService  # noqa: E402
from mrlab.domain.errors import ConfigError, DataError, LabError  # noqa: E402
from mrlab.domain.models import (  # noqa: E402
    PRICE_TYPE_LABELS,
    AssetClass,
    DataOrigin,
    PriceType,
    Segment,
)
from mrlab.model.ou import BANDS_VS_INTERVALS_NOTE, OUFit  # noqa: E402
from mrlab.reporting.export import (  # noqa: E402
    TEST_CONTAMINATED_TEXT,
    TEST_HISTORY_NOTE,
    export_basename,
    export_run,
)
from mrlab.reporting.run_config import (  # noqa: E402
    PairSpec,
    RunConfig,
    SeriesRef,
    load_run_config,
)
from mrlab.stats.integration import INTEGRATION_LABELS, format_pvalue  # noqa: E402
from pydantic import ValidationError  # noqa: E402

WIDE = "=" * 78
SEGMENT_TITLES = {
    Segment.TRAIN: "Formación (DENTRO de muestra: solo ilustrativo)",
    Segment.VALIDATION: "Validación (fuera de muestra)",
    Segment.TEST: "Prueba (fuera de muestra, configuración congelada)",
}
CONTAMINATED_TEST_TITLE = (
    "Prueba (CONTAMINADA: ya se había ejecutado con otra configuración; NO es una evaluación "
    "fuera de muestra)"
)
WIDE_RANGE = (date(1900, 1, 1), date(2100, 12, 31))

#: Roles de las series: A y B se analizan; exec_a y exec_b son los proxies ejecutables.
ROLES = ("a", "b", "exec_a", "exec_b")
ROLE_FLAGS = {"a": "--csv-a", "b": "--csv-b", "exec_a": "--exec-a", "exec_b": "--exec-b"}
ROLE_LABELS = {
    "a": "Serie A",
    "b": "Serie B",
    "exec_a": "Instrumento ejecutable de A",
    "exec_b": "Instrumento ejecutable de B",
}
ROLE_SUFFIX = {"a": "a", "b": "b", "exec_a": "exec-a", "exec_b": "exec-b"}


# --------------------------------------------------------------------------- formato


def num(x: float | None, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles; «—» si no hay valor."""
    if x is None:
        return "—"
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x: float | None, decimals: int = 2) -> str:
    return "—" if x is None else f"{num(100 * x, decimals)} %"


def p_value(p: float | None) -> str:
    if p is None:
        return "—"
    return "< 0,0001" if p < 1e-4 else num(p)


def title(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def bullet(lines: Iterable[str], indent: str = "  ") -> None:
    for line in lines:
        print(f"{indent}- {line}")


def contamination_text(result: RunResult) -> str:
    """Texto veraz de «Prueba contaminada» (el mismo que el LEEME de la exportación)."""
    flag = result.test_contaminated
    return TEST_CONTAMINATED_TEXT[None if flag is None else bool(flag)]


# --------------------------------------------------------------------------- datos


def _option(args: argparse.Namespace, field: str, role: str) -> str | None:
    return getattr(args, f"{field}_{role}", None)


def _default_symbol(path: Path) -> str:
    """Nombre del archivo en mayúsculas, sin el prefijo ``SINTETICO_``."""
    stem = path.stem
    if stem.upper().startswith(SYNTHETIC_FILE_PREFIX):
        stem = stem[len(SYNTHETIC_FILE_PREFIX) :] or stem
    return stem.upper()


def _meta_for(path: Path, args: argparse.Namespace, role: str) -> CSVMetadata:
    """Metadatos del CSV: su archivo ``.meta.json`` si existe; si no, los de la línea de comandos.

    A y B toman las opciones generales (--currency, --asset-class, --price-type) si no
    tienen las propias. Los instrumentos ejecutables heredan la moneda y el tipo de precio,
    pero su clase de activo debe declararse aparte (suele diferir: índice frente a ETF).
    """
    sidecar = path.with_name(path.stem + META_SUFFIX)
    if sidecar.is_file():
        return parse_metadata(sidecar.read_bytes(), source=sidecar.name)
    executable = role.startswith("exec")
    pick = {
        "currency": _option(args, "currency", role) or args.currency,
        "asset_class": _option(args, "asset_class", role)
        or (None if executable else args.asset_class),
        "price_type": _option(args, "price_type", role) or args.price_type,
    }
    suffix = ROLE_SUFFIX[role]
    missing = [
        f"--{key.replace('_', '-')}" + (f"-{suffix}" if executable and key == "asset_class" else "")
        for key, value in pick.items()
        if value is None
    ]
    if missing:
        raise ConfigError(
            f"Falta declarar {', '.join(missing)} para «{path.name}» (no hay {sidecar.name}). "
            "Sin la moneda no se puede comprobar que las series estén en la misma, y sin la "
            "clase de activo no se puede elegir el calendario ni la anualización (decisión D4)."
            + (
                " La clase de activo del instrumento ejecutable se declara aparte porque suele "
                "ser distinta de la de la serie analizada (p. ej. índice frente a ETF)."
                if executable
                else ""
            )
            + " Consulta las opciones con --help.",
            code="DEMO_METADATOS_FALTANTES",
            details={"archivo": str(path), "faltan": missing},
        )
    symbol = _option(args, "symbol", role) or _default_symbol(path)
    try:
        return build_metadata(symbol=symbol, name=symbol, **pick)
    except DataError as exc:
        raise DataError(
            f"Revisa los datos declarados en la línea de comandos para «{path.name}» "
            f"({ROLE_FLAGS[role]}): {exc.message}",
            code=exc.code,
            details=exc.details,
        ) from None


def _synthetic_notice(path: Path, declared: CSVMetadata, origin: DataOrigin) -> str | None:
    """Aviso si los datos se tratan como SINTÉTICOS aunque lo declarado decía históricos."""
    if origin != DataOrigin.SYNTHETIC or declared.is_synthetic:
        return None
    if origin_from_filename(path.name) == DataOrigin.SYNTHETIC:
        why = f"el nombre del archivo empieza por «{SYNTHETIC_FILE_PREFIX}»"
    else:
        why = "la columna «tipo_datos» del archivo marca las filas como SINTETICO"
    return (
        f"«{path.name}»: {why}, así que la serie se trata como datos SINTÉTICOS aunque los "
        "metadatos declaraban datos históricos. Nunca se presenta un dato sintético como "
        "histórico."
    )


def _ref(provider: CSVProvider, symbol: str, args: argparse.Namespace) -> SeriesRef:
    """Referencia a la serie con el rango pedido (por defecto, toda la cobertura del CSV)."""
    full = provider.get_history(symbol, *WIDE_RANGE)
    start = args.start or full.meta.coverage_start
    end = args.end or full.meta.coverage_end
    return SeriesRef(provider="csv", symbol=symbol, start=start.isoformat(), end=end.isoformat())


def build_config(
    args: argparse.Namespace, uploads: Path
) -> tuple[RunConfig, CSVProvider, list[str]]:
    """RunConfig del par pedido, el proveedor CSV con las series y los avisos de la carga."""
    paths: dict[str, Path | None] = {
        "a": args.csv_a,
        "b": args.csv_b,
        "exec_a": args.exec_a,
        "exec_b": args.exec_b,
    }
    notices: list[str] = []
    if paths["a"] is None and paths["b"] is None:
        if paths["exec_a"] is not None or paths["exec_b"] is not None:
            raise ConfigError(
                "--exec-a y --exec-b indican el instrumento ejecutable de tus propias series: "
                "úsalos junto con --csv-a y --csv-b.",
                code="DEMO_ARGUMENTOS",
            )
        provider = CSVProvider(DATA_DIR)
        symbols = {"a": DEFAULT_SYMBOLS[0], "b": DEFAULT_SYMBOLS[1]}
    elif paths["a"] is None or paths["b"] is None:
        raise ConfigError("Indica los dos archivos: --csv-a y --csv-b.", code="DEMO_ARGUMENTOS")
    else:
        provider = CSVProvider(uploads)
        metas: dict[str, CSVMetadata] = {}
        for role, path in paths.items():
            if path is None:
                continue
            if not path.is_file():
                raise ConfigError(
                    f"No existe el archivo {path} ({ROLE_FLAGS[role]}).",
                    code="DEMO_ARCHIVO_NO_ENCONTRADO",
                    details={"archivo": str(path)},
                )
            metas[role] = _meta_for(path, args, role)
        seen: dict[str, str] = {}
        for role, meta in metas.items():
            key = meta.symbol.strip().upper()
            if key in seen:
                other = seen[key]
                raise ConfigError(
                    f"{ROLE_FLAGS[other]} y {ROLE_FLAGS[role]} tienen el mismo símbolo "
                    f"(«{meta.symbol}»): usa --symbol-{ROLE_SUFFIX[other]} y "
                    f"--symbol-{ROLE_SUFFIX[role]} para distinguirlos.",
                    code="DEMO_SIMBOLO_REPETIDO",
                )
            seen[key] = role
        symbols = {}
        for role, meta in metas.items():
            path = paths[role]
            assert path is not None
            written = provider.register(path.read_bytes(), meta, filename=path.name)
            symbols[role] = written.symbol
            notice = _synthetic_notice(path, meta, written.meta.asset.origin)
            if notice:
                notices.append(notice)
    refs = {role: _ref(provider, symbol, args) for role, symbol in symbols.items()}
    pair = PairSpec(a=refs["a"], b=refs["b"], exec_a=refs.get("exec_a"), exec_b=refs.get("exec_b"))
    return RunConfig(pair=pair), provider, notices


# --------------------------------------------------------------------------- informe


def print_data(result: RunResult, notices: list[str]) -> None:
    title("1. Datos")
    for role in ROLES:
        series = result.series.get(role)
        if series is None:
            continue
        m, asset = series.meta, series.meta.asset
        print(f"  {ROLE_LABELS[role]} — {asset.symbol}: {asset.name}")
        executable = "ejecutable" if asset.executable else "NO ejecutable"
        print(
            f"    Moneda {asset.currency or 'desconocida'} · clase {asset.asset_class.value} "
            f"({executable}) · {PRICE_TYPE_LABELS[m.price_type]} · origen {asset.origin.value}"
        )
        print(
            f"    Cobertura {m.coverage_start} a {m.coverage_end} ({num(m.n_obs, 0)} observaciones)"
            f" · proveedor «{asset.provider}» · instantánea {m.snapshot_id}"
        )
        if m.seed is not None:
            print(f"    Semilla del generador: {m.seed}")
    if notices:
        print("  Avisos de la carga:")
        bullet(notices, indent="    ")


def print_validation(an: PairAnalysis) -> None:
    rep = an.validation
    title("2. Validación de datos")
    print(f"  {rep.status_message}")
    if rep.calendar:
        print(f"  Calendario: {rep.calendar.value}")
    bullet(f"[{i.severity.value}] {i.message}" for i in rep.issues)


def print_integration(an: PairAnalysis) -> None:
    if not an.integration:
        return
    title("3. Integración de cada serie (ADF y KPSS, solo formación)")
    for symbol, res in an.integration.items():
        print(f"  {symbol}: {INTEGRATION_LABELS[res.classification]}")
        print(
            f"    Niveles: ADF {format_pvalue(res.level_adf)} (rezagos {res.level_adf.lags}), "
            f"KPSS {format_pvalue(res.level_kpss)}"
        )
        print(
            f"    Diferencias: ADF {format_pvalue(res.diff_adf)}, "
            f"KPSS {format_pvalue(res.diff_kpss)}"
        )


def print_cointegration(an: PairAnalysis) -> None:
    if not an.eg:
        return
    title("4. Engle–Granger en ambas direcciones (solo formación)")
    print("  Dirección                         Estadístico  p original  p ajustado  Rezagos")
    for direction, eg in an.eg.items():
        label = f"{eg.dependent} sobre {eg.independent}"
        mark = " ←" if direction == an.chosen_direction else ""
        print(
            f"  {label:<33} {num(eg.statistic, 3):>11}  {p_value(eg.pvalue):>10}  "
            f"{p_value(an.eg_p_adjusted.get(direction)):>10}  {eg.lags:>7}{mark}"
        )
    chosen = an.eg[an.chosen_direction]
    crit = ", ".join(f"{k}: {num(v, 3)}" for k, v in chosen.critical_values.items())
    print(f"  Valores críticos ({chosen.dependent} sobre {chosen.independent}): {crit}")
    print(f"  Observaciones: {chosen.nobs} · tendencia «{chosen.trend}» · rezagos por AIC")
    if an.eg_decision:
        print(f"  {an.eg_decision.message}")
    if an.stability:
        print("  Estabilidad entre ventanas:")
        bullet(an.stability.reasons, indent="    ")


def print_status(an: PairAnalysis) -> None:
    title("5. Estado del par")
    print(f"  {STATUS_LABELS[an.status]}")
    bullet(c.message for c in an.causes)
    if an.discard_reasons:
        print("  ¿Por qué se descartaría este par?")
        bullet((c.message for c in an.discard_reasons), indent="    ")


def _unit_root_line(ou: OUFit) -> str | None:
    """Resultado de la prueba de raíz unitaria del AR(1) que decide la validez (decisión O7)."""
    if ou.df_stat is None or ou.df_crit is None:
        return None
    critical = (
        "valor crítico de Engle–Granger (N = 2): el spread es el residuo de una regresión estimada"
        if ou.df_estimated_residual
        else "valor crítico de Dickey–Fuller (N = 1)"
    )
    verdict = (
        "se rechaza la raíz unitaria"
        if ou.df_stat <= ou.df_crit
        else "NO se rechaza la raíz unitaria: b no se distingue de 1"
    )
    return (
        f"Dickey–Fuller del AR(1): estadístico {num(ou.df_stat, 2)} frente a {num(ou.df_crit, 2)} "
        f"al {num(100 * ou.df_alpha, 0)} % ({critical}) → {verdict}."
    )


def print_model(an: PairAnalysis) -> None:
    if an.hedge is None:
        return
    h = an.hedge
    title("6. Spread (α y β por MCO, solo formación)")
    print(
        f"  α = {num(h.alpha)} (EE {num(h.alpha_se)}) · β = {num(h.beta)} (EE {num(h.beta_se)}) "
        f"· R² = {num(h.r_squared, 3)} · {h.nobs} obs. ({h.estimation_start} a {h.estimation_end})"
    )
    print(f"  Tipo de spread: {h.spread_type.value}; unidades: {h.units}")
    print(f"  {h.interpretation}")

    title("7. Modelo Ornstein–Uhlenbeck (solo formación)")
    ou = an.ou
    if ou is None:
        return
    unit_root = _unit_root_line(ou)
    if not ou.valid:
        print(f"  OU NO VÁLIDO: {ou.message}")
        if ou.b is not None:
            print(
                f"  AR(1) estimado: a = {num(ou.a, 6)}, b = {num(ou.b, 4)} ({ou.nobs} transiciones)"
            )
        if unit_root:
            print(f"  {unit_root}")
        print("  No se muestran λ, μ, σ, vida media ni bandas; la estrategia queda bloqueada.")
        return
    unit = ou.time_unit
    print(f"  AR(1): a = {num(ou.a, 6)}, b = {num(ou.b, 4)} ({ou.nobs} transiciones)")
    if unit_root:
        print(f"  {unit_root}")
    print(f"  λ = {num(ou.lam)} por {unit}")
    print(f"  Vida media = {num(ou.half_life, 1)} {unit.replace('día', 'días', 1)}")
    print(f"  μ = {num(ou.mu, 5)} (EE {num(ou.mu_se, 5)}) — referencia estadística, no valor justo")
    print(f"  σ = {num(ou.sigma, 5)} unidades del spread por raíz de {unit}")
    print(f"  σ_eq = {num(ou.sigma_eq, 5)} (bandas μ ± 1 y ± 2 σ_eq)")
    print(f"  {ou.message}")
    bullet(ou.warnings)
    if an.forecasts:
        title("8. Pronósticos desde el último dato de formación")
        for fc in an.forecasts:
            print(
                f"  h = {fc.horizon:>2}: media {num(fc.mean, 5)}; intervalo al "
                f"{num(100 * fc.level, 0)} %: [{num(fc.lower, 5)}, {num(fc.upper, 5)}]"
            )
        print(f"  {BANDS_VS_INTERVALS_NOTE}")
    if an.residuals:
        title("9. Diagnóstico")
        print(f"  {an.residuals.message}")
        if an.ou_stability:
            print(f"  {an.ou_stability.message}")


def _segment_title(segment: Segment, result: RunResult) -> str:
    if segment == Segment.TEST and result.test_contaminated:
        return CONTAMINATED_TEST_TITLE
    return SEGMENT_TITLES[segment]


def print_metrics(segment: Segment, bt: BacktestResult, result: RunResult) -> None:
    m = bt.metrics
    print(f"\n  {_segment_title(segment, result)}")
    print(
        f"    Retorno neto: {pct(m.total_return)} · anualizado: {pct(m.annualized_return)} · "
        f"Sharpe: {num(m.sharpe, 2)} ({m.sharpe_convention})"
    )
    print(f"    Máximo drawdown: {pct(m.max_drawdown)} (duración {m.max_drawdown_duration} obs.)")
    print(
        f"    Operaciones cerradas: {m.n_trades} · abiertas al final: {m.n_open_at_end} · "
        f"aciertos: {pct(m.win_rate, 1)} · factor de beneficio: "
        f"{num(m.profit_factor, 2) if m.profit_factor is not None else 'no definido'}"
    )
    print(
        f"    Ganancia media: {num(m.avg_win, 2)} · pérdida media: {num(m.avg_loss, 2)} · "
        f"duración media: {num(m.avg_holding_bars, 1)} obs."
    )
    costs_share = pct(m.costs_pct_of_gross, 1) if m.costs_pct_of_gross is not None else "no aplica"
    print(
        f"    PnL bruto: {num(m.gross_pnl, 2)} · costes: {num(m.total_costs, 2)} "
        f"({costs_share} del bruto) · PnL neto: {num(m.net_pnl, 2)} "
        f"(realizado {num(m.realized_pnl, 2)}, no realizado {num(m.unrealized_pnl, 2)})"
    )
    print(
        f"    Rotación: {num(m.turnover, 2)}× · exposición bruta media: "
        f"{pct(m.avg_gross_exposure, 1)} · apalancamiento máximo: {num(m.max_leverage_used, 2)}×"
    )
    # El banner «solo ilustrativo» se muestra una vez, al comienzo de la sección.
    warnings = [w for w in bt.warnings + m.warnings if not w.startswith(ILLUSTRATIVE_BANNER)]
    bullet(dict.fromkeys(warnings), indent="    ")


def print_test_mark(result: RunResult) -> None:
    print(f"\n  Prueba contaminada: {contamination_text(result)}")
    if result.test_contaminated is not None:
        print(TEST_HISTORY_NOTE)


def print_backtests(result: RunResult) -> None:
    an = result.analysis
    title("10. Estrategia y backtest")
    if not an.strategy_allowed:
        print("  Backtest NO ejecutado: la estrategia está bloqueada (ver estado y descartes).")
        codes = [c.code for c in an.discard_reasons if c.code in STRATEGY_BLOCKING_CODES]
        if codes:
            print(f"  Bloqueo de la simulación: {', '.join(dict.fromkeys(codes))}.")
        if "PROXY_REQUERIDO" in codes:
            print(
                "  Para simular, indica el CSV del instrumento ejecutable con --exec-a / --exec-b "
                "(con su .meta.json o con --asset-class-exec-a, etc.)."
            )
        print_test_mark(result)
        return
    if an.illustrative_only:
        print(f"  *** {ILLUSTRATIVE_BANNER} ***")
        label = "Motivo" if len(an.illustrative_reasons) == 1 else "Motivos"
        print(f"  {label}:")
        bullet(an.illustrative_reasons, indent="    ")
    s = result.config.strategy
    print(
        f"  Entrada |z| > {num(s.entry_z, 1)} · salida {num(s.exit_z, 1)} · stop z "
        f"{num(s.stop_z, 1)} · stop monetario {pct(s.stop_loss_pct, 0)} del capital · capital "
        f"{num(s.initial_capital, 0)}"
    )
    print(f"  {EXECUTION_CONVENTION}")
    for segment, bt in result.backtests.items():
        print_metrics(segment, bt, result)
    for segment, message in result.segment_errors.items():
        print(f"\n  {_segment_title(segment, result)}: no se pudo simular. {message}")
    if Segment.TEST not in result.backtests and not result.test_included:
        print(
            "\n  Prueba: NO ejecutada. Usa --incluir-prueba para ejecutarla una sola vez con la "
            "configuración congelada (si después cambias los umbrales, queda contaminada)."
        )
    print_test_mark(result)


def print_report(result: RunResult, notices: list[str]) -> None:
    an = result.analysis
    print(WIDE)
    print("Laboratorio de Reversión a la Media — análisis de un par")
    print(WIDE)
    print(f"AVISO: {DISCLAIMER}")
    if result.is_synthetic:
        print("*** DATOS SINTÉTICOS: generados por el laboratorio, NO son precios históricos. ***")
    print(f"Par: {an.symbol_a} / {an.symbol_b}")
    print_data(result, notices)
    print_validation(an)
    print_integration(an)
    print_cointegration(an)
    print_status(an)
    print_model(an)
    print_backtests(result)
    warnings = [w for w in an.warnings if not w.startswith(ILLUSTRATIVE_BANNER)]
    if warnings:
        title("11. Avisos")
        bullet(warnings)


# --------------------------------------------------------------------------- argumentos

#: Traducción de los mensajes de error de argparse (vienen en inglés).
_ARGPARSE_ES = (
    ("the following arguments are required: ", "faltan los argumentos obligatorios: "),
    ("unrecognized arguments: ", "argumentos no reconocidos: "),
    ("invalid choice: ", "opción no válida: "),
    ("(choose from ", "(elige entre "),
    ("expected one argument", "falta el valor del argumento"),
    ("not allowed with argument", "no se puede usar junto con el argumento"),
    ("ambiguous option: ", "opción ambigua: "),
    (" could match ", " puede corresponder a "),
    ("argument ", "argumento "),
)


class SpanishArgumentParser(argparse.ArgumentParser):
    """argparse con la ayuda y los errores en español."""

    def error(self, message: str):  # type: ignore[override]
        for english, spanish in _ARGPARSE_ES:
            message = message.replace(english, spanish)
        self.exit(2, f"{self.prog}: error: {message}. Consulta las opciones con --help.\n")

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "uso:", 1)

    def format_help(self) -> str:
        text = super().format_help().replace("usage:", "uso:", 1)
        return text.replace("\noptions:\n", "\nopciones:\n", 1)


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"fecha no válida «{text}»: usa el formato AAAA-MM-DD (p. ej. 2020-01-31)"
        ) from None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    classes = [c.value for c in AssetClass]
    prices = [p.value for p in PriceType]
    parser = SpanishArgumentParser(
        add_help=False,
        description=(
            "Análisis completo de un par desde CSV. Sin argumentos usa el par SINTÉTICO de "
            "examples/data/. " + DISCLAIMER
        ),
        epilog=(
            "Cada CSV necesita columnas date y close (fechas AAAA-MM-DD). Si junto al CSV existe "
            "<nombre>.meta.json (lo crea make fetch-example), sus metadatos se usan tal cual; si "
            "no, declara --currency, --asset-class y --price-type (y, si la serie B difiere, "
            "--currency-b, --asset-class-b, --price-type-b). Un índice o futuro necesita un "
            "instrumento ejecutable para simular la estrategia: --exec-a/--exec-b con la clase "
            "de activo --asset-class-exec-a/--asset-class-exec-b (o su .meta.json)."
        ),
    )
    parser.add_argument("-h", "--help", action="help", help="Muestra esta ayuda y termina")
    parser.add_argument("--csv-a", type=Path, help="CSV de la serie A")
    parser.add_argument("--csv-b", type=Path, help="CSV de la serie B")
    parser.add_argument(
        "--exec-a",
        type=Path,
        help="CSV del instrumento ejecutable (proxy) de A, si A es un índice o futuro",
    )
    parser.add_argument(
        "--exec-b",
        type=Path,
        help="CSV del instrumento ejecutable (proxy) de B, si B es un índice o futuro",
    )
    parser.add_argument(
        "--currency", help="Moneda de todas las series (código de 3 letras, p. ej. USD)"
    )
    parser.add_argument(
        "--asset-class", choices=classes, help="Clase de activo de A y B (no de los ejecutables)"
    )
    parser.add_argument("--price-type", choices=prices, help="Tipo de precio de todas las series")
    for role in ROLES:
        suffix = ROLE_SUFFIX[role]
        name = ROLE_LABELS[role].lower()
        parser.add_argument(
            f"--symbol-{suffix}",
            dest=f"symbol_{role}",
            help=f"Símbolo de {name} (por defecto, el nombre del archivo)",
        )
        parser.add_argument(
            f"--currency-{suffix}", dest=f"currency_{role}", help=f"Moneda solo de {name}"
        )
        parser.add_argument(
            f"--asset-class-{suffix}",
            dest=f"asset_class_{role}",
            choices=classes,
            help=f"Clase de activo solo de {name}",
        )
        parser.add_argument(
            f"--price-type-{suffix}",
            dest=f"price_type_{role}",
            choices=prices,
            help=f"Tipo de precio solo de {name}",
        )
    parser.add_argument("--start", type=_iso_date, help="Fecha inicial AAAA-MM-DD")
    parser.add_argument("--end", type=_iso_date, help="Fecha final AAAA-MM-DD")
    parser.add_argument(
        "--incluir-prueba",
        action="store_true",
        help="Ejecuta también la muestra de prueba (una sola vez, configuración congelada)",
    )
    parser.add_argument(
        "--reproducir",
        type=Path,
        metavar="RUN_CONFIG",
        help=(
            "Reproduce un run_config.json exportado usando solo la caché local (incluye la "
            "prueba si la configuración la incluía)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Carpeta de exportación"
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Caché de instantáneas (por defecto, data/cache)"
    )
    parser.add_argument(
        "--excel-es", action="store_true", help="CSV para Excel en español (; y coma decimal)"
    )
    args = parser.parse_args(argv)
    if args.reproducir is not None:
        conflicting = [
            flag
            for flag, value in (
                ("--csv-a", args.csv_a),
                ("--csv-b", args.csv_b),
                ("--exec-a", args.exec_a),
                ("--exec-b", args.exec_b),
                ("--start", args.start),
                ("--end", args.end),
            )
            if value is not None
        ]
        if conflicting:
            parser.error(
                f"--reproducir no admite {', '.join(conflicting)}: las series y las fechas ya "
                "están fijadas en el run_config.json"
            )
    return args


def reproduce_command(config_path: Path, args: argparse.Namespace) -> str:
    """Orden que reproduce exactamente los mismos archivos (misma caché y mismo formato)."""
    parts = ["python", "examples/demo_ko_pep.py", "--reproducir", str(config_path)]
    if args.cache_dir is not None:
        parts += ["--cache-dir", str(args.cache_dir)]
    if args.excel_es:
        parts.append("--excel-es")
    return " ".join(shlex.quote(part) for part in parts)


# --------------------------------------------------------------------------- main


def _os_error_text(exc: OSError) -> str:
    target = exc.filename or "un archivo"
    if isinstance(exc, FileNotFoundError):
        return f"No existe {target}."
    if isinstance(exc, PermissionError):
        return f"No hay permiso para leer o escribir {target}."
    if isinstance(exc, IsADirectoryError):
        return f"{target} es una carpeta, no un archivo."
    return f"No se pudo leer o escribir {target} (error del sistema {exc.errno})."


def _validation_error_text(exc: ValidationError) -> str:
    fields = sorted({".".join(str(p) for p in err.get("loc", ())) for err in exc.errors()})
    return (
        "Hay valores no válidos en la configuración del laboratorio (variables MRLAB_* o "
        f"archivo .env): {', '.join(f or '(general)' for f in fields)}. Revísalos."
    )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # consolas sin UTF-8 (Windows)
    args = parse_args(argv)
    try:
        cache = SnapshotCache(args.cache_dir or get_settings().cache_dir)
        with tempfile.TemporaryDirectory(prefix="mrlab_demo_") as tmp:
            notices: list[str] = []
            if args.reproducir:
                config = load_run_config(args.reproducir)
                data = DataService(ProviderRegistry(), cache)  # solo caché: nunca descarga
            else:
                config, provider, notices = build_config(args, Path(tmp))
                data = DataService(ProviderRegistry({"csv": provider}), cache)
            # include_test solo puede ACTIVAR la prueba: al reproducir manda el run_config.json.
            result = run_pair(config, data, include_test=args.incluir_prueba)
            print_report(result, notices)
            out_dir = args.output_dir / export_basename(result)
            paths = export_run(result, out_dir, excel_es=args.excel_es)
    except LabError as exc:
        print(f"\nERROR ({exc.code}): {exc.message}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"\nERROR (ARCHIVO): {_os_error_text(exc)}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"\nERROR (CONFIGURACION): {_validation_error_text(exc)}", file=sys.stderr)
        return 2

    title("12. Exportación")
    bullet(str(p) for p in paths)
    config_path = next(p for p in paths if p.name.endswith("run_config.json"))
    print(f"  Hash de la configuración: {result.config.config_hash}")
    print(f"  Prueba incluida en la configuración: {'sí' if result.config.include_test else 'no'}")
    print(
        f"  Para reproducir (mismos archivos, byte a byte): {reproduce_command(config_path, args)}"
    )
    print("  (Añade --output-dir <otra carpeta> si quieres comparar los archivos con estos.)")
    print(f"\nAVISO: {DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
