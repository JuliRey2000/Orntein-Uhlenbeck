"""Demo reproducible: análisis completo de un par desde CSV (fase 1).

Sin argumentos analiza el par SINTÉTICO de ``examples/data/`` (no necesita
internet). Con ``--csv-a`` y ``--csv-b`` analiza tus propios CSV (por ejemplo, KO
y PEP descargados con ``make fetch-example``). Imprime un informe en español y
exporta los resultados (CSV + run_config.json + LEEME.txt) a
``examples/output/<par>/``. La muestra de PRUEBA no se ejecuta salvo que se pida
con ``--incluir-prueba``: se ejecuta una sola vez, con la configuración congelada.

Ejemplos (desde la raíz del repositorio):
    python examples/demo_ko_pep.py
    python examples/demo_ko_pep.py --csv-a examples/data/real/KO.csv --csv-b examples/data/real/PEP.csv
    python examples/demo_ko_pep.py --csv-a a.csv --csv-b b.csv --currency USD \\
        --asset-class accion --price-type cierre_ajustado
    python examples/demo_ko_pep.py --reproducir examples/output/<par>/run_config.json
"""

from __future__ import annotations

import argparse
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
from mrlab.analysis.pair_pipeline import STATUS_LABELS, PairAnalysis  # noqa: E402
from mrlab.analysis.runner import RunResult, run_pair  # noqa: E402
from mrlab.backtest.engine import EXECUTION_CONVENTION, BacktestResult  # noqa: E402
from mrlab.config import get_settings  # noqa: E402
from mrlab.data.cache import SnapshotCache  # noqa: E402
from mrlab.data.providers.csv_provider import META_SUFFIX, CSVMetadata, CSVProvider  # noqa: E402
from mrlab.data.registry import ProviderRegistry  # noqa: E402
from mrlab.data.service import DataService  # noqa: E402
from mrlab.domain.errors import LabError  # noqa: E402
from mrlab.domain.models import PRICE_TYPE_LABELS, AssetClass, PriceType, Segment  # noqa: E402
from mrlab.model.ou import BANDS_VS_INTERVALS_NOTE  # noqa: E402
from mrlab.reporting.export import export_basename, export_run  # noqa: E402
from mrlab.reporting.run_config import (  # noqa: E402
    PairSpec,
    RunConfig,
    SeriesRef,
    load_run_config,
)
from mrlab.stats.integration import INTEGRATION_LABELS, format_pvalue  # noqa: E402

WIDE = "=" * 78
SEGMENT_TITLES = {
    Segment.TRAIN: "Formación (DENTRO de muestra: solo ilustrativo)",
    Segment.VALIDATION: "Validación (fuera de muestra)",
    Segment.TEST: "Prueba (fuera de muestra, configuración congelada)",
}
WIDE_RANGE = (date(1900, 1, 1), date(2100, 12, 31))


# --------------------------------------------------------------------------- formato


def num(x: float | None, decimals: int = 4) -> str:
    """Número con coma decimal y punto de miles; «—» si no hay valor."""
    if x is None:
        return "—"
    return f"{x:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x: float | None, decimals: int = 2) -> str:
    return "—" if x is None else f"{num(100 * x, decimals)} %"


def p_value(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else num(p)


def title(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def bullet(lines: Iterable[str], indent: str = "  ") -> None:
    for line in lines:
        print(f"{indent}- {line}")


# --------------------------------------------------------------------------- datos


def _meta_for(path: Path, args: argparse.Namespace, leg: str) -> CSVMetadata:
    """Metadatos del CSV: su archivo ``.meta.json`` si existe; si no, los de la línea de comandos."""
    sidecar = path.with_name(path.stem + META_SUFFIX)
    if sidecar.exists():
        return CSVMetadata.model_validate_json(sidecar.read_text(encoding="utf-8"))
    pick = {
        "currency": getattr(args, f"currency_{leg}") or args.currency,
        "asset_class": getattr(args, f"asset_class_{leg}") or args.asset_class,
        "price_type": getattr(args, f"price_type_{leg}") or args.price_type,
    }
    missing = [f"--{k.replace('_', '-')}" for k, v in pick.items() if v is None]
    if missing:
        raise SystemExit(
            f"Falta declarar {', '.join(missing)} para «{path.name}» (no hay {sidecar.name}). "
            "Sin la moneda no se puede comprobar que ambas series estén en la misma, y sin la "
            "clase de activo no se puede elegir el calendario ni la anualización (decisión D4). "
            "Consulta las opciones con --help."
        )
    symbol = getattr(args, f"symbol_{leg}") or path.stem.upper()
    return CSVMetadata(symbol=symbol, name=symbol, **pick)


def _ref(provider: CSVProvider, symbol: str, args: argparse.Namespace) -> SeriesRef:
    """Referencia a la serie con el rango pedido (por defecto, toda la cobertura del CSV)."""
    full = provider.get_history(symbol, *WIDE_RANGE)
    start = args.start or full.meta.coverage_start
    end = args.end or full.meta.coverage_end
    return SeriesRef(provider="csv", symbol=symbol, start=start.isoformat(), end=end.isoformat())


def build_config(args: argparse.Namespace, uploads: Path) -> tuple[RunConfig, CSVProvider]:
    if args.csv_a is None and args.csv_b is None:
        provider = CSVProvider(DATA_DIR)
        symbols = DEFAULT_SYMBOLS
    elif args.csv_a is None or args.csv_b is None:
        raise SystemExit("Indica los dos archivos: --csv-a y --csv-b.")
    else:
        provider = CSVProvider(uploads)
        metas = []
        for leg, path in (("a", args.csv_a), ("b", args.csv_b)):
            if not path.is_file():
                raise SystemExit(f"No existe el archivo {path}.")
            metas.append(_meta_for(path, args, leg))
        if metas[0].symbol.upper() == metas[1].symbol.upper():
            raise SystemExit(
                f"Los dos CSV tienen el mismo símbolo («{metas[0].symbol}»): usa --symbol-a y "
                "--symbol-b para distinguirlos."
            )
        for path, meta in zip((args.csv_a, args.csv_b), metas, strict=True):
            provider.register(path.read_bytes(), meta)
        symbols = (metas[0].symbol, metas[1].symbol)
    pair = PairSpec(a=_ref(provider, symbols[0], args), b=_ref(provider, symbols[1], args))
    return RunConfig(pair=pair), provider


# --------------------------------------------------------------------------- informe


def print_data(result: RunResult) -> None:
    title("1. Datos")
    for role in ("a", "b", "exec_a", "exec_b"):
        series = result.series.get(role)
        if series is None:
            continue
        m, asset = series.meta, series.meta.asset
        print(f"  {asset.symbol}: {asset.name}")
        print(
            f"    Moneda {asset.currency} · clase {asset.asset_class.value} · "
            f"{PRICE_TYPE_LABELS[m.price_type]} · origen {asset.origin.value}"
        )
        print(
            f"    Cobertura {m.coverage_start} a {m.coverage_end} ({num(m.n_obs, 0)} observaciones)"
            f" · proveedor «{asset.provider}» · instantánea {m.snapshot_id}"
        )


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
            f"{p_value(an.eg_p_adjusted[direction]):>10}  {eg.lags:>7}{mark}"
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
    if not ou.valid:
        print(f"  OU NO VÁLIDO: {ou.message}")
        print("  No se muestran λ, μ, σ, vida media ni bandas; la estrategia queda bloqueada.")
        return
    unit = ou.time_unit
    print(f"  AR(1): a = {num(ou.a, 6)}, b = {num(ou.b, 4)} ({ou.nobs} transiciones)")
    print(f"  λ = {num(ou.lam)} por {unit}")
    print(f"  Vida media = {num(ou.half_life, 1)} {unit.replace('día', 'días', 1)}")
    print(f"  μ = {num(ou.mu, 5)} (EE {num(ou.mu_se, 5)}) — referencia estadística, no valor justo")
    print(f"  σ = {num(ou.sigma, 5)} unidades del spread por raíz de {unit}")
    print(f"  σ_eq = {num(ou.sigma_eq, 5)} (bandas μ ± 1 y ± 2 σ_eq)")
    print(f"  {ou.message}")
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


def print_metrics(segment: Segment, bt: BacktestResult) -> None:
    m = bt.metrics
    print(f"\n  {SEGMENT_TITLES[segment]}")
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
    bullet(dict.fromkeys(bt.warnings + m.warnings), indent="    ")


def print_backtests(result: RunResult) -> None:
    an = result.analysis
    title("10. Estrategia y backtest")
    if not an.strategy_allowed:
        print("  Backtest NO ejecutado: la estrategia está bloqueada (ver estado y descartes).")
        return
    s = result.config.strategy
    print(
        f"  Entrada |z| > {num(s.entry_z, 1)} · salida {num(s.exit_z, 1)} · stop z "
        f"{num(s.stop_z, 1)} · stop monetario {pct(s.stop_loss_pct, 0)} del capital · capital "
        f"{num(s.initial_capital, 0)}"
    )
    print(f"  {EXECUTION_CONVENTION}")
    for segment, bt in result.backtests.items():
        print_metrics(segment, bt)
    if Segment.TEST not in result.backtests:
        print(
            "\n  Prueba: NO ejecutada. Usa --incluir-prueba para ejecutarla una sola vez con la "
            "configuración congelada (si después cambias los umbrales, queda contaminada)."
        )


def print_report(result: RunResult) -> None:
    an = result.analysis
    print(WIDE)
    print("Laboratorio de Reversión a la Media — análisis de un par")
    print(WIDE)
    print(f"AVISO: {DISCLAIMER}")
    if result.is_synthetic:
        print("*** DATOS SINTÉTICOS: generados por el laboratorio, NO son precios históricos. ***")
    print(f"Par: {an.symbol_a} / {an.symbol_b}")
    print_data(result)
    print_validation(an)
    print_integration(an)
    print_cointegration(an)
    print_status(an)
    print_model(an)
    print_backtests(result)
    if an.warnings:
        title("11. Avisos")
        bullet(an.warnings)


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    classes = [c.value for c in AssetClass]
    prices = [p.value for p in PriceType]
    parser = argparse.ArgumentParser(
        description=(
            "Análisis completo de un par desde CSV. Sin argumentos usa el par SINTÉTICO de "
            "examples/data/. " + DISCLAIMER
        ),
        epilog=(
            "Cada CSV necesita columnas date y close (fechas AAAA-MM-DD). Si junto al CSV existe "
            "<nombre>.meta.json (lo crea make fetch-example), sus metadatos se usan tal cual; si "
            "no, declara --currency, --asset-class y --price-type (y, si la serie B difiere, "
            "--currency-b, --asset-class-b, --price-type-b)."
        ),
    )
    parser.add_argument("--csv-a", type=Path, help="CSV de la serie A")
    parser.add_argument("--csv-b", type=Path, help="CSV de la serie B")
    parser.add_argument("--symbol-a", help="Símbolo de A (por defecto, el nombre del archivo)")
    parser.add_argument("--symbol-b", help="Símbolo de B (por defecto, el nombre del archivo)")
    parser.add_argument(
        "--currency", help="Moneda de ambas series (código de 3 letras, p. ej. USD)"
    )
    parser.add_argument("--asset-class", choices=classes, help="Clase de activo de ambas series")
    parser.add_argument("--price-type", choices=prices, help="Tipo de precio de ambas series")
    for leg in ("a", "b"):
        name = leg.upper()
        parser.add_argument(f"--currency-{leg}", help=f"Moneda solo de la serie {name}")
        parser.add_argument(
            f"--asset-class-{leg}", choices=classes, help=f"Clase de activo solo de {name}"
        )
        parser.add_argument(
            f"--price-type-{leg}", choices=prices, help=f"Tipo de precio solo de {name}"
        )
    parser.add_argument("--start", type=date.fromisoformat, help="Fecha inicial AAAA-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="Fecha final AAAA-MM-DD")
    parser.add_argument(
        "--incluir-prueba",
        action="store_true",
        help="Ejecuta también la muestra de prueba (una sola vez, configuración congelada)",
    )
    parser.add_argument(
        "--reproducir",
        type=Path,
        metavar="RUN_CONFIG",
        help="Reproduce un run_config.json exportado usando solo la caché local",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # consolas sin UTF-8 (Windows)
    args = parse_args(argv)
    cache = SnapshotCache(args.cache_dir or get_settings().cache_dir)
    try:
        with tempfile.TemporaryDirectory(prefix="mrlab_demo_") as tmp:
            if args.reproducir:
                config = load_run_config(args.reproducir)
                data = DataService(ProviderRegistry(), cache)  # solo caché: nunca descarga
            else:
                config, provider = build_config(args, Path(tmp))
                data = DataService(ProviderRegistry({"csv": provider}), cache)
            result = run_pair(config, data, include_test=args.incluir_prueba)
            print_report(result)
            out_dir = args.output_dir / export_basename(result)
            paths = export_run(result, out_dir, excel_es=args.excel_es)
    except LabError as exc:
        print(f"\nERROR ({exc.code}): {exc.message}", file=sys.stderr)
        return 2

    title("12. Exportación")
    bullet(str(p) for p in paths)
    config_path = next(p for p in paths if p.name.endswith("run_config.json"))
    print(f"  Hash de la configuración: {result.config.config_hash}")
    print(f"  Para reproducir: python examples/demo_ko_pep.py --reproducir {config_path}")
    print(f"\nAVISO: {DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
