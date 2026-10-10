"""Descarga KO y PEP de Yahoo Finance a ``examples/data/real/`` (requiere internet).

Escribe, por cada símbolo, ``<SÍMBOLO>.csv`` (date, close y, si existen, open,
high, low, volume; precios ajustados por splits y dividendos) y
``<SÍMBOLO>.meta.json`` con la moneda, la clase de activo y el tipo de precio que
informa Yahoo. La demo lee ese archivo de metadatos automáticamente:

    python examples/fetch_ko_pep.py
    python examples/demo_ko_pep.py --csv-a examples/data/real/KO.csv --csv-b examples/data/real/PEP.csv

Estos datos NO se versionan (``examples/data/real/`` está en .gitignore): los
términos de uso de Yahoo desaconsejan redistribuirlos (decisión D9). Yahoo
Finance es una fuente gratuita y no oficial: puede fallar o limitar peticiones.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXAMPLES_DIR / "data" / "real"
DEFAULT_SYMBOLS = ("KO", "PEP")

try:
    import mrlab  # noqa: F401
except ImportError:  # permite ejecutarlo sin instalar el paquete
    sys.path.insert(0, str(EXAMPLES_DIR.parent / "backend" / "src"))

from mrlab import DISCLAIMER  # noqa: E402
from mrlab.config import get_settings  # noqa: E402
from mrlab.data.providers.csv_provider import CSVProvider, build_metadata  # noqa: E402
from mrlab.data.providers.yahoo import YahooProvider  # noqa: E402
from mrlab.domain.errors import LabError  # noqa: E402
from mrlab.domain.models import DataOrigin, PriceSeries, PriceType  # noqa: E402

COLUMNS = ("close", "open", "high", "low", "volume")


def to_csv(series: PriceSeries) -> str:
    cols = [c for c in COLUMNS if c in series.data.columns]
    lines = [",".join(["date", *cols])]
    for ts, row in series.data[cols].iterrows():
        cells = ["" if not math.isfinite(v) else format(float(v), ".12g") for v in row]
        lines.append(",".join([ts.date().isoformat(), *cells]))
    return "\n".join(lines) + "\n"


def fetch(
    provider: YahooProvider, target: CSVProvider, symbol: str, start: date, end: date
) -> None:
    series = provider.get_history(symbol, start, end)
    asset = series.meta.asset
    if not asset.currency:
        raise SystemExit(
            f"Yahoo no informó la moneda de «{symbol}»: no se puede crear su archivo de metadatos. "
            "Declárala a mano al usar la demo (--currency)."
        )
    retrieved = series.meta.retrieved_at.isoformat() if series.meta.retrieved_at else "?"
    meta = build_metadata(  # errores en español (CSV_METADATOS_INVALIDOS)
        symbol=asset.symbol,
        name=asset.name,
        currency=asset.currency,
        asset_class=asset.asset_class,
        price_type=PriceType.ADJUSTED_CLOSE,
        origin=DataOrigin.HISTORICAL,
        notes=(
            *asset.notes,
            series.meta.source_note,
            f"Descargado de Yahoo Finance el {retrieved} (UTC).",
        ),
    )
    written = target.register(to_csv(series), meta)
    print(
        f"{written.symbol}: {written.meta.n_obs} observaciones, {written.meta.coverage_start} a "
        f"{written.meta.coverage_end} ({asset.currency}, {asset.asset_class.value}) → "
        f"{target.directory}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Descarga series de Yahoo Finance a CSV para la demo. " + DISCLAIMER
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="Por defecto, hoy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    settings = get_settings()
    provider = YahooProvider(
        max_retries=settings.yahoo_max_retries, backoff_base=settings.yahoo_backoff_base_seconds
    )
    target = CSVProvider(args.output_dir)
    end = args.end or date.today()
    print("Yahoo Finance es una fuente gratuita y no oficial, adecuada para uso educativo.")
    try:
        for symbol in args.symbols:
            fetch(provider, target, symbol, args.start, end)
    except LabError as exc:
        print(f"ERROR ({exc.code}): {exc.message}", file=sys.stderr)
        return 1
    print("Estos datos son solo para tu uso local: no los subas al repositorio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
