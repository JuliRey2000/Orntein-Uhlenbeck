"""Regenera los CSV SINTÉTICOS de ``examples/data/`` (par cointegrado preset SINT-COI).

Los datos NO son históricos: salen de ``SyntheticProvider`` (escenario
«cointegrado», semilla 7, 1.500 días hábiles desde 2018-01-01) y se guardan como
CSV ``date,close`` con su archivo de metadatos ``.meta.json`` (origen SINTETICO).
El resultado es determinista: ejecutar este script otra vez produce los mismos
archivos byte a byte.

Uso (desde la raíz del repositorio):  python examples/generar_csv_sinteticos.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR / "data"

try:
    import mrlab  # noqa: F401
except ImportError:  # permite ejecutarlo sin instalar el paquete
    sys.path.insert(0, str(EXAMPLES_DIR.parent / "backend" / "src"))

from mrlab.data.providers.csv_provider import CSVMetadata, CSVProvider  # noqa: E402
from mrlab.data.providers.synthetic import SyntheticProvider  # noqa: E402
from mrlab.domain.models import DataOrigin, PriceSeries, PriceType  # noqa: E402

SYMBOLS = ("SINT-COI-A", "SINT-COI-B")


def to_csv(series: PriceSeries) -> str:
    """CSV ``date,close`` con 12 cifras significativas y fin de línea LF."""
    lines = ["date,close"]
    for ts, close in series.data["close"].items():
        lines.append(f"{ts.date().isoformat()},{format(float(close), '.12g')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    generator = SyntheticProvider()
    target = CSVProvider(DATA_DIR)
    for symbol in SYMBOLS:
        series = generator.get_history(symbol, date(1900, 1, 1), date(2100, 12, 31))
        asset = series.meta.asset
        meta = CSVMetadata(
            symbol=asset.symbol,
            name=asset.name,
            currency=asset.currency or "USD",
            asset_class=asset.asset_class,
            price_type=PriceType.SYNTHETIC,
            origin=DataOrigin.SYNTHETIC,
            notes=(series.meta.source_note,),
        )
        written = target.register(to_csv(series), meta)
        print(f"{written.symbol}: {written.meta.n_obs} observaciones SINTÉTICAS en {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
