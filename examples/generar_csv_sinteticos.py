"""Regenera los CSV SINTÉTICOS de ``examples/data/`` (par cointegrado preset SINT-COI).

Los datos NO son históricos: salen de ``SyntheticProvider`` (escenario
«cointegrado», semilla 7, 1.500 días hábiles desde 2018-01-01) y se guardan como
CSV ``date,close,tipo_datos`` con su archivo de metadatos ``.meta.json``.

La marca SINTÉTICO viaja con los datos por tres vías independientes, de modo que
ninguna copia parcial pueda presentarse como histórica:
- la columna ``tipo_datos = SINTETICO`` en cada fila del CSV;
- el prefijo ``SINTETICO_`` del nombre del archivo;
- los metadatos (origen SINTETICO y la semilla ``seed`` del generador).

El resultado es determinista: ejecutar este script otra vez produce los mismos
archivos byte a byte.

Uso (desde la raíz del repositorio):  python examples/generar_csv_sinteticos.py [carpeta]
(por defecto escribe en ``examples/data/``; con ``carpeta`` escribe allí, p. ej. para
comprobar que los archivos versionados coinciden con lo que genera el script).
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

from mrlab.data.providers.csv_provider import (  # noqa: E402
    DATA_TYPE_COLUMN,
    CSVProvider,
    build_metadata,
)
from mrlab.data.providers.synthetic import SyntheticProvider  # noqa: E402
from mrlab.domain.errors import LabError  # noqa: E402
from mrlab.domain.models import DataOrigin, PriceSeries, PriceType  # noqa: E402

SYMBOLS = ("SINT-COI-A", "SINT-COI-B")


def to_csv(series: PriceSeries) -> str:
    """CSV ``date,close,tipo_datos`` con 12 cifras significativas y fin de línea LF."""
    marker = DataOrigin.SYNTHETIC.value
    lines = [f"date,close,{DATA_TYPE_COLUMN}"]
    for ts, close in series.data["close"].items():
        lines.append(f"{ts.date().isoformat()},{format(float(close), '.12g')},{marker}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        print("Uso: python examples/generar_csv_sinteticos.py [carpeta]", file=sys.stderr)
        return 2
    directory = Path(args[0]) if args else DATA_DIR
    generator = SyntheticProvider()
    target = CSVProvider(directory)
    try:
        for symbol in SYMBOLS:
            series = generator.get_history(symbol, date(1900, 1, 1), date(2100, 12, 31))
            asset = series.meta.asset
            meta = build_metadata(
                symbol=asset.symbol,
                name=asset.name,
                currency=asset.currency or "USD",
                asset_class=asset.asset_class,
                price_type=PriceType.SYNTHETIC,
                origin=DataOrigin.SYNTHETIC,
                notes=(series.meta.source_note,),
                seed=series.meta.seed,
            )
            written = target.register(to_csv(series), meta)
            print(
                f"{written.symbol}: {written.meta.n_obs} observaciones SINTÉTICAS "
                f"(semilla {written.meta.seed}) en {directory}"
            )
    except LabError as exc:
        print(f"ERROR ({exc.code}): {exc.message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
