"""Exporta el esquema OpenAPI de la API a ``backend/openapi.json``.

El archivo se versiona: en la fase 3 genera los tipos TypeScript del frontend
(``openapi-typescript``) y una prueba comprueba que está al día.

Uso (desde la raíz del repositorio, con el paquete instalado):

    python backend/scripts/export_openapi.py            # escribe backend/openapi.json
    python backend/scripts/export_openapi.py --check    # solo comprueba (código 1 si difiere)
    python backend/scripts/export_openapi.py --output otra/ruta.json

No usa la red ni crea las carpetas de datos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "openapi.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exporta el esquema OpenAPI de la API del laboratorio a un archivo JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"archivo de salida (por defecto {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="no escribe nada: termina con código 1 si el archivo no está al día",
    )
    args = parser.parse_args(argv)

    from mrlab.api.main import openapi_text

    text = openapi_text()
    output: Path = args.output
    if args.check:
        current = output.read_text(encoding="utf-8") if output.is_file() else None
        if current != text:
            print(
                f"{output} no está al día con el código de la API. Actualízalo con "
                "`make openapi` (o `python backend/scripts/export_openapi.py`).",
                file=sys.stderr,
            )
            return 1
        print(f"{output} está al día.")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    print(f"Esquema OpenAPI escrito en {output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
