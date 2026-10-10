"""Utilidades comunes de la línea de órdenes de los ejemplos.

- ``SpanishArgumentParser``: argparse con la ayuda y los errores en español (argparse
  los escribe en inglés). Los errores terminan con código 2 y remiten a ``--help``.
- ``iso_date``: tipo de argumento para fechas AAAA-MM-DD con error en español.
- ``quote_command``: une una orden para copiarla y pegarla en la consola del sistema
  (comillas dobles en Windows, comillas simples POSIX en el resto).

Los scripts de ``examples/`` lo importan como módulo hermano (``from _cli import …``).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from collections.abc import Sequence
from datetime import date

#: Traducción de los mensajes de error de argparse, en orden: las frases que contienen
#: «argument » se traducen antes del reemplazo genérico (la última entrada).
ARGPARSE_ES: tuple[tuple[str, str], ...] = (
    ("the following arguments are required: ", "faltan los argumentos obligatorios: "),
    ("unrecognized arguments: ", "argumentos no reconocidos: "),
    ("invalid choice: ", "opción no válida: "),
    ("(choose from ", "(elige entre "),
    ("ignored explicit argument ", "esta opción no admite un valor; se recibió "),
    ("expected one argument", "falta el valor del argumento"),
    ("expected at most one argument", "admite como máximo un valor"),
    ("expected at least one argument", "falta al menos un valor"),
    ("not allowed with argument", "no se puede usar junto con el argumento"),
    ("ambiguous option: ", "opción ambigua: "),
    (" could match ", " puede corresponder a "),
    ("argument ", "argumento "),
)
#: «invalid int value: 'x'» y similares (tipos que lanzan ValueError).
_INVALID_VALUE = re.compile(r"invalid [\w.]+ value: ")


def translate_argparse_message(message: str) -> str:
    """Mensaje de error de argparse en español (los propios ya vienen en español)."""
    message = _INVALID_VALUE.sub("valor no válido: ", message)
    for english, spanish in ARGPARSE_ES:
        message = message.replace(english, spanish)
    return message


class SpanishArgumentParser(argparse.ArgumentParser):
    """argparse con la ayuda y los errores en español.

    ``-h``/``--help`` se añade con su texto en español (salvo ``add_help=False``).
    """

    def __init__(self, *args, add_help: bool = True, **kwargs) -> None:
        super().__init__(*args, add_help=False, **kwargs)
        if add_help:
            self.add_argument("-h", "--help", action="help", help="Muestra esta ayuda y termina")

    def error(self, message: str):  # type: ignore[override]
        text = translate_argparse_message(message)
        self.exit(2, f"{self.prog}: error: {text}. Consulta las opciones con --help.\n")

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "uso:", 1)

    def format_help(self) -> str:
        text = super().format_help().replace("usage:", "uso:", 1)
        text = text.replace("\npositional arguments:\n", "\nargumentos posicionales:\n", 1)
        return text.replace("\noptions:\n", "\nopciones:\n", 1)


def iso_date(text: str) -> date:
    """Fecha AAAA-MM-DD para argparse (error en español si no es válida)."""
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"fecha no válida «{text}»: usa el formato AAAA-MM-DD (p. ej. 2020-01-31)"
        ) from None


#: Caracteres que obligan a citar un argumento en cmd.exe/PowerShell (además de espacios).
_WINDOWS_SPECIAL = set(" \t&|<>^%()!;,=")


def _quote_windows(part: str) -> str:
    # Las rutas de Windows no pueden contener comillas dobles: basta con encerrarlas.
    # list2cmdline solo cita con espacios y no protege los metacaracteres de cmd.exe (&, |, %…).
    if part and not (set(part) & _WINDOWS_SPECIAL) and '"' not in part:
        return part
    return subprocess.list2cmdline([part]) if '"' in part else f'"{part}"'


def quote_command(parts: Sequence[str], *, windows: bool | None = None) -> str:
    """Orden lista para pegar en la consola: en Windows (``os.name == "nt"``) cada argumento
    con espacios o metacaracteres de cmd.exe va entre comillas dobles; en el resto, con
    ``shlex.quote`` (POSIX). Limitación: dentro de comillas, cmd.exe aún expande ``%var%``."""
    if windows is None:
        windows = os.name == "nt"
    if windows:
        return " ".join(_quote_windows(part) for part in parts)
    return " ".join(shlex.quote(part) for part in parts)
