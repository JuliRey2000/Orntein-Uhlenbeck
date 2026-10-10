"""Metadatos del paquete: ninguna ruta sale de backend/ (regresión de la instalación editable).

``pip install -e ./backend`` fallaba en un clon nuevo porque ``backend/pyproject.toml``
declaraba ``readme = "../README.md"``: setuptools no admite archivos fuera de la carpeta
del proyecto. Estas pruebas impiden que vuelva una ruta así en ``[project]`` o en
``[tool.setuptools]``.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

BACKEND = Path(__file__).resolve().parents[2]
PYPROJECT = BACKEND / "pyproject.toml"

#: pyproject.toml de la versión que no se podía instalar (commit b857822), resumido.
BROKEN_PYPROJECT = """
[project]
name = "mrlab"
version = "0.1.0"
readme = "../README.md"

[tool.setuptools.packages.find]
where = ["src"]
"""


def _strings(value: object, where: str) -> Iterator[tuple[str, str]]:
    """Todas las cadenas de un valor TOML con su ubicación (``tabla.clave[i]``)."""
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{where}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, f"{where}[{i}]")


def _path_entries(data: dict) -> list[tuple[str, str]]:
    """Rutas de archivos o carpetas que declaran los metadatos del paquete."""
    project = data.get("project", {})
    entries: list[tuple[str, str]] = []
    readme = project.get("readme")
    if isinstance(readme, str):
        entries.append(("project.readme", readme))
    elif isinstance(readme, dict) and "file" in readme:
        entries.append(("project.readme.file", readme["file"]))
    license_ = project.get("license")
    if isinstance(license_, dict) and "file" in license_:
        entries.append(("project.license.file", license_["file"]))
    entries += _strings(project.get("license-files", []), "project.license-files")
    # En [tool.setuptools] casi todo son rutas (where, package-dir, data-files, dynamic.*.file,
    # license-files…); los patrones de nombres de paquetes (include/exclude/namespaces) no.
    setuptools = data.get("tool", {}).get("setuptools", {})
    for where, value in _strings(setuptools, "tool.setuptools"):
        last = where.rsplit(".", 1)[-1].split("[", 1)[0]
        if last not in {"include", "exclude", "namespaces", "py-modules", "packages"}:
            entries.append((where, value))
    return entries


def _outside_backend(path: str) -> bool:
    for flavour in (PurePosixPath, PureWindowsPath):
        pure = flavour(path)
        if pure.is_absolute() or pure.drive or ".." in pure.parts:
            return True
    return not (BACKEND / path).resolve().is_relative_to(BACKEND)


def _offending(data: dict) -> list[str]:
    return [
        f"{where} = {value!r}" for where, value in _path_entries(data) if _outside_backend(value)
    ]


def test_pyproject_declares_no_path_outside_backend():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert _offending(data) == []


def test_readme_if_declared_is_a_file_inside_backend():
    """Hoy no se declara readme (el README del repositorio está fuera de backend/)."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    readme = project.get("readme")
    if readme is None:
        return
    path = readme if isinstance(readme, str) else readme.get("file")
    if path is not None:
        assert not _outside_backend(path), f"readme fuera de backend/: {path}"
        assert (BACKEND / path).is_file(), f"readme declarado pero inexistente: {path}"


def test_package_sources_are_found_inside_backend():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    where = data["tool"]["setuptools"]["packages"]["find"]["where"]
    assert where == ["src"]
    assert (BACKEND / "src" / "mrlab" / "__init__.py").is_file()


@pytest.mark.parametrize(
    "snippet",
    [
        BROKEN_PYPROJECT,
        '[project]\nname = "x"\nreadme = {file = "../README.md", content-type = "text/markdown"}\n',
        '[tool.setuptools.packages.find]\nwhere = ["../otro"]\n',
        '[tool.setuptools]\npackage-dir = {"" = "/abs/src"}\n',
        '[tool.setuptools]\ndata-files = {"share" = ["..\\\\datos\\\\x.csv"]}\n',
        '[tool.setuptools.dynamic]\nreadme = {file = ["../README.md"]}\n',
    ],
    ids=[
        "readme_original",
        "readme_tabla",
        "where",
        "package_dir_absoluto",
        "data_files",
        "dynamic",
    ],
)
def test_the_check_detects_paths_outside_backend(snippet):
    """La comprobación habría detectado el pyproject.toml que rompió la instalación."""
    assert _offending(tomllib.loads(snippet))
