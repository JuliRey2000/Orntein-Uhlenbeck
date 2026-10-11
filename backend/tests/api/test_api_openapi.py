"""/docs es parte del entregable: el esquema OpenAPI está completo y en español.

- Cada endpoint tiene resumen y descripción propios (no los que FastAPI deriva del nombre de la
  función), etiquetas en español y sus errores documentados con el formato ``ErrorResponse``.
- Ningún texto de FastAPI en inglés («Successful Response», «Validation Error»…), ningún esquema
  ``HTTPValidationError`` y ningún nombre automático ``Body_…``.
- Cada modelo y cada campo tienen descripción; los valores de enumeraciones y ejemplos no son
  palabras inglesas dirigidas al usuario.
- ``backend/openapi.json`` (versionado, para generar los tipos TypeScript) está al día.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute

from mrlab.api.main import API_PREFIX, TAGS, create_app, openapi_text
from mrlab.config import BACKEND_ROOT

OPENAPI_FILE = BACKEND_ROOT / "openapi.json"
EXPORT_SCRIPT = BACKEND_ROOT / "scripts" / "export_openapi.py"

#: Textos por defecto de FastAPI/Starlette (en inglés) que no deben aparecer en /docs.
FASTAPI_DEFAULTS = ("Successful Response", "Validation Error", "Unprocessable", "Not Found")

#: Palabras funcionales inglesas: su presencia en un texto delata una descripción en inglés.
ENGLISH_WORDS = frozenset(
    {
        "the", "and", "with", "for", "from", "is", "are", "this", "that", "these", "returns",
        "response", "successful", "request", "required", "should", "must", "not", "invalid",
        "found", "get", "create", "update", "delete", "upload", "search", "field", "input",
        "output", "default", "of", "to", "on", "an", "if", "when", "will", "be", "has", "have",
        "which", "list", "load", "run",
    }
)  # fmt: skip

#: Valores de enumeraciones o ejemplos que serían texto en inglés para el usuario.
ENGLISH_VALUES = frozenset(
    {
        "long", "short", "train", "training", "test", "testing", "validation", "index", "future",
        "futures", "equity", "stock", "crypto", "adjusted", "close", "synthetic", "historical",
        "pending", "running", "completed", "done", "failed", "cancelled", "canceled", "warning",
        "levels", "differences", "fixed", "rolling", "exact", "less", "greater", "than", "valid",
        "invalid", "insufficient", "unstable", "evidence", "doubtful", "forced", "success",
        "other", "file", "online", "price", "currency",
    }
)  # fmt: skip

HTTP_METHODS = {"get", "post", "put", "delete", "patch"}


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    from mrlab.api.main import app

    return app.openapi()


def _operations(spec: dict[str, Any]):
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if method in HTTP_METHODS:
                yield path, method, op


def _english_words(text: str) -> list[str]:
    """Palabras inglesas de un texto, sin contar identificadores ni valores citados.

    Se ignoran los fragmentos «…» y `…`, las palabras con «_», «.», «-», «/» o llaves (nombres
    de archivo, campos, rutas, «p-value») y las siglas en mayúsculas (GET, ZIP)."""
    text = re.sub(r"«[^»]*»|`[^`]*`", " ", text)
    found = []
    for raw in text.split():
        if re.search(r"[_./{}\-]", raw):
            continue
        word = re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúñÑüÜ]", "", raw)
        if not word or word.isupper():
            continue
        if word.lower() in ENGLISH_WORDS:
            found.append(word)
    return found


def _texts(node: Any, path: str = ""):
    """(ruta, texto) de cada summary/description del esquema."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}/{key}"
            if key in ("summary", "description") and isinstance(value, str):
                yield here, value
            yield from _texts(value, here)
    elif isinstance(node, list):
        for value in node:
            yield from _texts(value, path)


# --------------------------------------------------------------------------- endpoints


def _api_routes(routes) -> list[APIRoute]:
    """Rutas de la API, también las de routers incluidos (FastAPI 0.14x los incluye de forma
    diferida: ``app.routes`` contiene el router incluido, no sus rutas)."""
    found: list[APIRoute] = []
    for route in routes:
        if isinstance(route, APIRoute):
            found.append(route)
        sub = getattr(route, "original_router", None)
        if sub is not None:
            found += _api_routes(sub.routes)
    return found


def test_every_route_has_its_own_spanish_summary_and_description(spec):
    routes = _api_routes(create_app().routes)
    n_operations = sum(1 for _ in _operations(spec))
    assert sum(len(r.methods) for r in routes) == n_operations
    for route in routes:
        where = f"{sorted(route.methods)} {route.path}"
        # summary explícito: FastAPI, si falta, lo deriva del nombre (inglés) de la función.
        assert route.summary, f"{where}: falta summary"
        assert route.description and route.description.strip(), f"{where}: falta description"
        assert route.response_description != "Successful Response", where
        assert not _english_words(route.summary), f"{where}: {route.summary}"


def test_paths_and_tags_of_every_operation(spec):
    tag_names = {t["name"] for t in TAGS}
    for path, method, op in _operations(spec):
        where = f"{method.upper()} {path}"
        assert path.startswith(API_PREFIX + "/"), where
        assert op["tags"] and set(op["tags"]) <= tag_names, where
        assert op["summary"].strip() and op["description"].strip(), where


def test_tags_are_spanish_and_described(spec):
    names = [t["name"] for t in spec["tags"]]
    assert names == [t["name"] for t in TAGS]
    for tag in spec["tags"]:
        assert tag["description"].strip()
        assert not _english_words(tag["description"]), tag
    used = {tag for _, _, op in _operations(spec) for tag in op["tags"]}
    assert used == set(names)


def test_errors_are_documented_with_the_error_model(spec):
    for path, method, op in _operations(spec):
        where = f"{method.upper()} {path}"
        responses = op["responses"]
        assert "500" in responses, where
        has_input = bool(op.get("parameters")) or "requestBody" in op
        if has_input:
            # Con parámetros, el 422 propio (en español), nunca el HTTPValidationError de FastAPI.
            assert "422" in responses, where
        for code, response in responses.items():
            assert response["description"] not in FASTAPI_DEFAULTS, f"{where} {code}"
            if int(code) >= 400:
                schema = response["content"]["application/json"]["schema"]
                assert schema == {"$ref": "#/components/schemas/ErrorResponse"}, f"{where} {code}"
            else:
                assert code in ("200", "201", "202", "204"), f"{where} {code}"


def test_no_fastapi_english_defaults_anywhere(spec):
    text = json.dumps(spec, ensure_ascii=False)
    for default in FASTAPI_DEFAULTS:
        assert default not in text, default
    schemas = spec["components"]["schemas"]
    assert "HTTPValidationError" not in schemas
    assert "ValidationError" not in schemas
    assert not [name for name in schemas if name.startswith("Body_")]
    assert not [name for name in schemas if name.startswith("mrlab__")]  # nombres duplicados


def test_all_summaries_and_descriptions_are_spanish(spec):
    offenders = [(path, words) for path, text in _texts(spec) if (words := _english_words(text))]
    assert offenders == []


# --------------------------------------------------------------------------- modelos


def test_every_schema_and_field_is_described(spec):
    missing = []
    for name, schema in spec["components"]["schemas"].items():
        if not (schema.get("description") or "").strip():
            missing.append(name)
        for prop, info in (schema.get("properties") or {}).items():
            if not (info.get("description") or "").strip():
                missing.append(f"{name}.{prop}")
    assert missing == []


def _values(node: Any, inside_example: bool = False):
    """Valores de enum/const (y los textos de los ejemplos) de todo el esquema."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "enum":
                yield from (v for v in value if isinstance(v, str))
            elif key == "const" and isinstance(value, str):
                yield value
            elif key in ("example", "examples"):
                yield from _values(value, inside_example=True)
            else:
                yield from _values(value, inside_example)
    elif isinstance(node, list):
        for value in node:
            yield from _values(value, inside_example)
    elif inside_example and isinstance(node, str):
        yield node


def test_enum_and_example_values_are_not_english(spec):
    values = set(_values(spec))
    assert {"evidencia", "prueba", "SINTETICO", "largo_spread"} <= values
    offenders = sorted(
        v for v in values if any(t.lower() in ENGLISH_VALUES for t in re.split(r"[_\-\s]+", v))
    )
    assert offenders == []


def test_spanish_checks_detect_english():
    """Autoverificación de las heurísticas: un texto en inglés sí se detecta."""
    assert _english_words("Returns the list of jobs") == ["Returns", "the", "list", "of"]
    assert _english_words("Ejecuta el run_config.json con GET /api/jobs y el p-value") == []
    assert any(t in ENGLISH_VALUES for t in "long_spread".split("_"))


# --------------------------------------------------------------------------- archivo versionado


def _load_export_script():
    module_spec = importlib.util.spec_from_file_location("export_openapi", EXPORT_SCRIPT)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_versioned_openapi_file_is_up_to_date():
    assert OPENAPI_FILE.is_file(), "falta backend/openapi.json: ejecuta `make openapi`"
    current = OPENAPI_FILE.read_text(encoding="utf-8")
    assert current == openapi_text(), (
        "backend/openapi.json no está al día con el código de la API: ejecuta `make openapi` "
        "(python backend/scripts/export_openapi.py) y versiona el resultado."
    )
    assert json.loads(current)["info"]["title"].startswith("Laboratorio de Reversión a la Media")


def test_openapi_text_is_stable_and_independent_of_settings(tmp_path, lab_settings, registry):
    other = create_app(lab_settings, registry)
    assert openapi_text(other) == openapi_text()
    assert openapi_text() == openapi_text()
    assert not lab_settings.data_dir.exists()


def test_export_script_writes_and_checks(tmp_path: Path, capsys):
    script = _load_export_script()
    out = tmp_path / "openapi.json"
    assert script.main(["--check", "--output", str(out)]) == 1  # no existe
    assert "no está al día" in capsys.readouterr().err
    assert script.main(["--output", str(out)]) == 0
    assert out.read_bytes() == openapi_text().encode("utf-8")
    assert b"\r\n" not in out.read_bytes()
    assert script.main(["--check", "--output", str(out)]) == 0
    out.write_text("{}\n", encoding="utf-8")
    assert script.main(["--check", "--output", str(out)]) == 1
