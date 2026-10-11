"""Archivos de despliegue (fase 2): Dockerfile, docker-compose.yml, .env.example y Makefile.

Aquí no hay Docker: se comprueba su contenido de forma estática para que no se desalineen de la
configuración (``mrlab.config.Settings``) ni de lo que exige la especificación (usuario sin
privilegios, GIT_COMMIT, HEALTHCHECK a /api/health, volumen ./data, un solo proceso…).
"""

from __future__ import annotations

import re

import pytest

from mrlab.config import BACKEND_ROOT, REPO_ROOT, Settings

DOCKERFILE = BACKEND_ROOT / "Dockerfile"
DOCKERIGNORE = BACKEND_ROOT / ".dockerignore"
COMPOSE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
MAKEFILE = REPO_ROOT / "Makefile"

SETTINGS_VARS = {f"MRLAB_{name.upper()}" for name in Settings.model_fields}
DATA_VARS = {
    "MRLAB_DATA_DIR": "/app/data",
    "MRLAB_CACHE_DIR": "/app/data/cache",
    "MRLAB_STUDIES_DIR": "/app/data/studies",
    "MRLAB_UPLOADS_DIR": "/app/data/uploads",
    "MRLAB_PROVIDERS_FILE": "/app/providers.toml",
}


def _instructions(text: str) -> list[tuple[str, str]]:
    """(INSTRUCCIÓN, argumentos) del Dockerfile, uniendo las líneas continuadas con «\\»."""
    joined = re.sub(r"\\\n", " ", text)
    out = []
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            word, _, rest = line.partition(" ")
            out.append((word.upper(), rest.strip()))
    return out


def _env_assignments(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for word, rest in _instructions(text):
        if word == "ENV":
            for key, value in re.findall(r"(\w+)=(\S*)", rest):
                env[key] = value
    return env


def test_dockerfile_base_user_and_command():
    text = DOCKERFILE.read_text(encoding="utf-8")
    steps = _instructions(text)
    assert steps[0] == ("FROM", "python:3.12-slim")
    words = [w for w, _ in steps]
    assert any(w == "ARG" and r.startswith("GIT_COMMIT") for w, r in steps)
    assert _env_assignments(text)["GIT_COMMIT"] == "${GIT_COMMIT}"
    assert any(w == "RUN" and 'pip install ".[api]"' in r for w, r in steps)
    # Usuario sin privilegios, fijado después de instalar y antes de arrancar.
    user = [r for w, r in steps if w == "USER"]
    assert user and user[-1] not in ("root", "0")
    assert words.index("USER") > max(i for i, (w, r) in enumerate(steps) if "pip install" in r)
    health = [r for w, r in steps if w == "HEALTHCHECK"]
    assert health and "/api/health" in health[0]
    assert any(w == "EXPOSE" and r == "8000" for w, r in steps)
    cmd = [r for w, r in steps if w == "CMD"][-1]
    assert '"uvicorn", "mrlab.api.main:app"' in cmd
    assert '"--host", "0.0.0.0"' in cmd and '"--port", "8000"' in cmd
    assert '"--workers", "1"' in cmd  # un solo proceso (decisión F1)


def test_dockerfile_env_matches_settings_and_copies_existing_files():
    text = DOCKERFILE.read_text(encoding="utf-8")
    env = _env_assignments(text)
    mrlab_vars = {k for k in env if k.startswith("MRLAB_")}
    assert mrlab_vars <= SETTINGS_VARS
    for key, value in DATA_VARS.items():
        assert env[key] == value, key
    ignored = {
        line.strip().rstrip("/")
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for word, rest in _instructions(text):
        if word == "COPY":
            source = rest.split()[0]
            assert (BACKEND_ROOT / source).exists(), source
            assert source not in ignored, source


def test_compose_backend_service():
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"backend"}  # el frontend llega en la fase 3 (comentado)
    backend = services["backend"]
    assert backend["build"]["context"] == "./backend"
    assert "GIT_COMMIT" in backend["build"]["args"]
    assert "./data:/app/data" in backend["volumes"]
    assert any(str(p).endswith(":8000") for p in backend["ports"])
    assert backend["env_file"] == [{"path": ".env", "required": False}]
    env = backend["environment"]
    assert set(env) <= SETTINGS_VARS
    for key, value in DATA_VARS.items():
        assert env[key] == value, key
    assert {"MRLAB_JOB_WORKERS", "MRLAB_MAX_UPLOAD_MB", "MRLAB_CORS_ORIGINS"} <= set(env)
    assert "--workers" in backend["command"] and "1" in backend["command"]
    text = COMPOSE.read_text(encoding="utf-8")
    assert "# frontend:" in text  # dónde entra el servicio de la fase 3


def test_env_example_lists_only_real_settings():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    mentioned = set(re.findall(r"^#?\s*(MRLAB_\w+)=", text, flags=re.M))
    assert {"MRLAB_JOB_WORKERS", "MRLAB_MAX_UPLOAD_MB"} <= mentioned
    # Las de Settings más las que solo usa docker-compose.yml.
    assert mentioned <= SETTINGS_VARS | {"MRLAB_BIND", "MRLAB_PORT", "MRLAB_UID", "MRLAB_GID"}


def test_makefile_targets():
    text = MAKEFILE.read_text(encoding="utf-8")
    targets = set(re.findall(r"^([\w-]+):", text, flags=re.M))
    assert {"dev", "api", "openapi", "docker-up", "docker-config", "test", "demo"} <= targets
    dev = re.search(r"^dev:.*?(?=^\S|\Z)", text, flags=re.M | re.S)
    assert dev is not None
    assert "--reload" in dev.group(0)
    assert "--host $(HOST) --port $(PORT)" in dev.group(0)
    assert re.search(r"^HOST \?= 127\.0\.0\.1$", text, flags=re.M)
    assert re.search(r"^PORT \?= 8000$", text, flags=re.M)
