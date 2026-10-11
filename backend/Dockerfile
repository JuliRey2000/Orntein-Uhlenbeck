# Imagen del backend (API FastAPI) del Laboratorio de Reversión a la Media.
#
# Contexto de construcción: la carpeta backend/ (lo fija docker-compose.yml en la raíz):
#   docker compose up --build        → API en http://localhost:8000 (documentación en /docs)
#
# Un solo proceso de uvicorn: la caché de datos y los estudios viven en disco (volumen
# ./data → /app/data) y sus locks son por proceso (decisión F1 de docs/DECISIONS.md).

FROM python:3.12-slim

# Versión del código que se registra en cada run_config.json (no hay carpeta .git en la
# imagen). docker-compose.yml la pasa desde la variable GIT_COMMIT (make docker-up la calcula).
ARG GIT_COMMIT=""
# Usuario sin privilegios; su uid/gid debería coincidir con el dueño de ./data en el anfitrión
# (en Linux), para que el volumen se pueda escribir.
ARG MRLAB_UID=1000
ARG MRLAB_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GIT_COMMIT=${GIT_COMMIT} \
    MRLAB_DATA_DIR=/app/data \
    MRLAB_CACHE_DIR=/app/data/cache \
    MRLAB_STUDIES_DIR=/app/data/studies \
    MRLAB_UPLOADS_DIR=/app/data/uploads \
    MRLAB_PROVIDERS_FILE=/app/providers.toml

WORKDIR /app

RUN groupadd --system --gid "${MRLAB_GID}" mrlab \
    && useradd --system --uid "${MRLAB_UID}" --gid mrlab --home-dir /app \
       --shell /usr/sbin/nologin mrlab

# Paquete y dependencias de la API (sin las de desarrollo).
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[api]" && rm -rf ./src ./build

# Registro de proveedores (fuera del paquete instalado; ver MRLAB_PROVIDERS_FILE).
COPY providers.toml ./providers.toml

RUN mkdir -p /app/data/cache /app/data/studies /app/data/uploads \
    && chown -R mrlab:mrlab /app/data

USER mrlab

EXPOSE 8000

# Sin curl en la imagen slim: la comprobación usa Python (y no pasa por ningún proxy).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request as u; o = u.build_opener(u.ProxyHandler({})); sys.exit(0 if o.open('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "mrlab.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
