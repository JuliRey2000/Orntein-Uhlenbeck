# Laboratorio de Reversión a la Media — tareas habituales.
# En Windows, make requiere Git Bash o WSL; el README da los comandos equivalentes en PowerShell.
# La interfaz web (servicio frontend y `make dev` con Vite) llega en la fase 3.

PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000
GIT_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)

.PHONY: help install test test-fast demo fetch-example lint format dev api openapi openapi-check \
	docker-config docker-up docker-down

help:
	@echo "make install        Instala el paquete con la API y las dependencias de desarrollo (con el venv activo)"
	@echo "make test           Ejecuta todas las pruebas (incluidas las lentas)"
	@echo "make test-fast      Ejecuta las pruebas sin las marcadas como lentas"
	@echo "make lint           Revisa el estilo con ruff (reglas y formato)"
	@echo "make format         Aplica el formato de ruff"
	@echo "make demo           Demo con el par SINTÉTICO de examples/data (sin internet)"
	@echo "make fetch-example  Descarga KO y PEP de Yahoo Finance a examples/data/real/"
	@echo "make dev            API con recarga automática en http://$(HOST):$(PORT) (documentación en /docs)"
	@echo "make api            API sin recarga (un solo proceso) en http://$(HOST):$(PORT)"
	@echo "make openapi        Exporta el esquema OpenAPI a backend/openapi.json"
	@echo "make openapi-check  Comprueba que backend/openapi.json está al día"
	@echo "make docker-config  Valida docker-compose.yml (no necesita el daemon de Docker)"
	@echo "make docker-up      Construye y levanta la API con Docker en http://localhost:8000"
	@echo "make docker-down    Detiene los contenedores"

install:
	$(PYTHON) -m pip install -e "backend[api,dev]"

test:
	cd backend && $(PYTHON) -m pytest

test-fast:
	cd backend && $(PYTHON) -m pytest -m "not slow"

demo:
	$(PYTHON) examples/demo_ko_pep.py

fetch-example:
	$(PYTHON) examples/fetch_ko_pep.py

lint:
	$(PYTHON) -m ruff check backend examples
	$(PYTHON) -m ruff format --check backend examples

format:
	$(PYTHON) -m ruff format backend examples

# Un solo proceso de uvicorn: la caché y los estudios viven en disco con locks por proceso
# (decisión F1). --reload vigila solo el código del paquete.
dev:
	$(PYTHON) -m uvicorn mrlab.api.main:app --reload --reload-dir backend/src --host $(HOST) --port $(PORT)

api:
	$(PYTHON) -m uvicorn mrlab.api.main:app --host $(HOST) --port $(PORT) --workers 1

openapi:
	$(PYTHON) backend/scripts/export_openapi.py

openapi-check:
	$(PYTHON) backend/scripts/export_openapi.py --check

docker-config:
	docker compose config --quiet && echo "docker-compose.yml es válido."

docker-up:
	GIT_COMMIT=$(GIT_COMMIT) docker compose up --build

docker-down:
	docker compose down
