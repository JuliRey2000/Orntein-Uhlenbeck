# Laboratorio de Reversión a la Media — tareas habituales (fase 1).
# En Windows, make requiere Git Bash o WSL; el README da los comandos equivalentes.
# Los objetivos `dev` (API + interfaz web) y `docker` llegan en las fases 2 y 3.

PYTHON ?= python

.PHONY: help install test test-fast demo fetch-example lint

help:
	@echo "make install        Instala el paquete y las dependencias de desarrollo (con el venv activo)"
	@echo "make test           Ejecuta todas las pruebas (incluidas las lentas)"
	@echo "make test-fast      Ejecuta las pruebas sin las marcadas como lentas"
	@echo "make demo           Demo con el par SINTÉTICO de examples/data (sin internet)"
	@echo "make fetch-example  Descarga KO y PEP de Yahoo Finance a examples/data/real/"
	@echo "make lint           Revisa el estilo con ruff"

install:
	$(PYTHON) -m pip install -e "backend[dev]"

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
