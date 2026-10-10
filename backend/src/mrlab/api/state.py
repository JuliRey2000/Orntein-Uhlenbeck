"""Estado de la aplicación (inyectable en pruebas): configuración, datos, estudios y trabajos."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from mrlab.api.jobs import JobManager
from mrlab.config import Settings
from mrlab.data.cache import SnapshotCache
from mrlab.data.registry import ProviderRegistry
from mrlab.data.service import DataService
from mrlab.studies import StudyStore


@dataclass
class AppState:
    settings: Settings
    registry: ProviderRegistry
    cache: SnapshotCache
    data: DataService
    studies: StudyStore
    jobs: JobManager


def build_state(settings: Settings, registry: ProviderRegistry | None = None) -> AppState:
    """Construye el estado. ``registry`` permite inyectar proveedores falsos en pruebas."""
    registry = registry or ProviderRegistry.from_toml(settings.providers_file)
    cache = SnapshotCache(settings.cache_dir)
    return AppState(
        settings=settings,
        registry=registry,
        cache=cache,
        data=DataService(registry, cache),
        studies=StudyStore(settings.studies_dir),
        jobs=JobManager(max_workers=settings.job_workers),
    )


def get_state(request: Request) -> AppState:
    return request.app.state.lab
