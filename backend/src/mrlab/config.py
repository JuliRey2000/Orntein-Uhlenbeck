"""Configuración leída del entorno (.env). Nada de esto llega al navegador."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"), env_prefix="MRLAB_", extra="ignore"
    )

    data_dir: Path = REPO_ROOT / "data"
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    studies_dir: Path = REPO_ROOT / "data" / "studies"
    uploads_dir: Path = REPO_ROOT / "data" / "uploads"
    providers_file: Path = BACKEND_ROOT / "providers.toml"
    yahoo_max_retries: int = 4
    yahoo_backoff_base_seconds: float = 1.0
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
