"""Configuración por variables de entorno (nunca en el código)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    model_dir: Path = Path("models/vision/v1.0.0")
    image_dir: Path = Path("appdata/images")
    db_path: Path = Path("appdata/line.db")
    demo_faults: bool = False

    vision_url: str = "http://localhost:8001"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    price_input_per_m: float = 0.40
    price_cached_per_m: float = 0.10
    price_output_per_m: float = 1.60
    daily_usd_cap: float = 1.50
    total_usd_cap: float = 3.50

    n8n_ticket_webhook_url: str = ""
    n8n_webhook_secret: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
