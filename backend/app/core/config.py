"""
Application configuration via Pydantic Settings.
All values can be overridden through environment variables or the .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Imagify"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Storage ───────────────────────────────────────────────────────────────
    # Where alert images are saved on disk after upload
    IMAGE_STORE_DIR: Path = Path("./data/images")
    # Where Chroma persists its vector index
    CHROMA_PERSIST_DIR: Path = Path("./data/chroma")
    CHROMA_COLLECTION_NAME: str = "imagify_alerts"

    # ── CLIP ──────────────────────────────────────────────────────────────────
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    EMBEDDING_DEVICE: str = "cpu"   # set to "cuda" on GPU machine

    # ── Upload limits ─────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: float = 20.0
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff",
    ]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 50


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
