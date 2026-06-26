"""
Application configuration via Pydantic Settings.
All values can be overridden through environment variables or the .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

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
    IMAGE_STORE_DIR: Path = Path("./data/images")
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

    # ── LLM (intent extraction) ───────────────────────────────────────────────
    # Point at any GGUF model file on disk, e.g. Mistral-7B-Instruct.
    # The model is loaded once as a singleton; see services/intent.py.
    # LLM_MODEL_PATH: Path = Path("./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    # LLM_GPU_LAYERS: int = 0    # 0 = CPU-only; set to 32+ to offload layers to GPU
    # # ── Query Understanding ───────────────────────────────────────────────

    LLM_MODEL_NAME: str = "qwen2.5:1.5b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    LLM_TEMPERATURE: float = 0.0

    LLM_MAX_TOKENS: int = 256

    QUERY_PROMPT_PATH: Path = Path("backend/app/prompts/query_parser.txt")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()