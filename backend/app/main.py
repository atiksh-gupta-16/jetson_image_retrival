"""
FastAPI application factory — backend/app/main.py
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import router
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, setup_logging
from backend.app.repositories.vector_store import get_vector_store


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the vector store (and CLIP) before the first request."""
    cfg = get_settings()
    setup_logging(cfg.LOG_LEVEL)
    logger.info("Starting %s v%s", cfg.APP_NAME, cfg.APP_VERSION)
    # Pre-initialise the singleton so the first request isn't slow
    get_vector_store()
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title=cfg.APP_NAME,
        version=cfg.APP_VERSION,
        description=(
            "Imagify — semantic image search for Jetson alert footage.\n\n"
            "Push alert images from your Jetson device via POST /api/v1/ingest, "
            "then search them in natural language via POST /api/v1/search."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
