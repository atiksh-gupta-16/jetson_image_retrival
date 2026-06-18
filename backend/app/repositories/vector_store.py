"""
Repository layer for ChromaDB.
All vector store interactions are isolated here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.core.config import get_settings
from backend.app.core.exceptions import VectorStoreError
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class VectorStoreRepository:
    """
    Persistent Chroma collection storing CLIP embeddings of alert images.

    Metadata stored per vector:
        id, camera_id, timestamp (ISO string), alert_type, confidence,
        location_label, image_path, image_filename, extra (JSON string),
        indexed_at (ISO string)
    """

    def __init__(self) -> None:
        cfg = get_settings()
        cfg.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(cfg.CHROMA_PERSIST_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready | collection=%s | count=%d",
            cfg.CHROMA_COLLECTION_NAME,
            self._col.count(),
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        record_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        try:
            self._col.upsert(
                ids=[record_id],
                embeddings=[embedding],
                metadatas=[_sanitise(metadata)],
            )
        except Exception as exc:
            raise VectorStoreError(f"Upsert failed: {exc}") from exc

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return list of (metadata, cosine_similarity) sorted best-first."""
        count = self._col.count()
        if count == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            result = self._col.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"Query failed: {exc}") from exc

        metas = result["metadatas"][0]
        dists = result["distances"][0]
        # Chroma cosine distance ∈ [0,2]  →  similarity ∈ [0,1]
        return [(m, 1.0 - d / 2.0) for m, d in zip(metas, dists)]

    def count(self) -> int:
        return self._col.count()

    def list_cameras(self) -> List[str]:
        result = self._col.get(include=["metadatas"])
        cameras: set[str] = set()
        for m in (result.get("metadatas") or []):
            if m and m.get("camera_id"):
                cameras.add(m["camera_id"])
        return sorted(cameras)

    def delete(self, record_id: str) -> None:
        self._col.delete(ids=[record_id])

    def reset(self) -> None:
        cfg = get_settings()
        self._client.delete_collection(cfg.CHROMA_COLLECTION_NAME)
        self._col = self._client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection reset.")


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[VectorStoreRepository] = None


def get_vector_store() -> VectorStoreRepository:
    """FastAPI dependency — returns the module-level singleton."""
    global _instance
    if _instance is None:
        _instance = VectorStoreRepository()
    return _instance


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise(m: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce all metadata values to Chroma-compatible scalar types."""
    out: Dict[str, Any] = {}
    for k, v in m.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v)
    return out
