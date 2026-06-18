"""
Retrieval service.

Takes a natural-language query, embeds it with CLIP, queries Chroma,
and returns SearchResult objects that include the base64-encoded image
so the Streamlit UI can render them directly.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from backend.app.core.config import get_settings
from backend.app.core.exceptions import RetrievalError
from backend.app.core.logging import get_logger
from backend.app.models.alert import AlertRecord, SearchResult
from backend.app.repositories.vector_store import VectorStoreRepository
from backend.app.services.rag import embed_text

logger = get_logger(__name__)


def search_alerts(
    query: str,
    vector_store: VectorStoreRepository,
    top_k: Optional[int] = None,
    camera_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    min_score: float = 0.0,
) -> List[SearchResult]:
    """
    Semantic search over indexed alert images.

    Parameters
    ----------
    query:        Natural-language description, e.g. "person in red jacket".
    vector_store: Injected repository.
    top_k:        Max results (falls back to DEFAULT_TOP_K from config).
    camera_id:    Optional filter — only return alerts from this camera.
    alert_type:   Optional filter — only return alerts of this type.
    min_score:    Cosine similarity threshold in [0, 1].

    Returns
    -------
    List of SearchResult sorted by descending similarity, each with a
    base64-encoded image ready for rendering.
    """
    cfg = get_settings()
    k = top_k or cfg.DEFAULT_TOP_K

    if not query.strip():
        raise RetrievalError("Query must not be empty.")

    logger.info("Search | query=%r | top_k=%d | camera=%s", query, k, camera_id or "*")

    # 1. Embed the query
    query_vector = embed_text(query)

    # 2. Build optional Chroma where-filter
    # Chroma only supports a single field in $and / $or when using the default
    # hnsw index, so we apply secondary filters in Python.
    where = None
    if camera_id:
        where = {"camera_id": camera_id}

    # 3. Fetch from vector store (fetch extra so we can post-filter)
    fetch_k = min(k * 4, cfg.MAX_TOP_K * 4)
    raw = vector_store.query(query_embedding=query_vector, top_k=fetch_k, where=where)

    # 4. Build results
    results: List[SearchResult] = []
    rank = 1

    for meta, score in raw:
        if score < min_score:
            continue

        # Secondary filter: alert_type
        if alert_type and meta.get("alert_type", "") != alert_type:
            continue

        record = _meta_to_record(meta)
        image_b64 = _load_image_b64(record.image_path)

        results.append(
            SearchResult(record=record, score=round(score, 4), rank=rank, image_b64=image_b64)
        )
        rank += 1
        if rank > k:
            break

    logger.info("Returned %d results for query %r", len(results), query)
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meta_to_record(meta: dict) -> AlertRecord:
    extra_raw = meta.get("extra", "{}")
    try:
        extra = json.loads(extra_raw) if isinstance(extra_raw, str) else extra_raw
    except Exception:
        extra = {}

    confidence_raw = meta.get("confidence", -1.0)
    confidence = float(confidence_raw) if float(confidence_raw) >= 0 else None

    return AlertRecord(
        id=meta.get("id", ""),
        image_path=meta.get("image_path", ""),
        image_filename=meta.get("image_filename", ""),
        camera_id=meta.get("camera_id", ""),
        timestamp=datetime.fromisoformat(meta["timestamp"]),
        alert_type=meta.get("alert_type") or None,
        confidence=confidence,
        location_label=meta.get("location_label") or None,
        extra=extra,
        indexed_at=datetime.fromisoformat(meta["indexed_at"]),
    )


def _load_image_b64(image_path: str) -> str:
    """Read image bytes from disk and return base64-encoded string."""
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image file not found: %s", image_path)
        return ""
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception as exc:
        logger.error("Cannot read image %s: %s", image_path, exc)
        return ""
