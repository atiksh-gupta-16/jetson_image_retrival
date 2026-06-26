"""
Query pipeline — hybrid RAG orchestrator.

This is the single public entry point for the Streamlit UI (or any future
front-end).  It owns the end-to-end flow:

    1. Extract structured intent from the user's query  (intent.py)
    2. Translate IntentFilter → Chroma `where` clause    (local)
    3. Run hybrid retrieval                              (retrieval.py)
    4. (future) Post-process / summarise results

By design, neither intent.py nor retrieval.py is aware of each other.
All composition happens here.

Public API
----------
run_query(query, vector_store, top_k, min_score) -> HybridQueryResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

from backend.app.core.config import get_settings
from backend.app.core.exceptions import QueryPipelineError
from backend.app.core.logging import get_logger
from backend.app.models.alert import SearchResult
from backend.app.models.intent import IntentFilter
from backend.app.repositories.vector_store import VectorStoreRepository
from backend.app.services.intent import extract_intent
from backend.app.services.retrieval import search_alerts

logger = get_logger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class HybridQueryResult:
    """
    Everything the UI needs to render a search response.

    Attributes
    ----------
    results     : Ranked list of matching alert images.
    intent      : The structured filters extracted by the LLM — useful for
                  displaying "Searching camera 2 for persons after 15:00" in
                  the UI without re-parsing.
    where_clause: The Chroma filter that was applied (for debugging/logging).
    """
    results: List[SearchResult]
    intent: IntentFilter
    where_clause: Optional[Dict[str, Any]]


# ── Public API ────────────────────────────────────────────────────────────────

def run_query(
    query: str,
    vector_store: VectorStoreRepository,
    top_k: Optional[int] = None,
    min_score: float = 0.0,
) -> HybridQueryResult:
    """
    Run the full hybrid RAG pipeline for one user query.

    Parameters
    ----------
    query        : Raw natural-language string from the user.
    vector_store : Injected Chroma repository.
    top_k        : Maximum results to return (falls back to config default).
    min_score    : Cosine similarity threshold passed to retrieval.

    Returns
    -------
    HybridQueryResult with ranked images, extracted intent, and Chroma filter.
    """
    cfg = get_settings()
    k = top_k or cfg.DEFAULT_TOP_K

    # ── Step 0: Check for basic greetings ─────────────────────────────────────
    clean_query = query.lower().strip().rstrip("?.!")
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "whats up", "yo"}
    is_greet = clean_query in greetings or any(clean_query.startswith(g + " ") for g in greetings)
    if is_greet:
        logger.info("Query %r identified as greeting. Skipping search/retrieval.", query)
        from backend.app.models.intent import IntentFilter
        return HybridQueryResult(
            results=[],
            intent=IntentFilter(semantic_query=""),
            where_clause=None
        )

    # ── Step 1: Intent extraction ─────────────────────────────────────────────
    intent = extract_intent(query)
    logger.info("Intent: %r", intent)

    # ── Step 2: Build Chroma where-clause ────────────────────────────────────
    where = _build_where_clause(intent)
    logger.info("Chroma where-clause: %s", where)

    # ── Step 3: Hybrid retrieval ──────────────────────────────────────────────
    try:
        results = search_alerts(
            query=intent.semantic_query,
            vector_store=vector_store,
            top_k=k,
            min_score=min_score,
            where=where,            # pre-built filter — retrieval.py passes it
        )                           # straight through to Chroma
    except Exception as exc:
        raise QueryPipelineError(f"Retrieval failed: {exc}") from exc

    logger.info(
        "Pipeline complete | query=%r | filters=%s | results=%d",
        query,
        where,
        len(results),
    )

    return HybridQueryResult(results=results, intent=intent, where_clause=where)


# ── Chroma filter builder ─────────────────────────────────────────────────────

def _build_where_clause(intent: IntentFilter) -> Optional[Dict[str, Any]]:
    """
    Translate an IntentFilter into a Chroma `where` dict.

    Chroma supports:
        {"field": {"$eq": value}}
        {"field": {"$gte": value}}
        {"$and": [clause, clause, ...]}

    Timestamp handling:
        Chroma stores timestamps as ISO-8601 strings.
        String comparison works correctly for ISO format, so we build
        ISO datetime strings from the date + time components and use
        $gte / $lte on the "timestamp" field.

    Returns None if no filters are active (= search entire collection).
    """
    if not intent.has_metadata_filters():
        return None

    conditions: List[Dict[str, Any]] = []

    # Exact-match filters
    if intent.camera_id:
        conditions.append({"camera_id": {"$eq": intent.camera_id}})

    # Both intent.label (e.g., 'person') and intent.alert_type (e.g., 'motion') map to Chroma's alert_type metadata field
    target_alert_type = intent.label or intent.alert_type
    if target_alert_type:
        conditions.append({"alert_type": {"$eq": target_alert_type}})

    if intent.min_confidence is not None:
        conditions.append({"confidence": {"$gte": intent.min_confidence}})

    # Timestamp range filters
    # We resolve date + time independently so either can be None.
    # If only a time is given (no date), we cannot safely build an ISO
    # timestamp without knowing the date, so we skip the filter and let
    # Python-side post-filtering handle it (see _post_filter_by_time).
    if intent.date:
        ts_after, ts_before = _build_timestamp_range(
            intent.date, intent.time_after, intent.time_before
        )
        if ts_after:
            conditions.append({"timestamp": {"$gte": ts_after}})
        if ts_before:
            conditions.append({"timestamp": {"$lte": ts_before}})

    if not conditions:
        return None

    return {"$and": conditions} if len(conditions) > 1 else conditions[0]


def _build_timestamp_range(
    date_str: str,
    time_after: Optional[str],
    time_before: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Build ISO timestamp strings for a date + optional time bounds.

    Returns (ts_after, ts_before).  Either can be None.
    """
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        logger.warning("Could not parse date: %r — skipping timestamp filter", date_str)
        return None, None

    ts_after: Optional[str] = None
    ts_before: Optional[str] = None

    if time_after:
        try:
            h, m = map(int, time_after.split(":"))
            ts_after = datetime.combine(d, time(h, m)).isoformat()
        except Exception:
            logger.warning("Could not parse time_after: %r", time_after)

    if time_before:
        try:
            h, m = map(int, time_before.split(":"))
            ts_before = datetime.combine(d, time(h, m)).isoformat()
        except Exception:
            logger.warning("Could not parse time_before: %r", time_before)

    # If only a date is given (no time bounds), filter the whole day
    if ts_after is None and ts_before is None:
        ts_after = datetime.combine(d, time.min).isoformat()
        ts_before = datetime.combine(d, time.max).isoformat()

    return ts_after, ts_before