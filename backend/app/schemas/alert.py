"""
FastAPI request / response schemas (HTTP boundary layer).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Ingest response (request uses multipart/form-data so no request schema) ──

class IngestResponse(BaseModel):
    id: str
    camera_id: str
    timestamp: datetime
    alert_type: Optional[str]
    confidence: Optional[float]
    location_label: Optional[str]
    image_filename: str
    message: str = "Alert indexed successfully"


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["person near the gate at night"])
    top_k: int = Field(10, ge=1, le=50)
    camera_id: Optional[str] = Field(None, description="Filter by camera ID")
    alert_type: Optional[str] = Field(None, description="Filter by alert type")
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class AlertResultItem(BaseModel):
    rank: int
    score: float
    id: str
    camera_id: str
    timestamp: datetime
    alert_type: Optional[str]
    confidence: Optional[float]
    location_label: Optional[str]
    image_filename: str
    image_b64: str          # base64-encoded image, ready for <img src="data:...">
    extra: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[AlertResultItem]


# ── Collection stats ──────────────────────────────────────────────────────────

class CollectionStatsResponse(BaseModel):
    total_alerts: int
    cameras: List[str]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    total_indexed: int
    clip_model: str
