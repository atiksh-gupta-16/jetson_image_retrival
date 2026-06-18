"""
Internal domain models — the canonical data shapes used across all services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AlertRecord(BaseModel):
    """
    One alert event pushed from a Jetson device.
    The image is saved to disk; this model holds metadata + the disk path.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))

    # Image on disk
    image_path: str              # absolute path to the saved image file
    image_filename: str          # original filename as stored

    # Core Jetson metadata
    camera_id: str
    timestamp: datetime

    # Optional enrichment (can be added later by the device or by the server)
    alert_type: Optional[str] = None        # e.g. "motion", "person", "vehicle"
    confidence: Optional[float] = None      # detection confidence 0–1
    location_label: Optional[str] = None    # human-readable camera location

    # Arbitrary extra fields from the Jetson payload
    extra: Dict[str, Any] = Field(default_factory=dict)

    indexed_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """One item returned from a semantic search."""
    record: AlertRecord
    score: float        # cosine similarity in [0, 1]
    rank: int
    image_b64: str      # base64-encoded image for direct rendering
