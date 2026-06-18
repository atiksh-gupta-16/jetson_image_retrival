"""
Shared image utility helpers.
"""

from __future__ import annotations

import base64
from pathlib import Path


def file_to_b64(path: str | Path) -> str:
    """Return base64-encoded contents of an image file."""
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def b64_data_uri(b64: str, mime: str = "image/jpeg") -> str:
    """Wrap a base64 string in a data URI for HTML <img> tags."""
    return f"data:{mime};base64,{b64}"
