#!/usr/bin/env python3
"""
scripts/send_alert.py

Simulate a Jetson device pushing an alert image to the Imagify API.
Can also be imported and called from Jetson alert pipeline code.

Usage
-----
    python scripts/send_alert.py \
        --image /path/to/alert.jpg \
        --camera-id "cam-01" \
        --alert-type "person" \
        --confidence 0.92 \
        --location "Front Gate" \
        --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


def send_alert(
    image_path: str,
    camera_id: str,
    api_url: str = "http://localhost:8000",
    alert_type: str | None = None,
    confidence: float | None = None,
    location_label: str | None = None,
    extra: dict | None = None,
    timestamp: datetime | None = None,
) -> dict:
    """
    POST one alert image to the Imagify ingest endpoint.

    This is the function your Jetson pipeline code should call.

    Returns the parsed JSON response from the server.
    """
    ts = timestamp or datetime.now(tz=timezone.utc)
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    form_data: dict = {
        "camera_id": camera_id,
        "timestamp": ts.isoformat(),
    }
    if alert_type:
        form_data["alert_type"] = alert_type
    if confidence is not None:
        form_data["confidence"] = str(confidence)
    if location_label:
        form_data["location_label"] = location_label
    if extra:
        form_data["extra_json"] = json.dumps(extra)

    with open(path, "rb") as f:
        files = {"image": (path.name, f, "image/jpeg")}
        response = requests.post(
            f"{api_url}/api/v1/ingest",
            data=form_data,
            files=files,
            timeout=30,
        )

    response.raise_for_status()
    result = response.json()
    print(f"✅  Indexed alert {result['id']} from {camera_id}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a Jetson alert image to Imagify")
    parser.add_argument("--image", required=True, help="Path to the alert image")
    parser.add_argument("--camera-id", required=True, help="Camera / device identifier")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Imagify API base URL")
    parser.add_argument("--alert-type", default=None, help="Alert type label")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence 0-1")
    parser.add_argument("--location", default=None, help="Camera location label")
    parser.add_argument("--extra", default=None, help="Extra metadata as JSON string")

    args = parser.parse_args()

    extra = None
    if args.extra:
        try:
            extra = json.loads(args.extra)
        except json.JSONDecodeError as exc:
            print(f"❌  --extra is not valid JSON: {exc}")
            sys.exit(1)

    try:
        send_alert(
            image_path=args.image,
            camera_id=args.camera_id,
            api_url=args.api_url,
            alert_type=args.alert_type,
            confidence=args.confidence,
            location_label=args.location,
            extra=extra,
        )
    except Exception as exc:
        print(f"❌  {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
