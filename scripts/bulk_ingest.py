#!/usr/bin/env python3
"""
scripts/bulk_ingest.py

Walk an alerts/ directory with the structure:
    alerts/
        cam00/  image1.jpg  image2.jpg ...
        cam01/  ...
        cam17/  ...

and push every image to the Imagify API, using the folder name as camera_id.

Usage
-----
    # Ingest everything
    python scripts/bulk_ingest.py --alerts-dir ./alerts

    # Only specific cameras
    python scripts/bulk_ingest.py --alerts-dir ./alerts --cameras cam00 cam01

    # Dry run (print what would be sent, don't actually send)
    python scripts/bulk_ingest.py --alerts-dir ./alerts --dry-run

    # Custom API URL
    python scripts/bulk_ingest.py --alerts-dir ./alerts --api-url http://192.168.1.50:8000
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
DEFAULT_API_URL = "http://localhost:8000"


# ── Core sender ───────────────────────────────────────────────────────────────

def send_image(
    image_path: Path,
    camera_id: str,
    api_url: str,
    dry_run: bool = False,
) -> bool:
    """
    POST one image to /api/v1/ingest.
    Returns True on success, False on failure.
    """
    # Use file modification time as the alert timestamp
    mtime = image_path.stat().st_mtime
    timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    if dry_run:
        print(f"  [DRY RUN] would send: {image_path}  camera={camera_id}  ts={timestamp}")
        return True

    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{api_url}/api/v1/ingest",
                data={
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                },
                files={"image": (image_path.name, f, "image/jpeg")},
                timeout=30,
            )
        response.raise_for_status()
        result = response.json()
        print(f"  ✅  {image_path.name}  →  id={result['id']}")
        return True

    except requests.HTTPError as exc:
        print(f"  ❌  {image_path.name}  →  HTTP {exc.response.status_code}: {exc.response.text[:120]}")
        return False
    except Exception as exc:
        print(f"  ❌  {image_path.name}  →  {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-ingest an alerts/ directory into Imagify"
    )
    parser.add_argument(
        "--alerts-dir",
        default="./alerts",
        help="Root alerts directory (default: ./alerts)",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Only ingest these camera folders, e.g. --cameras cam00 cam03. "
             "Omit to ingest all cameras found.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Imagify API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be sent without actually sending anything",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between requests (default: 0). "
             "Use e.g. 0.1 to be gentle on a slow server.",
    )
    args = parser.parse_args()

    alerts_root = Path(args.alerts_dir)
    if not alerts_root.exists():
        print(f"❌  alerts directory not found: {alerts_root.resolve()}")
        sys.exit(1)

    # Discover camera folders
    cam_dirs = sorted(
        [d for d in alerts_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not cam_dirs:
        print(f"❌  No sub-directories found in {alerts_root.resolve()}")
        sys.exit(1)

    # Apply --cameras filter
    if args.cameras:
        requested = set(args.cameras)
        cam_dirs = [d for d in cam_dirs if d.name in requested]
        missing = requested - {d.name for d in cam_dirs}
        if missing:
            print(f"⚠️   Cameras not found in {alerts_root}: {', '.join(sorted(missing))}")

    if not cam_dirs:
        print("❌  No matching camera directories to process.")
        sys.exit(1)

    # Count total images up front
    all_images: list[tuple[Path, str]] = []   # (image_path, camera_id)
    for cam_dir in cam_dirs:
        images = sorted(
            [f for f in cam_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
        )
        for img in images:
            all_images.append((img, cam_dir.name))

    if not all_images:
        print("❌  No images found in any camera directory.")
        sys.exit(1)

    # Summary before starting
    cam_summary = ", ".join(d.name for d in cam_dirs)
    print(f"\n{'='*60}")
    print(f"  Imagify Bulk Ingest")
    print(f"  Alerts dir : {alerts_root.resolve()}")
    print(f"  Cameras    : {cam_summary}")
    print(f"  Total imgs : {len(all_images)}")
    print(f"  API        : {args.api_url}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{'='*60}\n")

    # Process
    succeeded = 0
    failed = 0
    current_cam = None

    for image_path, camera_id in all_images:
        if camera_id != current_cam:
            current_cam = camera_id
            cam_count = sum(1 for _, c in all_images if c == camera_id)
            print(f"\n📷  {camera_id}  ({cam_count} images)")

        ok = send_image(image_path, camera_id, args.api_url, dry_run=args.dry_run)
        if ok:
            succeeded += 1
        else:
            failed += 1

        if args.delay > 0:
            time.sleep(args.delay)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Done.  ✅ {succeeded} indexed   ❌ {failed} failed   total {len(all_images)}")
    print(f"{'='*60}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
