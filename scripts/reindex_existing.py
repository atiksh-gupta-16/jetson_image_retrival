#!/usr/bin/env python3
"""
scripts/reindex_existing.py

Scans the local data/images directory, reconstructs/mocks metadata
for any image files present under camera folders (0, 1, 2, 3, 4, etc.),
and indexes them directly into the Chroma DB.
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Ensure the root directory is on the path so we can import backend modules
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from backend.app.core.config import get_settings
    from backend.app.repositories.vector_store import get_vector_store
    from backend.app.models.alert import AlertRecord
    from backend.app.services.rag import index_record
except ImportError as exc:
    print(f"Error importing backend modules: {exc}")
    print("Please run this script from the project root directory.")
    sys.exit(1)

# Configuration mapping camera IDs to descriptive location labels
CAMERA_LOCATIONS = {
    "0": "Front Gate",
    "1": "Backyard Patio",
    "2": "Driveway",
    "3": "Garage Interior",
    "4": "Side Alley",
    "cam-temp": "Temporary Demo Gate",
}

ALERT_TYPES = ["person", "vehicle", "animal", "motion"]


def _load_sidecar_metadata(camera_dir: Path) -> dict:
    metadata_path = camera_dir / "metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        data = json.loads(metadata_path.read_text())
    except Exception as exc:
        print(f"  Warning: could not read {metadata_path.name} in {camera_dir.name}: {exc}")
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def _parse_timestamp(raw_value: object, fallback_path: Path) -> datetime:
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass

    mtime = fallback_path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


def _build_record_metadata(camera_id: str, location: str, sidecar: dict, img_path: Path) -> tuple[datetime, str, float, dict, str]:
    timestamp = _parse_timestamp(
        sidecar.get("timestamp") or sidecar.get("event_time") or sidecar.get("captured_at"),
        img_path,
    )

    alert_type = str(sidecar.get("alert_type") or "motion")

    confidence_raw = sidecar.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 0.5
    except (TypeError, ValueError):
        confidence = 0.5

    extra = sidecar.get("extra") if isinstance(sidecar.get("extra"), dict) else {}
    extra = {
        **extra,
        "source": sidecar.get("source", "reindex_script"),
        "reindexed": True,
        "camera_folder": camera_id,
    }

    if sidecar.get("location_label"):
        location = str(sidecar["location_label"])

    image_path = str(img_path.resolve())

    return timestamp, alert_type, confidence, extra, image_path

def main():
    cfg = get_settings()

    if cfg.CHROMA_PERSIST_DIR.exists():
        print(f"Deleting existing Chroma DB at: {cfg.CHROMA_PERSIST_DIR.resolve()}")
        shutil.rmtree(cfg.CHROMA_PERSIST_DIR)

    vector_store = get_vector_store()
    print("Resetting Vector Store...")
    vector_store.reset()
    
    print(f"Chroma DB Collection Name: {cfg.CHROMA_COLLECTION_NAME}")
    print(f"Current document count in Chroma: {vector_store.count()}")
    print(f"Scanning images store directory: {cfg.IMAGE_STORE_DIR.resolve()}")
    
    if not cfg.IMAGE_STORE_DIR.exists():
        print(f"Image store directory does not exist: {cfg.IMAGE_STORE_DIR}")
        sys.exit(1)

    # Find all camera directories and process any sidecar metadata they provide.
    cam_dirs = [d for d in cfg.IMAGE_STORE_DIR.iterdir() if d.is_dir()]
    
    if not cam_dirs:
        print("No camera directories found in the image store.")
        return

    print(f"Found camera folders: {', '.join(d.name for d in cam_dirs)}")
    
    total_indexed = 0
    
    # We can distribute timestamps slightly if the file mtimes are all identical
    # so they feel like a real timeline.
    base_time = datetime.now(timezone.utc) - timedelta(days=7)
    
    for cam_dir in cam_dirs:
        camera_id = cam_dir.name
        location = CAMERA_LOCATIONS.get(camera_id, f"Camera {camera_id}")
        sidecar = _load_sidecar_metadata(cam_dir)
        
        # Get all image files in this directory
        image_files = sorted(
            [f for f in cam_dir.iterdir() if f.is_file() and f.suffix.lower() in cfg.ALLOWED_IMAGE_EXTENSIONS]
        )
        
        if not image_files:
            print(f"No images in camera {camera_id} folder.")
            continue
            
        print(f"Processing camera {camera_id} ({location}) - {len(image_files)} images...")
        
        for idx, img_path in enumerate(image_files):
            import uuid

            record_id = img_path.stem
            try:
                uuid.UUID(record_id)
            except ValueError:
                record_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{camera_id}/{img_path.name}"))

            timestamp, alert_type, confidence, extra, image_path = _build_record_metadata(
                camera_id=camera_id,
                location=location,
                sidecar=sidecar,
                img_path=img_path,
            )

            record = AlertRecord(
                id=record_id,
                image_path=image_path,
                image_filename=img_path.name,
                camera_id=camera_id,
                timestamp=timestamp,
                alert_type=alert_type,
                confidence=confidence,
                location_label=location,
                extra=extra,
                indexed_at=datetime.now(timezone.utc)
            )
            
            try:
                index_record(record, vector_store)
                total_indexed += 1
                print(f"  Indexed: {img_path.name} -> {alert_type} (conf={confidence})")
            except Exception as e:
                print(f"  Failed to index {img_path.name}: {e}")
                
    print(f"\nCompleted re-indexing. Added {total_indexed} records to Chroma DB.")
    print(f"New total document count in Chroma: {vector_store.count()}")

if __name__ == "__main__":
    main()
