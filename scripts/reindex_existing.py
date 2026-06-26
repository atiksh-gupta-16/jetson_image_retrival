#!/usr/bin/env python3
"""
scripts/reindex_existing.py

Scans the local data/images directory, reconstructs/mocks metadata
for any image files present under camera folders (0, 1, 2, 3, 4, etc.),
and indexes them directly into the Chroma DB.
"""

import sys
import random
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
}

ALERT_TYPES = ["person", "vehicle", "animal", "motion"]

def main():
    cfg = get_settings()
    vector_store = get_vector_store()
    print("Resetting Vector Store...")
    vector_store.reset()
    
    print(f"Chroma DB Collection Name: {cfg.CHROMA_COLLECTION_NAME}")
    print(f"Current document count in Chroma: {vector_store.count()}")
    print(f"Scanning images store directory: {cfg.IMAGE_STORE_DIR.resolve()}")
    
    if not cfg.IMAGE_STORE_DIR.exists():
        print(f"Image store directory does not exist: {cfg.IMAGE_STORE_DIR}")
        sys.exit(1)

    # Find all camera directories (ignoring cam-temp if desired, but we can process any numeric/valid dir)
    cam_dirs = [d for d in cfg.IMAGE_STORE_DIR.iterdir() if d.is_dir() and d.name != "cam-temp"]
    
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
        
        # Get all image files in this directory
        image_files = sorted(
            [f for f in cam_dir.iterdir() if f.is_file() and f.suffix.lower() in cfg.ALLOWED_IMAGE_EXTENSIONS]
        )
        
        if not image_files:
            print(f"No images in camera {camera_id} folder.")
            continue
            
        print(f"Processing camera {camera_id} ({location}) - {len(image_files)} images...")
        
        for idx, img_path in enumerate(image_files):
            # Infer or extract a stable ID
            # The filename is usually <record_id>.<ext>
            record_id = img_path.stem
            
            # Check if record_id is a valid UUID, otherwise generate a new one
            try:
                # Test if it's a valid uuid format
                import uuid
                uuid.UUID(record_id)
            except ValueError:
                # If it's not a UUID, generate one based on file name or random
                record_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, img_path.name))
            
            # Get timestamp from file modification time
            mtime = img_path.stat().st_mtime
            timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
            
            # Mock rich metadata
            # Seeds are deterministic based on filename to keep values stable across runs
            random.seed(img_path.name)
            alert_type = random.choice(ALERT_TYPES)
            confidence = round(random.uniform(0.65, 0.99), 2)
            
            # Make path relative, e.g. "data/images/0/filename.jpg"
            relative_image_path = f"data/images/{camera_id}/{img_path.name}"

            record = AlertRecord(
                id=record_id,
                image_path=relative_image_path,
                image_filename=img_path.name,
                camera_id=camera_id,
                timestamp=timestamp,
                alert_type=alert_type,
                confidence=confidence,
                location_label=location,
                extra={"reindexed": True, "source": "reindex_script"},
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
