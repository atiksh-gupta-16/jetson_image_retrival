"""
Highlight service — CLIP-guided region highlighting.

Given a surveillance image (which may contain many people) and a natural-
language query like "red shirt guy", this module:

1. Slides a set of overlapping crops across the image at multiple scales.
2. Embeds each crop with CLIP and scores it against the query text embedding.
3. Draws a bright bounding box + translucent overlay around the top-scoring
   region on the original image.
4. Returns the annotated image as raw PIL Image.

Public API
----------
highlight_image(image_path, query_text) -> PIL.Image.Image | None
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFilter

from backend.app.core.logging import get_logger
from backend.app.services.rag import _load_clip   # reuse the singleton

logger = get_logger(__name__)


# ── Sliding window generation ────────────────────────────────────────────────

def _generate_crops(
    img_w: int,
    img_h: int,
    scales: Tuple[float, ...] = (0.10, 0.15, 0.20),
    stride_ratio: float = 0.25,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate (x1, y1, x2, y2) crop boxes at several scales.

    Each scale defines the crop size as a fraction of the smaller image
    dimension.  Crops are slid with ``stride_ratio`` overlap.
    """
    boxes: List[Tuple[int, int, int, int]] = []
    min_dim = min(img_w, img_h)

    for scale in scales:
        crop_size = max(int(min_dim * scale), 32)
        stride = max(int(crop_size * stride_ratio), 16)

        for y in range(0, img_h - crop_size + 1, stride):
            for x in range(0, img_w - crop_size + 1, stride):
                boxes.append((x, y, x + crop_size, y + crop_size))

    return boxes


# ── Core scoring ─────────────────────────────────────────────────────────────

def _score_crops(
    img: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    query_text: str,
) -> List[float]:
    """
    Score every crop box against *query_text* using CLIP.

    Returns a list of cosine-similarity scores (same order as *boxes*).
    """
    from backend.app.core.config import get_settings

    cfg = get_settings()
    model, processor = _load_clip()

    # 1. Embed the query text once
    text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(cfg.EMBEDDING_DEVICE) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_feat = model.get_text_features(**text_inputs)
        if not isinstance(text_feat, torch.Tensor):
            text_feat = text_feat.pooler_output
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)   # (1, D)

    # 2. Embed crops in batches to avoid OOM
    BATCH = 32
    all_scores: List[float] = []

    for i in range(0, len(boxes), BATCH):
        batch_boxes = boxes[i : i + BATCH]
        crops = [img.crop(b).resize((224, 224)) for b in batch_boxes]

        img_inputs = processor(images=crops, return_tensors="pt")
        img_inputs = {k: v.to(cfg.EMBEDDING_DEVICE) for k, v in img_inputs.items()}

        with torch.no_grad():
            img_feats = model.get_image_features(**img_inputs)
            if not isinstance(img_feats, torch.Tensor):
                img_feats = img_feats.pooler_output
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # (B, D)

            sims = (img_feats @ text_feat.T).squeeze(-1)  # (B,)
            all_scores.extend(sims.cpu().tolist())

    return all_scores


# ── Drawing ──────────────────────────────────────────────────────────────────

def _nms(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_threshold: float = 0.2) -> List[int]:
    """Non-maximum suppression to filter out overlapping boxes."""
    if len(boxes) == 0:
        return []

    # Sort boxes by score in descending order
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    def get_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    while len(indices) > 0:
        current_idx = indices.pop(0)
        keep.append(current_idx)

        # Filter out remaining boxes that have high overlap with current box
        indices = [i for i in indices if get_iou(boxes[current_idx], boxes[i]) <= iou_threshold]

    return keep


def _draw_highlights(
    img: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    line_width: int = 3,
) -> Image.Image:
    """
    Draw bright bounding boxes around all given *boxes*.
    """
    result = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(result)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        for offset in range(line_width):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=(*color, 255),
            )

    return result.convert("RGB")


# ── Public API ───────────────────────────────────────────────────────────────

def highlight_image(
    image_path: Path | str,
    query_text: str,
    min_score_threshold: float = 0.15,
) -> Optional[Image.Image]:
    """
    Open *image_path*, find the region that best matches *query_text*,
    and return an annotated PIL Image with that region highlighted.

    Returns ``None`` if the image cannot be opened or no crop exceeds
    the minimum score threshold (avoids false-positive highlights).
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("highlight: image not found: %s", path)
        return None

    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        logger.error("highlight: cannot open image %s: %s", path, exc)
        return None

    w, h = img.size
    boxes = _generate_crops(w, h)
    if not boxes:
        logger.warning("highlight: no crops generated for %dx%d image", w, h)
        return None

    logger.info("highlight: scoring %d crops for query %r", len(boxes), query_text)
    scores = _score_crops(img, boxes, query_text)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]
    
    # Adaptive threshold: keep boxes that score at least 90% as well as the best box,
    # but never below a hard baseline of 0.20 to avoid matching background.
    adaptive_threshold = max(best_score * 0.90, 0.20)
    
    valid_indices = [i for i, s in enumerate(scores) if s >= adaptive_threshold]
    
    if not valid_indices:
        logger.info("highlight: best score (%.4f) below baseline threshold — no highlight", best_score)
        return None
        
    valid_boxes = [boxes[i] for i in valid_indices]
    valid_scores = [scores[i] for i in valid_indices]
    
    keep_indices = _nms(valid_boxes, valid_scores, iou_threshold=0.2)
    final_boxes = [valid_boxes[i] for i in keep_indices]

    logger.info(
        "highlight: found %d boxes above threshold after NMS (threshold=%.2f)",
        len(final_boxes),
        min_score_threshold,
    )

    return _draw_highlights(img, final_boxes)


def highlight_image_b64(
    image_path: Path | str,
    query_text: str,
) -> Optional[str]:
    """
    Convenience wrapper: returns the highlighted image as a base64 string
    (ready for ``<img src="data:image/jpeg;base64,...">``) or None.
    """
    highlighted = highlight_image(image_path, query_text)
    if highlighted is None:
        return None

    buf = BytesIO()
    highlighted.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
