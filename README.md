# Imagify 🎥

Semantic image search for Jetson alert footage.

Your Jetson device pushes alert images to the API. Imagify embeds them with CLIP and stores the vectors in Chroma. You then search the footage in plain English — *"person near the gate at night"*, *"vehicle on camera 3"* — and get matching images back.

---

## Architecture

```
Jetson device
    │
    │  POST /api/v1/ingest  (image + camera_id + timestamp + optional metadata)
    ▼
FastAPI backend
    ├── services/ingest.py    validate + save image to disk
    ├── services/rag.py       CLIP-embed the image, upsert into Chroma
    └── repositories/         Chroma vector store
    │
    │  POST /api/v1/search  (text query → base64 images + metadata)
    ▼
Streamlit chatbot UI
    └── frontend/app.py       type a query, see matching alert images
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp configs/.env.example .env
# Edit .env if needed (paths, device, model)
```

### 3. Start the API

```bash
uvicorn backend.app.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 4. Start the chatbot

```bash
streamlit run frontend/app.py
# → http://localhost:8501
```

### 5. Push a test alert (from your Jetson or locally)

```bash
python scripts/send_alert.py \
    --image /path/to/alert.jpg \
    --camera-id "cam-01" \
    --alert-type "person" \
    --confidence 0.92 \
    --location "Front Gate"
```

---

## Docker

```bash
docker-compose up --build
# API  → http://localhost:8000
# UI   → http://localhost:8501
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/ingest` | Push alert image from Jetson |
| `POST` | `/api/v1/search` | Semantic search by text query |
| `GET`  | `/api/v1/collections` | Stats (total alerts, camera list) |
| `DELETE` | `/api/v1/alerts/{id}` | Remove an alert |
| `GET`  | `/api/v1/health` | Liveness probe |

Full interactive docs at `/docs` when the API is running.

### Ingest payload (multipart/form-data)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | ✅ | Alert image (JPEG, PNG, BMP, WEBP, TIFF) |
| `camera_id` | string | ✅ | Camera / device identifier |
| `timestamp` | ISO-8601 datetime | ✅ | Alert time from the device |
| `alert_type` | string | ❌ | e.g. `motion`, `person`, `vehicle` |
| `confidence` | float 0–1 | ❌ | Detection confidence |
| `location_label` | string | ❌ | Human-readable camera location |
| `extra_json` | JSON string | ❌ | Any additional metadata |

### Jetson integration (Python)

```python
from scripts.send_alert import send_alert

send_alert(
    image_path="/path/to/frame.jpg",
    camera_id="cam-01",
    api_url="http://your-server:8000",
    alert_type="person",
    confidence=0.94,
    location_label="Back Entrance",
)
```

---

## Project Structure

```
Imagify/
├── backend/app/
│   ├── api/v1/router.py        FastAPI endpoints
│   ├── core/                   config, logging, exceptions
│   ├── models/alert.py         domain models
│   ├── schemas/alert.py        HTTP request/response schemas
│   ├── repositories/           Chroma vector store abstraction
│   ├── services/
│   │   ├── ingest.py           save image to disk, build AlertRecord
│   │   ├── rag.py              CLIP embedding + index into Chroma
│   │   └── retrieval.py        text query → ranked image results
│   ├── prompts/                LLM caption templates (optional enrichment)
│   ├── utils/                  shared helpers
│   └── main.py                 FastAPI app factory
├── frontend/app.py             Streamlit chatbot UI
├── scripts/send_alert.py       Jetson push helper / CLI test tool
├── configs/.env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Extending later

- **LLM captions** — enable `OLLAMA_ENABLED=true` + run LLaVA locally to auto-caption each image on ingest, giving richer text metadata for search.
- **Larger CLIP** — swap `CLIP_MODEL_NAME` to `openai/clip-vit-large-patch14` for better accuracy.
- **GPU** — set `EMBEDDING_DEVICE=cuda` on any NVIDIA machine.
- **More metadata** — add fields to `AlertRecord` and include them in the ingest form. No schema migrations needed; Chroma stores them as arbitrary metadata.
