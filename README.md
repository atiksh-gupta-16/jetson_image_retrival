# Imagify 🎥 - Semantic Image Search & Highlighting

Semantic image search for Jetson alert footage with intelligent region highlighting.

Your Jetson device pushes alert images to the API. Imagify embeds them with OpenAI's CLIP model and stores the vectors in ChromaDB. You can then search the footage in plain English — *"red shirt guy"*, *"vehicle on camera 3"* — and the system not only retrieves the matching images but automatically draws a bounding box around the exact region of the image that matches your query.

---

## How It Works (The Pipeline Explained)

### 1. Ingestion (Storing Images)
- **Tool**: `backend/app/services/ingest.py` & `backend/app/services/rag.py`
- When a Jetson device sends an image to the `/api/v1/ingest` endpoint, it is first saved to disk.
- Then, the entire image is passed to **OpenAI's CLIP Model** (`openai/clip-vit-base-patch32` loaded via the `transformers` library in `rag.py`).
- CLIP generates a mathematical representation (a high-dimensional vector) of the visual contents of the image.
- This vector, along with metadata (camera ID, timestamp, disk path), is stored in **ChromaDB** (a local vector database).

### 2. Search & Intent Extraction
- **Tool**: `backend/app/services/intent.py` & `backend/app/services/query_pipeline.py`
- When you type a query like *"red shirt guy"* in the Streamlit UI, the query first goes to a local **Ollama LLM** (`qwen2.5:1.5b`).
- The LLM parses the sentence to extract structured filters (e.g., if you said "camera 2", it sets `camera_id="2"`). It also isolates the pure semantic query (e.g., "red shirt guy").
- The pipeline then passes this parsed intent to the retrieval engine.

### 3. Semantic Image Retrieval
- **Tool**: `backend/app/services/retrieval.py` & `backend/app/services/rag.py`
- The semantic query ("red shirt guy") is passed through the same **CLIP Model** to generate a text embedding vector.
- The system queries **ChromaDB** to find image vectors that are mathematically closest (highest cosine similarity) to the text vector.
- This quickly filters down the thousands of images to the top matching ones that conceptually resemble the text description.

### 4. Intelligent Region Highlighting (Drawing the Box)
- **Tool**: `backend/app/services/highlight.py`
- Once the top images are retrieved, they are passed to the highlight service.
- The service uses a **sliding window approach**: it generates overlapping "crops" (sub-regions) of the image at different sizes (25%, 35%, 50% of the image size).
- Every single crop is passed through the **CLIP Model** and scored against your text query ("red shirt guy").
- The crop with the highest similarity score is identified as the exact location of the object/person you asked for.
- A **green bounding box** is then drawn around this highest-scoring crop using the `Pillow` (PIL) library before returning the image to the frontend.

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
│   │   ├── retrieval.py        text query → ranked image results
│   │   └── highlight.py        CLIP-guided bounding box highlighting
│   ├── prompts/                LLM caption templates
│   ├── utils/                  shared helpers
│   └── main.py                 FastAPI app factory
├── frontend/app.py             Streamlit chatbot UI
├── scripts/send_alert.py       Jetson push helper / CLI test tool
├── configs/.env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
