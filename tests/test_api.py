"""
Basic smoke tests — no CLIP model or Chroma needed.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    from backend.app.main import app
    from backend.app.repositories.vector_store import get_vector_store

    mock_vs = MagicMock()
    mock_vs.count.return_value = 0
    mock_vs.list_cameras.return_value = []

    app.dependency_overrides[get_vector_store] = lambda: mock_vs
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_collections(client):
    r = client.get("/api/v1/collections")
    assert r.status_code == 200
    data = r.json()
    assert "total_alerts" in data
    assert "cameras" in data
