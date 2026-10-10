"""
Integration Tests for FastAPI Service and Input Validation.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Bilingual Document Q&A (RAG) Service"
    assert "english" in data["supported_languages"]
    assert "arabic" in data["supported_languages"]


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "collection_name" in data
    assert "embedding_model" in data


def test_query_validation_too_short():
    # Query length < 3 characters should fail with 422 Unprocessable Entity
    response = client.post(
        "/api/v1/query",
        json={"query": "ab"}
    )
    assert response.status_code == 422


def test_query_validation_invalid_top_k():
    # top_k > 20 should fail validation
    response = client.post(
        "/api/v1/query",
        json={"query": "Valid query here?", "top_k": 50}
    )
    assert response.status_code == 422


def test_query_validation_invalid_mode():
    # retrieval_mode not in ['baseline', 'reranked']
    response = client.post(
        "/api/v1/query",
        json={"query": "Valid query here?", "retrieval_mode": "non_existent_mode"}
    )
    assert response.status_code == 422


def test_query_english_success():
    payload = {
        "query": "What was the total enterprise AI investment in 2024?",
        "retrieval_mode": "reranked",
        "top_k": 3
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "english"
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "latency_ms" in data
    assert "citations" in data


def test_query_arabic_success():
    payload = {
        "query": "ما هي نسبة نمو الناتج المحلي الإجمالي غير النفطي؟",
        "retrieval_mode": "baseline",
        "top_k": 3
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "arabic"
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_retrieve_endpoint():
    payload = {
        "query": "Renewable energy and solar power capacity",
        "retrieval_mode": "baseline",
        "top_k": 3
    }
    response = client.post("/api/v1/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "passages" in data
    assert data["retrieval_mode"] == "baseline"
    assert "total_retrieved" in data
