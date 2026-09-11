import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "disclaimer" in data

def test_speech_analyze_unsupported_format():
    response = client.post(
        "/api/speech/analyze",
        files={"file": ("test.txt", b"dummy audio content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]

def test_pain_analyze_unsupported_format():
    response = client.post(
        "/api/pain/analyze",
        files={"file": ("test.txt", b"dummy image content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]
