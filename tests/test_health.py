"""Tests for API health endpoint."""

from fastapi.testclient import TestClient

from triage.api.app import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 with ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "services" in data
    assert data["services"]["api"] == "healthy"
