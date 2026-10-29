"""Tests for FastAPI triage service endpoints."""

from fastapi.testclient import TestClient

from triage.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "services" in data
    assert data["services"]["api"] == "healthy"


def test_triage_endpoint_mock_call(monkeypatch):
    # Mock agent run to verify API schema and validation without waiting for full embedding search
    mock_response = {
        "ticket_number": 187117,
        "predicted_days_to_close": {
            "p50": None,
            "p90": None,
            "confidence": "low",
            "confidence_reason": "Empirical survival C-index is ~0.514 (close to chance).",
            "status": "no reliable signal for resolution time",
        },
        "p_not_planned": 0.222,
        "p_not_planned_confidence": "moderate",
        "top_shap_features": [
            {
                "feature": "n_missing_form_fields",
                "impact": 0.12,
                "direction": "increases_risk",
                "value": 2.0,
            }
        ],
        "similar_tickets": [
            {
                "ticket_number": 182001,
                "title": "Shelly timeout on reboot",
                "state": "closed",
                "integration": "shelly",
                "similarity_score": 0.81,
            }
        ],
        "missing_info": ["Diagnostic logs or Python traceback"],
        "suggested_label": "integration: shelly",
        "label_confidence": 0.85,
        "draft_reply": "[AI draft, needs review]\nThanks for reporting. Could you provide diagnostic logs?",
    }

    class MockAgent:
        def run(self, raw_issue):
            return mock_response

    monkeypatch.setattr("triage.api.app.get_agent", lambda: MockAgent())

    payload = {
        "ticket_number": 187117,
        "title": "Shelly integration fails after power cycle",
        "body": "The device stays unavailable when powered back on.",
        "created_at": "2026-10-26T12:00:00Z",
        "labels": [],
    }

    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["ticket_number"] == 187117
    assert data["predicted_days_to_close"]["confidence"] == "low"
    assert data["predicted_days_to_close"]["status"] == "no reliable signal for resolution time"
    assert data["p_not_planned"] == 0.222
    assert len(data["similar_tickets"]) == 1
    assert data["suggested_label"] == "integration: shelly"
    assert data["draft_reply"].startswith("[AI draft, needs review]")


def test_triage_endpoint_live():
    """Live end-to-end test of POST /triage invoking full agent graph."""
    payload = {
        "ticket_number": 188051,
        "title": "TP-Link Omada platform initialisation failure",
        "body": "### The problem\nPlatform error during setup.\n\n### Integration causing the issue\ntplink_omada\n\n### What version of Home Assistant Core has the issue?\n2026.10.1\n\n### What was the last working version?\n2026.9.0\n\n### What type of installation are you running?\nHome Assistant OS\n\n### Traceback\n```\nKeyError: 'currentVersion'\n```",
        "created_at": "2026-10-26T12:00:00Z",
        "labels": ["integration: tplink_omada"],
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["ticket_number"] == 188051
    assert "predicted_days_to_close" in data
    assert data["predicted_days_to_close"]["confidence"] == "low"
    assert "p_not_planned" in data
    assert "suggested_label" in data
    assert data["draft_reply"].startswith("[AI draft, needs review]")
