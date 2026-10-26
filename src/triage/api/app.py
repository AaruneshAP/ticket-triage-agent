"""FastAPI Service for Ticket Triage Agent.

Provides endpoints:
- GET  /health: System and model readiness probe.
- POST /triage: Full triage prediction, completeness evaluation, duplicate retrieval, and draft response.
"""

import logging
import os

# Limit OpenBLAS and OpenMP thread concurrency to prevent Windows heap allocation exhaustion
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ticket Triage Agent API",
    description="Intelligent GitHub issue triage service with survival analysis, risk assessment, dense retrieval, and draft response generation.",
    version="1.0.0",
)

# Global agent singleton
_agent_instance = None


def get_agent():
    global _agent_instance
    if _agent_instance is None:
        from triage.agent.graph import TriageAgentGraph

        _agent_instance = TriageAgentGraph()
    return _agent_instance


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TriageRequest(BaseModel):
    ticket_number: int = Field(
        ..., description="GitHub issue number identifier", json_schema_extra={"example": 187117}
    )
    title: str = Field(
        ...,
        description="Issue title",
        json_schema_extra={"example": "Shelly integration device unavailable after reboot"},
    )
    body: str = Field(..., description="Raw markdown body of the issue report")
    created_at: Optional[str] = Field(
        None,
        description="ISO-8601 creation timestamp",
        json_schema_extra={"example": "2026-10-26T12:00:00Z"},
    )
    labels: Optional[List[str]] = Field(
        default_factory=list, description="Initial issue labels if any"
    )
    author: Optional[str] = Field("github-user", description="Author GitHub login")
    author_association: Optional[str] = Field(
        "NONE", description="Author association with repository"
    )


class DaysToClosePrediction(BaseModel):
    p50: Optional[float] = Field(None, description="Estimated median days to close (suppressed due to lack of signal)")
    p90: Optional[float] = Field(None, description="Estimated 90th percentile days to close (suppressed due to lack of signal)")
    confidence: str = Field("low", description="Confidence level: always 'low' due to lack of out-of-sample signal")
    confidence_reason: str = Field(
        ..., description="Explanation of model uncertainty and empirical boundaries"
    )
    status: str = Field(
        "no reliable signal for resolution time",
        description="Operational status indicator",
    )


class ShapFeatureImpact(BaseModel):
    feature: str = Field(..., description="Feature identifier")
    impact: float = Field(..., description="SHAP value contribution")
    direction: str = Field(..., description="increases_risk or reduces_risk")
    value: float = Field(..., description="Normalized feature value in ticket")


class SimilarTicket(BaseModel):
    ticket_number: int = Field(..., description="Peer issue number")
    title: str = Field(..., description="Peer issue title")
    state: str = Field(..., description="Issue state: open or closed")
    integration: Optional[str] = Field(None, description="Integration label of peer issue")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")


class FixedHorizonPrediction(BaseModel):
    p_closed_within_14d: Optional[float] = Field(
        None, description="Diagnostic probability only (suppressed from actionable display)"
    )
    p_closed_within_30d: Optional[float] = Field(
        None, description="Diagnostic probability only (suppressed from actionable display)"
    )
    roc_auc_14d: float = Field(
        0.4621, description="Strict out-of-sample ROC-AUC for 14-day horizon (0.46, chance)"
    )
    roc_auc_30d: float = Field(
        0.5329, description="Strict out-of-sample ROC-AUC for 30-day horizon (0.53, chance)"
    )
    confidence: str = Field(
        "none", description="Confidence level: 'none' due to lack of out-of-sample signal"
    )
    confidence_reason: str = Field(
        ..., description="Explanation: no reliable signal for resolution time"
    )
    status: str = Field(
        "no reliable signal for resolution time",
        description="Status indicator",
    )


class TriageResponse(BaseModel):
    ticket_number: int = Field(..., description="GitHub issue number")
    predicted_days_to_close: DaysToClosePrediction = Field(
        ..., description="Resolution latency prediction (suppressed due to lack of signal)"
    )
    fixed_horizon_probabilities: Optional[FixedHorizonPrediction] = Field(
        None, description="Fixed horizon closure prediction (suppressed due to lack of signal)"
    )
    p_not_planned: float = Field(
        ..., description="Probability of issue closing without a fix (0.0 to 1.0)"
    )
    p_not_planned_confidence: str = Field(..., description="Confidence level of risk prediction")
    top_shap_features: List[ShapFeatureImpact] = Field(
        default_factory=list, description="Top SHAP explainability drivers"
    )
    similar_tickets: List[SimilarTicket] = Field(
        default_factory=list, description="Top semantically similar prior issues"
    )
    missing_info: List[str] = Field(
        default_factory=list, description="List of missing diagnostic or version items"
    )
    suggested_label: str = Field(..., description="Recommended integration label or 'other'")
    label_confidence: float = Field(..., description="Confidence score for label suggestion")
    draft_reply: str = Field(..., description="Automated non-promising triage response draft")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Service health status")
    version: str = Field(default="1.0.0", description="API version")
    environment: str = Field(default="development", description="Environment mode")
    services: Dict[str, str] = Field(
        default_factory=lambda: {
            "api": "healthy",
            "models": "ready",
            "warehouse": "databricks" if os.getenv("DATABRICKS_HOST") else "local_fallback",
        }
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health() -> HealthResponse:
    """Check health and subsystem readiness."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        environment=os.getenv("APP_ENV", "development"),
        services={
            "api": "healthy",
            "models": "ready" if os.path.exists("models/survival_aft.joblib") else "unloaded",
            "warehouse": "databricks" if os.getenv("DATABRICKS_HOST") else "local_fallback",
        },
    )


@app.post("/triage", response_model=TriageResponse, tags=["Triage"])
def post_triage(request: TriageRequest) -> TriageResponse:
    """
    Execute full triage workflow:
    1. Parse form sections
    2. Check completeness & diagnostic logs
    3. Predict survival time (with Low Confidence flag), not-planned risk (SHAP), and label
    4. Retrieve prior similar tickets (temporally bounded)
    5. Draft polite response
    """
    try:
        agent = get_agent()
        result = agent.run(request.model_dump())
        return TriageResponse(**result)
    except Exception as e:
        logger.exception(f"Error during triage execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    uvicorn.run("triage.api.app:app", host=host, port=port, reload=True)
