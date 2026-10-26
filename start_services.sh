#!/usr/bin/env bash
# ==============================================================================
# Ticket Triage Agent - Service Starter (Bash)
# Starts FastAPI on port 8000 and Streamlit on port 8501
# ==============================================================================

set -euo pipefail

API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-0.0.0.0}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

echo "==> Starting Ticket Triage Agent Services..."
echo "  FastAPI Host/Port:    http://localhost:$API_PORT"
echo "  Streamlit Host/Port:  http://localhost:$STREAMLIT_PORT"

if command -v uv &>/dev/null; then
    uv run uvicorn triage.api.app:app --host "$API_HOST" --port "$API_PORT" &
    API_PID=$!
    uv run streamlit run src/triage/api/streamlit_app.py --server.port "$STREAMLIT_PORT" --server.headless true &
    ST_PID=$!
else
    source .venv/bin/activate
    python -m uvicorn triage.api.app:app --host "$API_HOST" --port "$API_PORT" &
    API_PID=$!
    streamlit run src/triage/api/streamlit_app.py --server.port "$STREAMLIT_PORT" --server.headless true &
    ST_PID=$!
fi

echo ""
echo "Services launched in background:"
echo "  FastAPI PID:     $API_PID"
echo "  Streamlit PID:   $ST_PID"
echo "  Health Check:    http://localhost:$API_PORT/health"
echo "  Documentation:   http://localhost:$API_PORT/docs"
echo "  Review Queue:    http://localhost:$STREAMLIT_PORT"
