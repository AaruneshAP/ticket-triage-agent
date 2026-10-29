FROM python:3.12-slim

# Avoid writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential and curl for uv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv from astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency manifests and readme required for hatchling metadata
COPY pyproject.toml uv.lock README.md ./

# Install project dependencies without installing root package first (cached layer)
RUN uv sync --frozen --no-dev --all-extras --no-install-project

# Copy source code, documentation, and data assets
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY docs/ ./docs/

# Final sync to install root package
RUN uv sync --frozen --no-dev --all-extras

# Add virtualenv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Default entrypoint starts FastAPI service
CMD ["uvicorn", "triage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
