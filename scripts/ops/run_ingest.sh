#!/usr/bin/env bash
# ==============================================================================
# Ticket Triage Agent - Ingestion Runner (Bash)
# Runs incremental sync or backfill, exports interim parquet and CSV
# ==============================================================================

set -euo pipefail

MODE="${1:-incremental}"
MONTHS="${2:-6}"

echo "==> Executing Ingestion Pipeline in '$MODE' mode..."

if [ "$MODE" = "backfill" ]; then
    uv run python -m triage.ingest.backfill --months "$MONTHS"
elif [ "$MODE" = "comments" ]; then
    uv run python -m triage.ingest.comments
else
    uv run python -m triage.ingest.incremental
fi

echo "Ingestion pipeline finished successfully."
