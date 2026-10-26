#!/usr/bin/env bash
# ==============================================================================
# Ticket Triage Agent - Bootstrap Script (Bash)
# Verifies Python, ensures uv, syncs virtual environment, and sets up .env
# ==============================================================================

set -euo pipefail

echo "==> Starting Ticket Triage Agent Bootstrap..."

# 1. Check Python installation (>= 3.11)
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3.11+ is required but not found in PATH." >&2
    exit 1
fi

PY_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PY_VER"

# 2. Check and locate or install uv
if command -v uv &>/dev/null; then
    UV_CMD="uv"
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    UV_CMD="$HOME/.cargo/bin/uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_CMD="$HOME/.local/bin/uv"
else
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_CMD="$HOME/.local/bin/uv"
fi

echo "Using uv at: $UV_CMD"

# 3. Create .env from .env.example if missing
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ".env created. Please configure secrets appropriately."
else
    echo ".env already exists."
fi

# 4. Sync dependencies
echo "Syncing dependencies with uv..."
"$UV_CMD" sync --all-extras
echo "Virtual environment synced successfully."

echo ""
echo "Environment setup complete. See docs/SETUP.md for optional service keys."

echo ""
echo "Bootstrap completed successfully! Run ./start_services.sh to start services."
