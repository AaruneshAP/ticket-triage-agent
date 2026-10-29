"""Production API runner script."""

import os

# Thread limits to prevent Windows OpenBLAS memory heap errors
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "triage.api.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
