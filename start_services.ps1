# ==============================================================================
# Ticket Triage Agent - Service Starter (PowerShell)
# Starts FastAPI on port 8000 and Streamlit on port 8501
# ==============================================================================

$apiPort = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$apiHost = if ($env:API_HOST) { $env:API_HOST } else { "0.0.0.0" }
$streamlitPort = if ($env:STREAMLIT_PORT) { $env:STREAMLIT_PORT } else { "8501" }

$workDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }

Write-Host "==> Starting Ticket Triage Agent Services..." -ForegroundColor Cyan
Write-Host "  FastAPI Host/Port:    http://localhost:$apiPort" -ForegroundColor Green
Write-Host "  Streamlit Host/Port:  http://localhost:$streamlitPort" -ForegroundColor Green

$venvPython = Join-Path $workDir ".venv\Scripts\python.exe"
$venvStreamlit = Join-Path $workDir ".venv\Scripts\streamlit.exe"

if (Test-Path $venvPython) {
    Write-Host "Launching FastAPI service via venv python..." -ForegroundColor Cyan
    cmd.exe /c "start `"`" `"$venvPython`" -m uvicorn triage.api.app:app --host $apiHost --port $apiPort"

    Write-Host "Launching Streamlit service via venv streamlit..." -ForegroundColor Cyan
    cmd.exe /c "start `"`" `"$venvStreamlit`" run src/triage/api/streamlit_app.py --server.port $streamlitPort --server.headless true"
} else {
    Write-Host "Launching services via uv..." -ForegroundColor Cyan
    cmd.exe /c "start `"`" uv run uvicorn triage.api.app:app --host $apiHost --port $apiPort"
    cmd.exe /c "start `"`" uv run streamlit run src/triage/api/streamlit_app.py --server.port $streamlitPort --server.headless true"
}

Write-Host "`nServices started!" -ForegroundColor Green
Write-Host "Health Check:    http://localhost:$apiPort/health" -ForegroundColor Yellow
Write-Host "Documentation:   http://localhost:$apiPort/docs" -ForegroundColor Yellow
Write-Host "Review Queue:    http://localhost:$streamlitPort" -ForegroundColor Yellow
