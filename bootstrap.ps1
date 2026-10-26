# ==============================================================================
# Ticket Triage Agent - Bootstrap Script (PowerShell)
# Verifies Python, ensures uv, syncs virtual environment, and sets up .env
# ==============================================================================

$ErrorActionPreference = "Stop"
Write-Host "==> Starting Ticket Triage Agent Bootstrap..." -ForegroundColor Cyan

# 1. Check Python installation (>= 3.11)
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py -3"
} else {
    Write-Error "Python 3.11+ is required but not found in PATH."
    exit 1
}

$pyVer = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Detected Python version: $pyVer" -ForegroundColor Green

# 2. Check and locate or install uv
$uvPath = $null
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $uvPath = "uv"
} elseif (Test-Path "$HOME\AppData\Roaming\Python\Python313\Scripts\uv.exe") {
    $uvPath = "$HOME\AppData\Roaming\Python\Python313\Scripts\uv.exe"
} elseif (Test-Path "$HOME\.local\bin\uv.exe") {
    $uvPath = "$HOME\.local\bin\uv.exe"
} elseif (Test-Path "$HOME\.cargo\bin\uv.exe") {
    $uvPath = "$HOME\.cargo\bin\uv.exe"
} else {
    Write-Host "uv not found in PATH. Installing uv via pip..." -ForegroundColor Yellow
    & $pythonCmd -m pip install uv
    if (Test-Path "$HOME\AppData\Roaming\Python\Python313\Scripts\uv.exe") {
        $uvPath = "$HOME\AppData\Roaming\Python\Python313\Scripts\uv.exe"
    } else {
        $uvPath = "uv"
    }
}

Write-Host "Using uv at: $uvPath" -ForegroundColor Green

# 3. Create .env from .env.example if missing
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host ".env created. Please configure secrets appropriately." -ForegroundColor Green
} else {
    Write-Host ".env already exists." -ForegroundColor Gray
}

# 4. Sync dependencies using uv
Write-Host "Syncing dependencies with uv..." -ForegroundColor Cyan
& $uvPath sync --all-extras
Write-Host "Virtual environment synced successfully." -ForegroundColor Green

Write-Host "`nEnvironment setup complete. See docs/SETUP.md for optional service keys." -ForegroundColor Cyan


Write-Host "Bootstrap completed successfully! Run ./start_services.ps1 to start services." -ForegroundColor Green
