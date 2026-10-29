# ==============================================================================
# Ticket Triage Agent - Ingestion Runner (PowerShell)
# Runs incremental sync or backfill, exports interim parquet and CSV
# ==============================================================================

param (
    [string]$Mode = "incremental",
    [int]$Months = 6
)

$ErrorActionPreference = "Stop"

$uvCmd = if (Get-Command uv -ErrorAction SilentlyContinue) { "uv" } else { "$HOME\AppData\Roaming\Python\Python313\Scripts\uv.exe" }

Write-Host "==> Executing Ingestion Pipeline in '$Mode' mode..." -ForegroundColor Cyan

if ($Mode -eq "backfill") {
    & $uvCmd run python -m triage.ingest.backfill --months $Months
} elseif ($Mode -eq "comments") {
    & $uvCmd run python -m triage.ingest.comments
} else {
    & $uvCmd run python -m triage.ingest.incremental
}

Write-Host "Ingestion pipeline finished successfully." -ForegroundColor Green
