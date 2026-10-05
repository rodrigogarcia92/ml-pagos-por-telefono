# ---------------------------------------------------------------------------
# MLflow tracking server
#
#   .\scripts\start_mlflow.ps1
#
# Backend : SQLite  (the Model Registry REQUIRES a database backend; the default
#                    file store cannot register models, and the Registry is what
#                    FastAPI loads from at roadmap step 11)
# Artifacts: GCS gs://pagos-telefono-26-mlflow  (us-central1 - the always-free
#                    5 GB is REGIONAL; the US multi-region is not covered)
# Workers  : 1  (deliberate, not a default - see the note above the command)
#
# Leave this running in its own terminal for the whole modelling session.
# Ctrl-C to stop.  Docs: docs/mlflow_setup.md
#
# ASCII ONLY IN THIS FILE. Windows PowerShell 5.1 reads .ps1 as ANSI (CP1252)
# unless the file has a UTF-8 BOM. A UTF-8 em dash then decodes as three CP1252
# characters, the last of which is a smart closing quote - which PowerShell
# accepts as a string delimiter, so the string ends early and the parser fails
# somewhere unrelated. Keep every character in this file plain ASCII.
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# Resolve to the project root regardless of where the script is invoked from.
# This is what lets the SQLite URI below stay relative, which sidesteps the
# Windows drive-letter problem in SQLAlchemy URIs entirely.
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if (-not (Test-Path "mlflow")) {
    New-Item -ItemType Directory -Path "mlflow" | Out-Null
    Write-Host "Created .\mlflow (gitignored)" -ForegroundColor Yellow
}

# Fail early and legibly if ADC are missing, rather than 20 minutes into a
# sweep when the first artifact upload is attempted.
try {
    gcloud auth application-default print-access-token *> $null
} catch {
    Write-Host "ERROR: no Application Default Credentials." -ForegroundColor Red
    Write-Host "Run:  gcloud auth application-default login" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  MLflow tracking server" -ForegroundColor Cyan
Write-Host "  backend   : sqlite:///mlflow/mlflow.db"
Write-Host "  artifacts : gs://pagos-telefono-26-mlflow"
Write-Host "  workers   : 1  (SQLite - see note in this file)"
Write-Host "  UI        : http://127.0.0.1:5000"
Write-Host ""

# --workers 1 is deliberate. MLflow 3 defaults to 4 uvicorn workers, each with
# its own SQLAlchemy engine against the SAME SQLite file. SQLite serialises
# writers, so concurrent workers produce lock contention -> HTTP 500s and hung
# client calls. There is exactly one client here (the development machine), so extra workers buy
# nothing and cost reliability.
#
# The same reasoning applies to sweep.py: joblib workers fit models and RETURN
# results; the parent process does all the MLflow logging. One writer, always.
mlflow server `
    --backend-store-uri "sqlite:///mlflow/mlflow.db" `
    --artifacts-destination "gs://pagos-telefono-26-mlflow" `
    --serve-artifacts `
    --host 127.0.0.1 `
    --port 5000 `
    --workers 1
