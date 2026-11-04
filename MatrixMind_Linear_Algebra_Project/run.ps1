$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.venv')) {
    Write-Host 'Creating Python virtual environment…' -ForegroundColor Cyan
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { py -m venv .venv } else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment. Install Python 3.11 or later and retry.' }
}
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host 'Created .env. Add GROQ_API_KEY for AI-generated answers; local RAG works without it.' -ForegroundColor Yellow
}
Write-Host 'Starting MatrixMind at http://127.0.0.1:8000' -ForegroundColor Green
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
