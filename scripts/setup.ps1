# Orbimesh one-time developer setup (Windows PowerShell)
# Usage: Right-click → Run with PowerShell, or:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1

$ErrorActionPreference = 'Stop'

function Write-Section($msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "! $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "✗ $msg" -ForegroundColor Red }

# Resolve repo root
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'

# 1) Backend: Python venv + deps
Write-Section 'Backend: Python environment'
Set-Location $Backend
if (!(Test-Path '.venv')) {
  Write-Host 'Creating virtual environment (.venv)...'
  python -m venv .venv
}
$Py = Join-Path $Backend '.venv\Scripts\python.exe'
& $Py -m pip install --upgrade pip
& $Py -m pip install -r requirements.txt
Write-Ok 'Python dependencies installed'

# 2) Backend: .env
if (!(Test-Path '.env')) {
  if (Test-Path '.env.example') {
    Copy-Item '.env.example' '.env'
    Write-Warn 'Created backend/.env from example. Please edit your API keys and DB settings.'
  } else {
    Write-Warn 'backend/.env.example not found. Please create backend/.env manually.'
  }
}

# 3) Backend: DB setup (create DB, enable pgvector, run migrations)
Write-Section 'Backend: Database setup'
& $Py 'setup.py'
Write-Ok 'Database created/migrated'

# 4) Backend: Sync agent definitions
Write-Section 'Backend: Agent sync'
& $Py 'manage.py' sync
if ($LASTEXITCODE -ne 0) {
  Write-Warn 'Agent sync completed with warnings. If you see request_format column errors, run Alembic: alembic upgrade head and re-run this script.'
} else {
  Write-Ok 'Agents synced'
}

# 5) Frontend: Node deps
Write-Section 'Frontend: Dependencies'
Set-Location $Frontend
# Prefer pnpm if available
$pnpm = (Get-Command pnpm -ErrorAction SilentlyContinue)
if ($pnpm) {
  pnpm install
} else {
  Write-Warn 'pnpm not found, using npm (install pnpm for faster installs: npm i -g pnpm)'
  npm install --legacy-peer-deps
}
Write-Ok 'Frontend dependencies installed'

# 6) Frontend: .env.local
if (!(Test-Path '.env.local')) {
  @(
    'NEXT_PUBLIC_API_URL=http://127.0.0.1:8000',
    'NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_your_clerk_key_here',
    'NEXT_PUBLIC_CLERK_JWT_TEMPLATE=backend'
  ) | Out-File -FilePath '.env.local' -Encoding utf8 -Force
  Write-Warn 'Created frontend/.env.local. Update Clerk keys as needed.'
}

Write-Section 'Setup Complete'
Write-Host 'Next steps:' -ForegroundColor Cyan
Write-Host '  1) Edit backend/.env with real keys (GROQ_API_KEY, PG_PASSWORD, etc.)'
Write-Host '  2) Start dev: scripts/dev.ps1'
Write-Host '  3) Open http://localhost:3000'
