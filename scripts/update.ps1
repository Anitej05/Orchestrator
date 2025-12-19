# Orbimesh Update Script - For existing installations
# Updates dependencies, database schema, and agent definitions
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\update.ps1

$ErrorActionPreference = 'Stop'

function Write-Section($msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Resolve repo root
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'

Write-Host @"

╔═══════════════════════════════════════════════════════════╗
║                  ORBIMESH UPDATE SCRIPT                   ║
║          Updates: Dependencies + DB + Agents              ║
╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# 1) Backend: Update Python dependencies
Write-Section 'Backend: Updating Python dependencies'
Set-Location $Backend

# Find Python executable (venv or system)
$Py = $null
if (Test-Path '.venv\Scripts\python.exe') {
  $Py = Join-Path $Backend '.venv\Scripts\python.exe'
  Write-Host "Using venv Python: .venv\Scripts\python.exe"
} elseif (Test-Path 'venv\Scripts\python.exe') {
  $Py = Join-Path $Backend 'venv\Scripts\python.exe'
  Write-Host "Using venv Python: venv\Scripts\python.exe"
} else {
  $Py = 'python'
  Write-Warn 'No virtual environment found. Using system Python.'
}

# Upgrade pip first
Write-Host 'Upgrading pip...'
& $Py -m pip install --upgrade pip --quiet

# Install/update all requirements
Write-Host 'Installing/updating backend dependencies...'
& $Py -m pip install -r requirements.txt --upgrade
Write-Ok 'Backend dependencies updated'

# 2) Backend: Run database migrations
Write-Section 'Backend: Database migrations'
Write-Host 'Running setup.py (creates DB if needed, runs migrations)...'
& $Py 'setup.py'
if ($LASTEXITCODE -eq 0) {
  Write-Ok 'Database schema updated'
} else {
  Write-Err 'Database setup failed. Check logs above.'
  exit 1
}

# 3) Backend: Sync agent definitions
Write-Section 'Backend: Syncing agent definitions'
Write-Host 'Running agent sync...'
& $Py 'manage.py' sync
if ($LASTEXITCODE -eq 0) {
  Write-Ok 'Agents synced successfully'
} else {
  Write-Warn 'Agent sync completed with warnings. Check logs above.'
}

# 4) Frontend: Update Node dependencies
Write-Section 'Frontend: Updating Node dependencies'
Set-Location $Frontend

# Prefer pnpm if available
$pnpm = (Get-Command pnpm -ErrorAction SilentlyContinue)
if ($pnpm) {
  Write-Host 'Using pnpm...'
  pnpm install
} else {
  Write-Host 'Using npm...'
  npm install --legacy-peer-deps
}
Write-Ok 'Frontend dependencies updated'

# 5) Summary
Write-Section 'Update Complete'
Write-Host @"

✓ Backend dependencies updated
✓ Database migrations applied
✓ Agent definitions synced
✓ Frontend dependencies updated

Next steps:
  1) Review any warnings above
  2) Start dev environment: .\scripts\dev.ps1
  3) Test your changes at http://localhost:3000

"@ -ForegroundColor Green

Set-Location $Root
