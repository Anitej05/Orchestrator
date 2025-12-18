# Orbimesh developer run script (Windows PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1

$ErrorActionPreference = 'Stop'

function Start-Terminal($title, $workDir, $command) {
  $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("Set-Location `"$workDir`"; $command"))
  Start-Process powershell -ArgumentList "-NoExit -EncodedCommand $encoded" -WindowStyle Normal -WorkingDirectory $workDir -Verb RunAs -PassThru | Out-Null
}

# Paths
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$Py = Join-Path $Backend '.venv\Scripts\python.exe'

# 1) Start backend API
Start-Terminal 'Orbimesh Backend' $Backend "$Py -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

# 2) Start microservice agents
$agentsBat = Join-Path $Backend 'start_agents.bat'
if (Test-Path $agentsBat) {
  Start-Terminal 'Agents' $Backend ".\start_agents.bat"
}

# 3) Start frontend
$pm = if (Get-Command pnpm -ErrorAction SilentlyContinue) { 'pnpm' } else { 'npm run' }
$cmd = if ($pm -eq 'pnpm') { 'pnpm dev' } else { 'npm run dev' }
Start-Terminal 'Frontend' $Frontend $cmd

Write-Host "Launched backend, agents, and frontend in separate terminals." -ForegroundColor Green
