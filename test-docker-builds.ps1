# Test script for Docker builds (PowerShell)
# Run this script to verify all Docker images build successfully

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Testing Orbimesh Docker Builds" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 1: Build Orchestrator
Write-Host "📦 Building Orchestrator image..." -ForegroundColor Yellow
try {
    docker build -t orbimesh-orchestrator:test -f backend/Dockerfile backend
    Write-Host "✅ Orchestrator build successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Orchestrator build failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 2: Build Gmail Agent
Write-Host "📦 Building Gmail Agent image..." -ForegroundColor Yellow
try {
    docker build -t orbimesh-gmail-agent:test -f backend/agents/gmail_agent/Dockerfile backend
    Write-Host "✅ Gmail Agent build successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Gmail Agent build failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 3: Build Integrations Agent
Write-Host "📦 Building Integrations Agent image..." -ForegroundColor Yellow
try {
    docker build -t orbimesh-integrations-agent:test -f backend/agents/integrations_agent/Dockerfile backend
    Write-Host "✅ Integrations Agent build successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Integrations Agent build failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# List built images
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Built Images:" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
docker images | Select-String "orbimesh"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "✅ All Docker builds completed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Copy .env.docker.example to .env and fill in your API keys"
Write-Host "2. Run: docker-compose up --build"
Write-Host "3. Access services at:"
Write-Host "   - Orchestrator: http://localhost:8000"
Write-Host "   - Gmail Agent: http://localhost:8001"
Write-Host "   - Integrations Agent: http://localhost:8003"
