#!/bin/bash
# Test script for Docker builds
# Run this script to verify all Docker images build successfully

set -e  # Exit on error

echo "========================================="
echo "Testing Orbimesh Docker Builds"
echo "========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Test 1: Build Orchestrator
echo "📦 Building Orchestrator image..."
docker build -t orbimesh-orchestrator:test -f backend/Dockerfile backend
if [ $? -eq 0 ]; then
    echo "✅ Orchestrator build successful"
else
    echo "❌ Orchestrator build failed"
    exit 1
fi
echo ""

# Test 2: Build Gmail Agent
echo "📦 Building Gmail Agent image..."
docker build -t orbimesh-gmail-agent:test -f backend/agents/gmail_agent/Dockerfile backend
if [ $? -eq 0 ]; then
    echo "✅ Gmail Agent build successful"
else
    echo "❌ Gmail Agent build failed"
    exit 1
fi
echo ""

# Test 3: Build Integrations Agent
echo "📦 Building Integrations Agent image..."
docker build -t orbimesh-integrations-agent:test -f backend/agents/integrations_agent/Dockerfile backend
if [ $? -eq 0 ]; then
    echo "✅ Integrations Agent build successful"
else
    echo "❌ Integrations Agent build failed"
    exit 1
fi
echo ""

# List built images
echo "========================================="
echo "Built Images:"
echo "========================================="
docker images | grep orbimesh

echo ""
echo "========================================="
echo "✅ All Docker builds completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.docker.example to .env and fill in your API keys"
echo "2. Run: docker-compose up --build"
echo "3. Access services at:"
echo "   - Orchestrator: http://localhost:8000"
echo "   - Gmail Agent: http://localhost:8001"
echo "   - Integrations Agent: http://localhost:8003"
