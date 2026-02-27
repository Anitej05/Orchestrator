# Docker Build Testing Documentation

This document describes the Docker containerization testing process for task 6.2.4.

## Overview

Three Docker images have been created:
1. **Orchestrator** - Main FastAPI orchestrator service
2. **Gmail Agent** - Specialized Gmail integration agent
3. **General Agent** - Fallback agent for any Composio-supported app

## Testing Prerequisites

- Docker Desktop installed and running
- Docker version 20.10+ 
- Docker Compose version 2.0+
- At least 4GB of available disk space for images

## Manual Testing Steps

### Step 1: Verify Docker is Running

```bash
# Check Docker version
docker --version

# Check Docker is running
docker info
```

Expected output: Docker version and system information

### Step 2: Test Individual Builds

#### Test Orchestrator Build

```bash
docker build -t orbimesh-orchestrator:test -f backend/Dockerfile backend
```

**Expected Result:**
- Build completes successfully
- Image size: ~1.5-2GB
- No errors during dependency installation

#### Test Gmail Agent Build

```bash
docker build -t orbimesh-gmail-agent:test -f backend/agents/gmail_agent/Dockerfile backend
```

**Expected Result:**
- Build completes successfully
- Image size: ~1.5-2GB
- Shares layers with orchestrator image (faster subsequent builds)

#### Test General Agent Build

```bash
docker build -t orbimesh-general-agent:test -f backend/agents/general_agent/Dockerfile backend
```

**Expected Result:**
- Build completes successfully
- Image size: ~1.5-2GB
- Shares layers with other images

### Step 3: Verify Built Images

```bash
docker images | grep orbimesh
```

**Expected Output:**
```
orbimesh-orchestrator    test    <image-id>    <time>    ~1.5GB
orbimesh-gmail-agent     test    <image-id>    <time>    ~1.5GB
orbimesh-general-agent   test    <image-id>    <time>    ~1.5GB
```

### Step 4: Test Image Functionality

#### Test Orchestrator Container

```bash
# Run container
docker run -d --name test-orchestrator -p 8000:8000 \
  -e DATABASE_URL=sqlite:///./test.db \
  -e COMPOSIO_API_KEY=test \
  -e CONNECTION_ENCRYPTION_KEY=test \
  orbimesh-orchestrator:test

# Wait for startup (5-10 seconds)
sleep 10

# Check health endpoint
curl http://localhost:8000/health

# Check logs
docker logs test-orchestrator

# Stop and remove
docker stop test-orchestrator
docker rm test-orchestrator
```

**Expected Result:**
- Container starts successfully
- Health endpoint returns 200 OK (may return 503 if DB not properly configured)
- Logs show FastAPI startup messages

#### Test Gmail Agent Container

```bash
# Run container
docker run -d --name test-gmail-agent -p 8001:8000 \
  -e DATABASE_URL=sqlite:///./test.db \
  -e COMPOSIO_API_KEY=test \
  orbimesh-gmail-agent:test

# Wait for startup
sleep 10

# Check health endpoint
curl http://localhost:8001/health

# Check logs
docker logs test-gmail-agent

# Stop and remove
docker stop test-gmail-agent
docker rm test-gmail-agent
```

**Expected Result:**
- Container starts successfully
- Health endpoint returns 200 OK
- Logs show "Gmail Agent" startup

#### Test General Agent Container

```bash
# Run container
docker run -d --name test-general-agent -p 8003:8000 \
  -e DATABASE_URL=sqlite:///./test.db \
  -e COMPOSIO_API_KEY=test \
  orbimesh-general-agent:test

# Wait for startup
sleep 10

# Check health endpoint
curl http://localhost:8003/health

# Check logs
docker logs test-general-agent

# Stop and remove
docker stop test-general-agent
docker rm test-general-agent
```

**Expected Result:**
- Container starts successfully
- Health endpoint returns 200 OK
- Logs show "General Fallback Agent" startup

### Step 5: Test Docker Compose

```bash
# Create .env file (copy from .env.docker.example)
cp .env.docker.example .env

# Edit .env and add test values (can use dummy values for build test)
# At minimum, set:
# COMPOSIO_API_KEY=test
# CONNECTION_ENCRYPTION_KEY=test

# Build all services
docker-compose build

# Start all services
docker-compose up -d

# Wait for services to start
sleep 30

# Check service status
docker-compose ps

# Check health endpoints
curl http://localhost:8000/health  # Orchestrator
curl http://localhost:8001/health  # Gmail Agent
curl http://localhost:8003/health  # General Agent

# Check logs
docker-compose logs orchestrator
docker-compose logs gmail_agent
docker-compose logs general_agent
docker-compose logs postgres

# Stop services
docker-compose down
```

**Expected Result:**
- All 4 services (postgres, orchestrator, gmail_agent, general_agent) show as "Up"
- Health endpoints return 200 OK
- No critical errors in logs
- Services can communicate with PostgreSQL

## Automated Testing Scripts

Two test scripts are provided for convenience:

### Linux/Mac: test-docker-builds.sh

```bash
chmod +x test-docker-builds.sh
./test-docker-builds.sh
```

### Windows: test-docker-builds.ps1

```powershell
.\test-docker-builds.ps1
```

Both scripts will:
1. Check if Docker is running
2. Build all three images
3. Report success/failure for each build
4. Display built images

## Common Issues and Solutions

### Issue: Docker Desktop Not Running

**Error:** `error during connect: ... cannot find the file specified`

**Solution:** Start Docker Desktop and wait for it to fully initialize

### Issue: Port Already in Use

**Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution:** 
- Stop the conflicting service
- Or change the port mapping in docker-compose.yml:
  ```yaml
  ports:
    - "8080:8000"  # Use 8080 instead of 8000
  ```

### Issue: Out of Disk Space

**Error:** `no space left on device`

**Solution:**
```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused containers
docker container prune
```

### Issue: Build Fails on Dependency Installation

**Error:** `ERROR: Could not find a version that satisfies the requirement...`

**Solution:**
- Check requirements.txt for version conflicts
- Ensure Python 3.11 base image is being used
- Try building with `--no-cache` flag:
  ```bash
  docker build --no-cache -t orbimesh-orchestrator:test -f backend/Dockerfile backend
  ```

### Issue: Health Check Fails

**Error:** Health check returns 503 or connection refused

**Possible Causes:**
1. Service hasn't fully started yet (wait longer)
2. Database connection issue (check DATABASE_URL)
3. Missing required environment variables
4. Port mapping incorrect

**Solution:**
- Check container logs: `docker logs <container-name>`
- Verify environment variables are set
- Ensure database is accessible

## Test Results Documentation

### Test Execution Date
[To be filled when tests are run]

### Test Environment
- OS: [Windows/Linux/Mac]
- Docker Version: [version]
- Docker Compose Version: [version]

### Build Test Results

| Image | Build Status | Build Time | Image Size | Notes |
|-------|-------------|------------|------------|-------|
| Orchestrator | ⬜ Pass / ⬜ Fail | | | |
| Gmail Agent | ⬜ Pass / ⬜ Fail | | | |
| General Agent | ⬜ Pass / ⬜ Fail | | | |

### Container Test Results

| Container | Start Status | Health Check | Logs | Notes |
|-----------|-------------|--------------|------|-------|
| Orchestrator | ⬜ Pass / ⬜ Fail | ⬜ Pass / ⬜ Fail | ⬜ OK / ⬜ Errors | |
| Gmail Agent | ⬜ Pass / ⬜ Fail | ⬜ Pass / ⬜ Fail | ⬜ OK / ⬜ Errors | |
| General Agent | ⬜ Pass / ⬜ Fail | ⬜ Pass / ⬜ Fail | ⬜ OK / ⬜ Errors | |
| PostgreSQL | ⬜ Pass / ⬜ Fail | ⬜ Pass / ⬜ Fail | ⬜ OK / ⬜ Errors | |

### Docker Compose Test Results

| Test | Status | Notes |
|------|--------|-------|
| Build all services | ⬜ Pass / ⬜ Fail | |
| Start all services | ⬜ Pass / ⬜ Fail | |
| Service health checks | ⬜ Pass / ⬜ Fail | |
| Inter-service communication | ⬜ Pass / ⬜ Fail | |
| Volume mounts | ⬜ Pass / ⬜ Fail | |

## Validation Checklist

- [ ] Docker Desktop is installed and running
- [ ] All three images build successfully without errors
- [ ] Images are reasonable size (~1.5-2GB each)
- [ ] Containers start successfully
- [ ] Health check endpoints return 200 OK
- [ ] No critical errors in container logs
- [ ] Docker Compose builds all services
- [ ] Docker Compose starts all services
- [ ] All services show as "Up" in `docker-compose ps`
- [ ] PostgreSQL is accessible from other containers
- [ ] Environment variables are properly passed to containers
- [ ] Volume mounts work correctly
- [ ] Services can be stopped and restarted cleanly

## Next Steps After Testing

Once all tests pass:

1. ✅ Mark task 6.2.4 as complete
2. ✅ Mark task 6.2 as complete
3. Document any issues encountered and solutions
4. Proceed to task 6.3 (Database migration documentation)

## Notes

- These Docker images are for **local development and testing only**
- Production deployment will use Azure Container Apps (see task 6.4)
- The docker-compose.yml uses SQLite for quick testing, but production uses PostgreSQL
- Health checks may initially fail until services fully initialize (allow 10-30 seconds)
- Images share base layers, so subsequent builds are faster
