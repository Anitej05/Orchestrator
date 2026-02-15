# Docker Deployment Guide

This guide explains how to run Orbimesh using Docker and Docker Compose for local development and testing.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Required API keys (Composio, OpenAI, etc.)

## Quick Start

### 1. Set Up Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.docker.example .env
```

Edit `.env` and add your actual API keys:
- `COMPOSIO_API_KEY`: Your Composio API key
- `CONNECTION_ENCRYPTION_KEY`: Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- `OPENAI_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`: Your LLM API keys
- `CLERK_SECRET_KEY`: Your Clerk authentication key

### 2. Build and Start Services

Build all Docker images and start the services:

```bash
docker-compose up --build
```

Or run in detached mode:

```bash
docker-compose up -d --build
```

### 3. Verify Services

Check that all services are running:

```bash
docker-compose ps
```

You should see:
- `orbimesh-postgres` (PostgreSQL database)
- `orbimesh-orchestrator` (Main orchestrator)
- `orbimesh-gmail-agent` (Gmail agent)
- `orbimesh-general-agent` (General fallback agent)

### 4. Access Services

- **Orchestrator API**: http://localhost:8000
- **Orchestrator Docs**: http://localhost:8000/docs
- **Gmail Agent**: http://localhost:8001
- **General Agent**: http://localhost:8003

### 5. Check Health

Verify all services are healthy:

```bash
# Orchestrator
curl http://localhost:8000/health

# Gmail Agent
curl http://localhost:8001/health

# General Agent
curl http://localhost:8003/health
```

## Service Architecture

```
┌─────────────────────────────────────────┐
│         Docker Compose Network          │
│                                         │
│  ┌──────────────┐    ┌──────────────┐  │
│  │ Orchestrator │◄───┤  PostgreSQL  │  │
│  │   :8000      │    │   :5432      │  │
│  └──────┬───────┘    └──────────────┘  │
│         │                               │
│    ┌────┴────┐                          │
│    │         │                          │
│  ┌─▼──────┐ ┌▼──────────┐              │
│  │ Gmail  │ │  General  │              │
│  │ Agent  │ │  Agent    │              │
│  │ :8001  │ │  :8003    │              │
│  └────────┘ └───────────┘              │
└─────────────────────────────────────────┘
```

## Managing Services

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f orchestrator
docker-compose logs -f gmail_agent
docker-compose logs -f general_agent
docker-compose logs -f postgres
```

### Stop Services

```bash
docker-compose stop
```

### Restart Services

```bash
docker-compose restart
```

### Stop and Remove Containers

```bash
docker-compose down
```

### Stop and Remove Containers + Volumes (⚠️ Deletes database data)

```bash
docker-compose down -v
```

## Database Management

### Access PostgreSQL

```bash
docker-compose exec postgres psql -U orbimesh -d orbimesh
```

### Run Database Migrations

```bash
docker-compose exec orchestrator alembic upgrade head
```

### Backup Database

```bash
docker-compose exec postgres pg_dump -U orbimesh orbimesh > backup.sql
```

### Restore Database

```bash
docker-compose exec -T postgres psql -U orbimesh orbimesh < backup.sql
```

## Development Workflow

### Rebuild After Code Changes

```bash
# Rebuild specific service
docker-compose up -d --build orchestrator

# Rebuild all services
docker-compose up -d --build
```

### Hot Reload (Development Mode)

For development with hot reload, you can mount the code as a volume. Add to `docker-compose.yml`:

```yaml
services:
  orchestrator:
    volumes:
      - ./backend:/app
```

Then restart:

```bash
docker-compose restart orchestrator
```

## Troubleshooting

### Service Won't Start

Check logs for errors:
```bash
docker-compose logs orchestrator
```

### Database Connection Issues

Verify PostgreSQL is healthy:
```bash
docker-compose ps postgres
docker-compose logs postgres
```

### Port Already in Use

If ports 8000, 8001, 8003, or 5432 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
services:
  orchestrator:
    ports:
      - "8080:8000"  # Change 8080 to any available port
```

### Health Check Failures

Check if the service is responding:
```bash
docker-compose exec orchestrator curl http://localhost:8000/health
```

### Clear Everything and Start Fresh

```bash
# Stop and remove all containers, networks, and volumes
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Rebuild and start
docker-compose up --build
```

## Production Considerations

This Docker Compose setup is designed for **local development and testing only**. For production deployment:

1. **Use Azure Container Apps or AKS** (see Azure deployment documentation)
2. **Use Azure PostgreSQL** instead of containerized PostgreSQL
3. **Store secrets in Azure Key Vault** instead of .env files
4. **Enable SSL/TLS** for all connections
5. **Configure proper resource limits** (CPU, memory)
6. **Set up monitoring** with Azure Monitor/Application Insights
7. **Implement proper backup strategy** for database
8. **Use managed identities** instead of API keys where possible

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `COMPOSIO_API_KEY` | Composio API key | Yes |
| `CONNECTION_ENCRYPTION_KEY` | Fernet encryption key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `GROQ_API_KEY` | Groq API key | Optional |
| `CEREBRAS_API_KEY` | Cerebras API key | Optional |
| `CLERK_SECRET_KEY` | Clerk authentication key | Yes |
| `ENVIRONMENT` | Environment name (development/production) | No |

## Next Steps

- Review [Azure Deployment Documentation](docs/azure-deployment.md) for production deployment
- Check [Database Migration Guide](docs/database-migration.md) for SQLite to PostgreSQL migration
- See [Monitoring Setup](docs/monitoring.md) for observability configuration
