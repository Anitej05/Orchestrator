# Azure Cloud Architecture for Orbimesh

**Document Version:** 1.0  
**Last Updated:** 2025-02-15  
**Status:** Production Ready

## Overview

This document outlines the Azure cloud architecture for deploying Orbimesh, a multiagent orchestration system with Brain-Hands pattern. The architecture is designed for initial deployment with cost-effectiveness and simplicity as primary goals, while maintaining the ability to scale as usage grows.

## Architecture Principles

- **Microservices Architecture**: Orchestrator and agents deployed as independent containers
- **Managed Services**: Leverage Azure managed services to reduce operational overhead
- **Security First**: Secrets in Key Vault, encrypted data at rest, managed identities
- **Observability**: Comprehensive monitoring and logging with Azure Monitor
- **Cost Optimization**: Minimal resource allocation with ability to scale up as needed
- **Simplicity**: Single-instance deployment where appropriate to minimize complexity

## Azure Services

### 1. Azure Container Apps

**Purpose**: Host orchestrator and agent microservices

**Configuration**:
- **Orchestrator Service**:
  - 1 replica (can scale to 2-3 as needed)
  - 0.5 vCPU, 1GB RAM per instance
  - Auto-scale: 1-3 instances based on CPU (80% threshold)
  - Health check: `/health` endpoint every 30s
  
- **Gmail Agent Service**:
  - 1 replica (can scale to 2 as needed)
  - 0.25 vCPU, 512MB RAM per instance
  - Auto-scale: 1-2 instances based on request count
  
- **General Agent Service**:
  - 1 replica (can scale to 2 as needed)
  - 0.25 vCPU, 512MB RAM per instance
  - Auto-scale: 1-2 instances based on request count

**Why Container Apps**:
- Serverless container platform with built-in scaling
- Simpler than AKS for microservices workloads
- Built-in ingress and service discovery
- Lower cost than AKS for small-to-medium workloads
- Integrated with Azure Monitor and Application Insights

### 2. Azure Database for PostgreSQL (Flexible Server)

**Purpose**: Primary database for user connections, logs, and agent registry

**Configuration**:
- **Tier**: Burstable (B1ms) - cost-effective for initial deployment
- **Compute**: 1 vCore, 2GB RAM
- **Storage**: 32GB with auto-grow enabled
- **Backup**: 7-day retention, locally-redundant backups
- **High Availability**: Single instance (can enable zone-redundancy later)

**Database Schema**:
- `user_connections`: Encrypted Composio connection IDs
- `connection_logs`: Audit trail for all connection events
- `agent_entries`: Agent registry for service discovery

**Security**:
- SSL/TLS enforced for all connections
- Private endpoint within VNet
- Firewall rules limiting access to Container Apps only
- Managed identity authentication (no passwords)

**Migration from SQLite**:
- Export SQLite data to JSON
- Create PostgreSQL schema
- Import data with validation
- Dual-write period for safety
- Cutover with rollback plan

### 3. Azure Key Vault

**Purpose**: Secure storage for secrets and encryption keys

**Secrets Stored**:
- `COMPOSIO_API_KEY`: Composio API authentication
- `CONNECTION_ENCRYPTION_KEY`: Fernet key for encrypting connection IDs
- `DATABASE_URL`: PostgreSQL connection string
- `CLERK_SECRET_KEY`: Authentication service key
- `OPENAI_API_KEY`: LLM service key

**Access Control**:
- Managed identities for Container Apps
- RBAC policies limiting access per service
- Audit logging for all secret access
- Automatic key rotation every 90 days

**Integration**:
```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://orbimesh-kv.vault.azure.net/", credential=credential)
composio_key = client.get_secret("COMPOSIO-API-KEY").value
```

### 4. Azure Monitor + Application Insights

**Purpose**: Observability, monitoring, and alerting

**Metrics Collected**:
- Request rate, response time, error rate
- CPU and memory utilization per service
- Database connection pool metrics
- Custom metrics: task execution time, agent selection time

**Logging**:
- Structured JSON logs from all services
- Log levels: DEBUG (dev), INFO (prod)
- Retention: 30 days in Log Analytics
- Query with Kusto Query Language (KQL)

**Distributed Tracing**:
- OpenTelemetry instrumentation
- End-to-end request tracing across services
- Dependency tracking (database, Composio API)

**Alerts**:
- Error rate > 5% for 5 minutes
- Response time p95 > 2 seconds
- Database connection failures
- Container restart events
- Cost threshold exceeded

### 5. Azure Container Registry (ACR)

**Purpose**: Store and manage Docker images

**Configuration**:
- **Tier**: Basic (sufficient for small teams)
- **Geo-replication**: Disabled (single region deployment)
- **Webhook**: Trigger deployments on image push

**Images**:
- `orbimesh.azurecr.io/orchestrator:latest`
- `orbimesh.azurecr.io/gmail-agent:latest`
- `orbimesh.azurecr.io/general-agent:latest`

**Security**:
- Admin user disabled
- Managed identity access from Container Apps
- Image scanning for vulnerabilities
- Retention policy: Keep last 10 tags

### 6. Azure Front Door (Optional - Not Recommended Initially)

**Purpose**: Global load balancing, SSL termination, WAF protection

**Features**:
- SSL/TLS termination with managed certificates
- Web Application Firewall (WAF) for DDoS protection
- Caching for static assets
- Custom domain support

**Recommendation**:
- Skip for initial deployment to reduce costs
- Use Container Apps built-in ingress with custom domain
- Add later when scaling to multiple regions or requiring advanced security

### 7. Azure Virtual Network (VNet)

**Purpose**: Network isolation and security

**Configuration**:
- **Address Space**: 10.0.0.0/16
- **Subnets**:
  - Container Apps: 10.0.1.0/24
  - PostgreSQL: 10.0.2.0/24
  - Private Endpoints: 10.0.3.0/24

**Security**:
- Network Security Groups (NSGs) for traffic control
- Private endpoints for PostgreSQL and Key Vault
- No public internet access to database
- Service endpoints for Azure services


## Infrastructure Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Azure Cloud                                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Azure Front Door (Optional)                         │ │
│  │                  SSL Termination + WAF + CDN                           │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               │                                              │
│  ┌────────────────────────────▼───────────────────────────────────────────┐ │
│  │                    Azure Container Apps Environment                    │ │
│  │                                                                         │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │ │
│  │  │ Orchestrator │    │ Gmail Agent  │    │General Agent │           │ │
│  │  │   (FastAPI)  │    │   (FastAPI)  │    │  (FastAPI)   │           │ │
│  │  │              │    │              │    │              │           │ │
│  │  │ 1-3 replicas │    │ 1-2 replicas │    │ 1-2 replicas │           │ │
│  │  │ 0.5vCPU/1GB  │    │ 0.25vCPU/512MB│   │ 0.25vCPU/512MB│          │ │
│  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │ │
│  │         │                   │                   │                     │ │
│  │         │ /health           │ /health           │ /health             │ │
│  │         │ /api/brain        │ /execute          │ /execute            │ │
│  │         │ /api/hands        │                   │                     │ │
│  └─────────┼───────────────────┼───────────────────┼─────────────────────┘ │
│            │                   │                   │                       │
│            │                   │                   │                       │
│  ┌─────────▼───────────────────▼───────────────────▼─────────────────────┐ │
│  │                  Azure Database for PostgreSQL                        │ │
│  │                      (Flexible Server)                                │ │
│  │                                                                        │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │ │
│  │  │user_connections  │  │ connection_logs  │  │  agent_entries   │  │ │
│  │  │                  │  │                  │  │                  │  │ │
│  │  │ - id             │  │ - id             │  │ - id             │  │ │
│  │  │ - user_id        │  │ - user_id        │  │ - name           │  │ │
│  │  │ - app_slug       │  │ - app_slug       │  │ - capabilities   │  │ │
│  │  │ - connection_id  │  │ - event_type     │  │ - base_url       │  │ │
│  │  │   (encrypted)    │  │ - status         │  │ - is_active      │  │ │
│  │  │ - status         │  │ - timestamp      │  │                  │  │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │ │
│  │                                                                        │ │
│  │  1 vCore, 2GB RAM, 32GB Storage                                       │ │
│  │  Single instance, 7-day backups                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Azure Key Vault                                │ │
│  │                                                                         │ │
│  │  Secrets:                                                              │ │
│  │  - COMPOSIO_API_KEY                                                    │ │
│  │  - CONNECTION_ENCRYPTION_KEY                                           │ │
│  │  - DATABASE_URL                                                        │ │
│  │  - CLERK_SECRET_KEY                                                    │ │
│  │  - OPENAI_API_KEY                                                      │ │
│  │                                                                         │ │
│  │  Access: Managed Identity (RBAC)                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              Azure Monitor + Application Insights                      │ │
│  │                                                                         │ │
│  │  - Request tracing (OpenTelemetry)                                     │ │
│  │  - Performance metrics (CPU, memory, response time)                    │ │
│  │  - Error logging and alerting                                          │ │
│  │  - Custom metrics (task execution time)                                │ │
│  │  - Log Analytics (30-day retention)                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   Azure Container Registry                             │ │
│  │                                                                         │ │
│  │  Images:                                                               │ │
│  │  - orbimesh.azurecr.io/orchestrator:latest                            │ │
│  │  - orbimesh.azurecr.io/gmail-agent:latest                             │ │
│  │  - orbimesh.azurecr.io/general-agent:latest                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Azure Virtual Network                             │ │
│  │                                                                         │ │
│  │  Address Space: 10.0.0.0/16                                            │ │
│  │  - Container Apps Subnet: 10.0.1.0/24                                  │ │
│  │  - PostgreSQL Subnet: 10.0.2.0/24                                      │ │
│  │  - Private Endpoints: 10.0.3.0/24                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

External Dependencies:
  - Composio API (composio.dev)
  - Clerk Authentication (clerk.dev)
  - OpenAI API (openai.com)
```

## Service Communication Flow

```
User Request
    │
    ▼
Azure Front Door (SSL/WAF)
    │
    ▼
Orchestrator Container App
    │
    ├─► Brain (LLM Decision Engine)
    │   └─► Selects appropriate agent
    │
    ├─► Hands (Execution Dispatcher)
    │   └─► Calls agent via HTTP
    │
    ▼
Agent Container App (Gmail/General)
    │
    ├─► Get Connection from PostgreSQL
    │   └─► Decrypt connection_id
    │
    ├─► Get Tools from Composio API
    │   └─► Cache tools (General Agent)
    │
    ├─► Execute Tool via Composio
    │   └─► Return result
    │
    ▼
Response to User
```


## Service Sizing and Scaling Strategy

### Orchestrator Service

**Initial Sizing**:
- **Replicas**: 1 (can scale to 2-3 as needed)
- **CPU**: 0.5 vCPU per instance
- **Memory**: 1GB per instance
- **Rationale**: Brain component uses LLM API calls (I/O bound), Hands is lightweight dispatcher

**Auto-Scaling Rules**:
- **Scale Out**: When CPU > 80% for 5 minutes, add 1 replica (max 3)
- **Scale In**: When CPU < 30% for 10 minutes, remove 1 replica (min 1)
- **Request-Based**: When concurrent requests > 30, add 1 replica

**Performance Targets**:
- Response time p95: < 3 seconds
- Throughput: 20 requests/second
- Error rate: < 2%

**Bottleneck Analysis**:
- LLM API latency is primary bottleneck (500-1500ms)
- Database queries are fast (< 50ms)
- Agent HTTP calls add 100-300ms
- Total expected latency: 600-1800ms per request

### Gmail Agent Service

**Initial Sizing**:
- **Replicas**: 1 (can scale to 2 as needed)
- **CPU**: 0.25 vCPU per instance
- **Memory**: 512MB per instance
- **Rationale**: Stateless service, mostly I/O bound (Composio API calls)

**Auto-Scaling Rules**:
- **Scale Out**: When request count > 50/minute, add 1 replica (max 2)
- **Scale In**: When request count < 15/minute, remove 1 replica (min 1)
- **CPU-Based**: When CPU > 85% for 5 minutes, add 1 replica

**Performance Targets**:
- Response time p95: < 1.5 seconds
- Throughput: 100 requests/minute
- Tool execution success rate: > 90%

**Rate Limiting**:
- 100 requests per minute per user (prevent API quota exhaustion)
- 500 requests per minute per service instance
- Retry with exponential backoff on rate limit errors

### General Agent Service

**Initial Sizing**:
- **Replicas**: 1 (can scale to 2 as needed)
- **CPU**: 0.25 vCPU per instance
- **Memory**: 512MB per instance
- **Rationale**: Similar to Gmail agent, with additional caching overhead

**Auto-Scaling Rules**:
- **Scale Out**: When request count > 40/minute, add 1 replica (max 2)
- **Scale In**: When request count < 10/minute, remove 1 replica (min 1)
- **Memory-Based**: When memory > 400MB, add 1 replica (cache pressure)

**Performance Targets**:
- Response time p95: < 2 seconds (includes tool discovery)
- Cache hit rate: > 75%
- Tool execution success rate: > 85% (lower than Gmail due to diverse apps)

**Cache Strategy**:
- TTL: 1 hour per app/user combination
- Max cache size: 100MB per instance
- Eviction: LRU when memory pressure detected
- Invalidation: On user disconnect event

### PostgreSQL Database

**Initial Sizing**:
- **Tier**: Burstable (B1ms)
- **vCores**: 1
- **Memory**: 2GB
- **Storage**: 32GB (auto-grow enabled)
- **IOPS**: 640 (burstable to 3200)

**Scaling Strategy**:
- **Vertical Scaling**: Upgrade to B2s (2 vCores, 4GB) when CPU > 75%
- **Storage Scaling**: Auto-grow in 32GB increments
- **Read Replicas**: Consider if read load > 500 queries/second

**Performance Targets**:
- Query latency p95: < 100ms
- Connection pool: 10 connections per service
- Max connections: 50 (sufficient for initial deployment)

**Optimization**:
- Indexes on `user_id`, `app_slug`, `status` columns
- Connection pooling with PgBouncer
- Query optimization for frequent lookups
- Partitioning `connection_logs` by month (if > 1M rows)

### Capacity Planning

**Expected Load (Initial Deployment - 10-20 Users)**:

| Metric | Value | Calculation |
|--------|-------|-------------|
| Requests/second | 2 | 15 users × 0.13 req/sec average |
| Peak requests/second | 10 | 5x average for burst traffic |
| Database queries/second | 6 | 3 queries per request average |
| Composio API calls/second | 3 | 1.5 calls per request average |
| Storage growth | 100MB/month | 10KB per connection log × 10K logs |

**Resource Utilization at Expected Load**:

| Service | CPU | Memory | Replicas | Headroom |
|---------|-----|--------|----------|----------|
| Orchestrator | 20% | 40% | 1 | 80% |
| Gmail Agent | 15% | 30% | 1 | 85% |
| General Agent | 15% | 35% | 1 | 85% |
| PostgreSQL | 25% | 40% | 1 | 75% |

**Scaling Triggers**:
- At 30 users: Scale orchestrator to 2 replicas
- At 50 users: Scale agents to 2 replicas each
- At 100 users: Upgrade PostgreSQL to B2s (2 vCores, 4GB)

### High Availability Strategy

**Service Level Objectives (SLOs)**:
- **Availability**: 99% uptime (7 hours downtime/month)
- **Durability**: 99.9% data durability (PostgreSQL backups)
- **Recovery Time Objective (RTO)**: 30 minutes
- **Recovery Point Objective (RPO)**: 1 hour

**Redundancy**:
- Single replica for services initially (cost optimization)
- Auto-scaling enabled to add replicas under load
- PostgreSQL single instance with automated backups
- Health checks every 30 seconds
- Automatic restart on failure

**Failure Scenarios**:

| Failure | Impact | Recovery | Time |
|---------|--------|----------|------|
| Single container crash | Brief interruption | Automatic restart | 1 min |
| Entire service down | Service unavailable | Deploy from ACR | 3 min |
| Database failure | Full outage | Restore from backup | 15 min |
| Composio API down | Tool execution fails | Retry with backoff, user notification | N/A |

**Monitoring and Alerting**:
- Health check failures → PagerDuty alert
- Error rate > 5% → Slack notification
- Response time > 2s → Email alert
- Database CPU > 80% → Auto-scale trigger
- Cost > $250/month → Budget alert


## Cost Estimation

### Monthly Cost Breakdown (Initial Deployment)

| Service | Configuration | Monthly Cost (USD) |
|---------|--------------|-------------------|
| **Azure Container Apps** | | |
| - Orchestrator | 1-3 instances × 0.5 vCPU, 1GB RAM | $20 |
| - Gmail Agent | 1-2 instances × 0.25 vCPU, 512MB RAM | $10 |
| - General Agent | 1-2 instances × 0.25 vCPU, 512MB RAM | $10 |
| **Azure PostgreSQL** | Flexible Server, B1ms, 32GB storage | $30 |
| **Azure Key Vault** | Standard tier, 500 operations/month | $3 |
| **Azure Monitor** | 2GB ingestion, 30-day retention | $8 |
| **Azure Container Registry** | Basic tier | $5 |
| **Azure Virtual Network** | Standard VNet | $5 |
| **Bandwidth** | 20GB egress | $2 |
| **Total** | | **$93/month** |

### Cost Optimization Strategies

**1. Right-Sizing**:
- Start with single replicas: Saves $50/month vs multi-replica
- Use Burstable PostgreSQL tier: Saves $90/month vs General Purpose
- Minimal monitoring ingestion: Saves $7/month vs full logging

**2. Development Environment**:
- Use Azure Container Instances for dev/test: Save 60% on compute
- Shared PostgreSQL instance for dev/staging: Save $30/month
- Disable monitoring in development: Save $8/month

**3. Storage Optimization**:
- Archive old connection logs to Azure Blob Storage: $0.01/GB vs $0.12/GB
- Delete logs older than 30 days: Reduce storage by 70%

**4. Monitoring Optimization**:
- Sample logs at 20% for non-critical services: Save $5/month
- Reduce retention to 7 days for debug logs: Save $3/month

**Projected Costs at Scale**:

| Users | Monthly Cost | Cost per User |
|-------|-------------|---------------|
| 10-20 | $93 | $6.20 |
| 50 | $140 | $2.80 |
| 100 | $210 | $2.10 |
| 500 | $450 | $0.90 |

**Cost Monitoring**:
- Set budget alerts at $100, $150, $200
- Weekly cost reports to engineering team
- Monthly cost review

## Security Architecture

### Identity and Access Management

**Managed Identities**:
- Each Container App has a system-assigned managed identity
- No passwords or connection strings in code
- Automatic credential rotation by Azure

**RBAC Policies**:
- Orchestrator: Read access to Key Vault, read/write to PostgreSQL
- Agents: Read access to Key Vault, read/write to PostgreSQL
- CI/CD Pipeline: Push access to ACR, deploy access to Container Apps
- Developers: Read-only access to logs, no access to secrets

**Network Security**:
- Private endpoints for PostgreSQL and Key Vault
- No public internet access to database
- NSG rules limiting traffic between subnets
- DDoS protection with Azure Front Door

### Data Security

**Encryption at Rest**:
- PostgreSQL: Transparent Data Encryption (TDE) enabled
- Key Vault: Hardware Security Module (HSM) backed keys
- Connection IDs: Fernet encryption with 256-bit keys
- Backups: Encrypted with Microsoft-managed keys

**Encryption in Transit**:
- TLS 1.2+ for all connections
- Certificate pinning for Composio API
- HTTPS only for all endpoints
- Mutual TLS for service-to-service communication

**Data Classification**:
- **Highly Sensitive**: Connection IDs, API keys, user credentials
- **Sensitive**: User emails, task descriptions, logs
- **Public**: Agent capabilities, service health status

**Compliance**:
- GDPR: User data deletion within 30 days of request
- SOC 2: Audit logs for all data access
- HIPAA: Not applicable (no health data)

### Secrets Management

**Key Vault Best Practices**:
- Separate Key Vaults for dev/staging/prod
- Soft-delete enabled (90-day retention)
- Purge protection enabled
- Access policies reviewed quarterly
- Secrets rotated every 90 days

**Secret Rotation**:
```python
# Automatic secret rotation with zero downtime
def rotate_composio_key():
    # 1. Generate new key in Composio
    new_key = composio_admin.create_api_key()
    
    # 2. Store new key in Key Vault with version
    kv_client.set_secret("COMPOSIO-API-KEY", new_key)
    
    # 3. Update Container Apps environment variables
    # (Container Apps automatically pick up new version)
    
    # 4. Wait 5 minutes for all instances to refresh
    time.sleep(300)
    
    # 5. Revoke old key in Composio
    composio_admin.revoke_api_key(old_key)
```

## Deployment Strategy

### CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
name: Deploy to Azure
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Login to Azure
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Build and push Docker images
        run: |
          az acr build --registry orbimesh --image orchestrator:${{ github.sha }} ./backend
          az acr build --registry orbimesh --image gmail-agent:${{ github.sha }} ./backend/agents/gmail_agent
          az acr build --registry orbimesh --image general-agent:${{ github.sha }} ./backend/agents/general_agent
      
      - name: Deploy to Container Apps
        run: |
          az containerapp update \
            --name orchestrator \
            --resource-group orbimesh-prod \
            --image orbimesh.azurecr.io/orchestrator:${{ github.sha }}
          
          az containerapp update \
            --name gmail-agent \
            --resource-group orbimesh-prod \
            --image orbimesh.azurecr.io/gmail-agent:${{ github.sha }}
      
      - name: Run database migrations
        run: |
          az containerapp exec \
            --name orchestrator \
            --resource-group orbimesh-prod \
            --command "python migrations/run_migrations.py"
      
      - name: Run smoke tests
        run: |
          curl -f https://orchestrator.orbimesh.com/health || exit 1
          curl -f https://gmail-agent.orbimesh.com/health || exit 1
      
      - name: Notify team
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Deployment failed: ${{ github.sha }}"
            }
```

### Blue-Green Deployment

**Strategy**:
1. Deploy new version to "green" environment (separate Container Apps)
2. Run smoke tests against green environment
3. Gradually shift traffic: 10% → 50% → 100%
4. Monitor error rates and response times
5. Rollback if error rate > 5% or response time > 2s
6. Keep blue environment for 24 hours, then decommission

**Traffic Shifting**:
```bash
# Shift 10% traffic to green
az containerapp ingress traffic set \
  --name orchestrator \
  --resource-group orbimesh-prod \
  --revision-weight blue=90 green=10

# Monitor for 15 minutes, then shift to 50%
az containerapp ingress traffic set \
  --name orchestrator \
  --resource-group orbimesh-prod \
  --revision-weight blue=50 green=50

# If all good, shift to 100%
az containerapp ingress traffic set \
  --name orchestrator \
  --resource-group orbimesh-prod \
  --revision-weight blue=0 green=100
```

### Rollback Strategy

**Automatic Rollback Triggers**:
- Error rate > 10% for 2 minutes
- Response time p95 > 5 seconds for 5 minutes
- Health check failures > 50%
- Database connection failures

**Manual Rollback**:
```bash
# Rollback to previous revision
az containerapp revision list \
  --name orchestrator \
  --resource-group orbimesh-prod

az containerapp ingress traffic set \
  --name orchestrator \
  --resource-group orbimesh-prod \
  --revision-weight <previous-revision>=100
```

**Database Rollback**:
- Point-in-time restore to 5 minutes before deployment
- Restore from geo-redundant backup
- Test restore procedure monthly


## Database Migration Plan

### Phase 1: Preparation (Week 1)

**Tasks**:
1. Audit current SQLite database schema
2. Create PostgreSQL schema with indexes
3. Write migration scripts
4. Test migration on copy of production data
5. Document rollback procedure

**Schema Differences**:
```sql
-- SQLite (current)
CREATE TABLE user_connections (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    app_slug TEXT NOT NULL,
    connection_id TEXT NOT NULL,  -- Encrypted
    status TEXT NOT NULL,
    auth_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- PostgreSQL (target)
CREATE TABLE user_connections (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    app_slug VARCHAR(100) NOT NULL,
    connection_id TEXT NOT NULL,  -- Encrypted
    status VARCHAR(50) NOT NULL,
    auth_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified TIMESTAMP,
    connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    app_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, app_slug)
);

CREATE INDEX idx_user_app ON user_connections(user_id, app_slug);
CREATE INDEX idx_status ON user_connections(status);
```

### Phase 2: Dual-Write (Week 2)

**Implementation**:
```python
class DualWriteSession:
    """Write to both SQLite and PostgreSQL during migration."""
    
    def __init__(self, sqlite_session, postgres_session):
        self.sqlite = sqlite_session
        self.postgres = postgres_session
        self.errors = []
    
    def add(self, obj):
        """Add object to both databases."""
        try:
            self.sqlite.add(obj)
            self.postgres.add(obj)
        except Exception as e:
            self.errors.append(f"Dual-write error: {e}")
            # Continue with SQLite only
            self.sqlite.add(obj)
    
    def commit(self):
        """Commit to both databases."""
        try:
            self.sqlite.commit()
            self.postgres.commit()
        except Exception as e:
            self.errors.append(f"Dual-commit error: {e}")
            self.sqlite.rollback()
            self.postgres.rollback()
            raise
```

**Monitoring**:
- Compare row counts between SQLite and PostgreSQL every hour
- Alert on discrepancies > 1%
- Log all dual-write errors
- Daily reconciliation job to sync missing records

### Phase 3: Data Migration (Week 3)

**Migration Script**:
```python
import sqlite3
import psycopg2
import json
from datetime import datetime

def migrate_data(sqlite_path: str, postgres_url: str):
    """Migrate data from SQLite to PostgreSQL."""
    
    # 1. Export from SQLite
    print("Exporting from SQLite...")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()
    
    cursor.execute("SELECT * FROM user_connections")
    connections = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM connection_logs")
    logs = [dict(row) for row in cursor.fetchall()]
    
    sqlite_conn.close()
    
    print(f"Exported {len(connections)} connections, {len(logs)} logs")
    
    # 2. Import to PostgreSQL
    print("Importing to PostgreSQL...")
    pg_conn = psycopg2.connect(postgres_url)
    pg_cursor = pg_conn.cursor()
    
    # Import connections
    for conn in connections:
        pg_cursor.execute("""
            INSERT INTO user_connections 
            (id, user_id, app_slug, connection_id, status, auth_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            conn["id"], conn["user_id"], conn["app_slug"],
            conn["connection_id"], conn["status"], conn["auth_timestamp"]
        ))
    
    # Import logs
    for log in logs:
        pg_cursor.execute("""
            INSERT INTO connection_logs 
            (user_id, app_slug, connection_id, event_type, status, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            log["user_id"], log["app_slug"], log.get("connection_id"),
            log["event_type"], log["status"], log["timestamp"]
        ))
    
    pg_conn.commit()
    pg_conn.close()
    
    print("Migration complete!")
    
    # 3. Verify data integrity
    verify_migration(sqlite_path, postgres_url)

def verify_migration(sqlite_path: str, postgres_url: str):
    """Verify data was migrated correctly."""
    
    sqlite_conn = sqlite3.connect(sqlite_path)
    pg_conn = psycopg2.connect(postgres_url)
    
    # Count rows
    sqlite_count = sqlite_conn.execute("SELECT COUNT(*) FROM user_connections").fetchone()[0]
    pg_count = pg_conn.cursor().execute("SELECT COUNT(*) FROM user_connections").fetchone()[0]
    
    assert sqlite_count == pg_count, f"Row count mismatch: {sqlite_count} vs {pg_count}"
    
    # Verify sample records
    sqlite_sample = sqlite_conn.execute("SELECT * FROM user_connections LIMIT 10").fetchall()
    for row in sqlite_sample:
        pg_row = pg_conn.cursor().execute(
            "SELECT * FROM user_connections WHERE id = %s", (row[0],)
        ).fetchone()
        assert pg_row is not None, f"Missing record: {row[0]}"
    
    print("Verification passed!")
```

### Phase 4: Cutover (Week 4)

**Cutover Steps**:
1. **Announce maintenance window**: 2 AM - 4 AM (low traffic)
2. **Stop writes to SQLite**: Set application to read-only mode
3. **Final sync**: Run migration script one last time
4. **Verify data**: Run verification script
5. **Switch DATABASE_URL**: Update environment variable to PostgreSQL
6. **Restart services**: Rolling restart of all Container Apps
7. **Monitor**: Watch error rates and response times for 1 hour
8. **Backup SQLite**: Keep SQLite file for 7 days

**Rollback Plan**:
- If error rate > 5%: Switch DATABASE_URL back to SQLite
- If data corruption detected: Restore PostgreSQL from backup
- If performance issues: Optimize queries, add indexes

### Phase 5: Cleanup (Week 5)

**Tasks**:
1. Remove dual-write code
2. Archive SQLite database to Azure Blob Storage
3. Update documentation
4. Delete SQLite file after 30 days
5. Celebrate! 🎉

## Monitoring and Observability

### Key Metrics

**Application Metrics**:
- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate (%)
- Task execution time (seconds)
- Agent selection time (milliseconds)
- LLM API latency (milliseconds)

**Infrastructure Metrics**:
- CPU utilization (%)
- Memory utilization (%)
- Network throughput (MB/s)
- Disk I/O (IOPS)
- Container restart count

**Business Metrics**:
- Active users (count)
- Tasks executed (count)
- Tool executions (count)
- Connection success rate (%)
- User retention (%)

### Logging Strategy

**Log Levels**:
- **DEBUG**: Development only, verbose output
- **INFO**: Normal operations, user actions
- **WARNING**: Recoverable errors, rate limits
- **ERROR**: Unrecoverable errors, exceptions
- **CRITICAL**: System failures, data corruption

**Structured Logging**:
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_event(event_type: str, user_id: str, details: dict):
    """Log structured event."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "service": "orchestrator",
        "environment": os.getenv("ENVIRONMENT", "production"),
        **details
    }
    logger.info(json.dumps(log_entry))

# Usage
log_event("task_executed", "user_123", {
    "task_name": "Send email",
    "agent": "gmail",
    "execution_time_ms": 1234,
    "success": True
})
```

**Log Aggregation**:
- All logs sent to Azure Monitor Log Analytics
- Retention: 30 days for INFO, 90 days for ERROR
- Query with Kusto Query Language (KQL)
- Export to Azure Blob Storage for long-term archival

### Alerting Rules

**Critical Alerts** (PagerDuty):
- Error rate > 10% for 2 minutes
- All services down
- Database connection failures
- Security breach detected

**Warning Alerts** (Slack):
- Error rate > 5% for 5 minutes
- Response time p95 > 2 seconds
- CPU > 80% for 10 minutes
- Disk space < 20%

**Info Alerts** (Email):
- Daily summary report
- Weekly cost report
- Monthly uptime report

### Dashboards

**Operations Dashboard**:
- Service health status
- Request rate and response time
- Error rate by service
- Active users and tasks
- Infrastructure utilization

**Business Dashboard**:
- Daily active users
- Tasks executed per day
- Most used agents
- Tool execution success rate
- User retention and churn

**Cost Dashboard**:
- Daily cost breakdown by service
- Cost per user
- Cost trends over time
- Budget vs actual

## Disaster Recovery

### Backup Strategy

**Database Backups**:
- Automated daily backups (7-day retention)
- Geo-redundant backups (secondary region)
- Point-in-time restore (up to 7 days)
- Monthly backup testing

**Configuration Backups**:
- Infrastructure as Code (Terraform/Bicep) in Git
- Container images in ACR (10 versions retained)
- Key Vault secrets versioned
- Environment variables in Azure App Configuration

### Recovery Procedures

**Scenario 1: Single Service Failure**
- **Detection**: Health check failure, automatic restart
- **Recovery**: Container Apps auto-restart failed instances
- **RTO**: 30 seconds
- **RPO**: 0 (no data loss)

**Scenario 2: Database Corruption**
- **Detection**: Data validation errors, query failures
- **Recovery**: Point-in-time restore to 5 minutes before corruption
- **RTO**: 15 minutes
- **RPO**: 5 minutes

**Scenario 3: Region Outage**
- **Detection**: All services unreachable, Azure status page
- **Recovery**: Manual failover to secondary region
- **RTO**: 1 hour
- **RPO**: 15 minutes (last backup)

**Scenario 4: Complete Data Loss**
- **Detection**: Database deleted, backups corrupted
- **Recovery**: Restore from geo-redundant backup
- **RTO**: 4 hours
- **RPO**: 24 hours (last geo-backup)

### Business Continuity Plan

**Communication Plan**:
1. Detect incident (automated monitoring)
2. Notify on-call engineer (PagerDuty)
3. Create incident channel (Slack)
4. Update status page (statuspage.io)
5. Notify customers (email)
6. Post-mortem after resolution

**Incident Response Team**:
- **Incident Commander**: Coordinates response
- **Technical Lead**: Implements fixes
- **Communications Lead**: Updates stakeholders
- **Customer Support**: Handles user inquiries

## Conclusion

This Azure architecture provides a cost-effective and scalable foundation for deploying Orbimesh for initial customer testing and feedback. Key highlights:

- **Cost-Effective**: $93/month for initial deployment with 10-20 users
- **Scalable**: Auto-scaling from 1 to 6+ instances based on load
- **Reliable**: 99% uptime SLO with automated backups and health checks
- **Security**: Managed identities, encrypted data, private networking
- **Observability**: Comprehensive monitoring, logging, and alerting
- **Growth Path**: Clear scaling strategy as user base grows

**Next Steps**:
1. Review and approve architecture
2. Create Azure resources (Terraform/Bicep)
3. Implement CI/CD pipeline
4. Migrate database from SQLite to PostgreSQL
5. Deploy to Azure environment
6. Monitor performance and gather feedback
7. Scale resources as usage grows

**References**:
- [Azure Container Apps Documentation](https://learn.microsoft.com/en-us/azure/container-apps/)
- [Azure PostgreSQL Best Practices](https://learn.microsoft.com/en-us/azure/postgresql/)
- [Azure Key Vault Security](https://learn.microsoft.com/en-us/azure/key-vault/)
- [Azure Monitor Overview](https://learn.microsoft.com/en-us/azure/azure-monitor/)

---

**Document Maintained By**: Platform Engineering Team  
**Last Review Date**: 2025-02-15  
**Next Review Date**: 2025-03-15
