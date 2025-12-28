# Zoho Books Agent Guide

## Overview

The Zoho Books Agent provides comprehensive integration with Zoho Books API, specifically designed for pharmaceutical companies. It automates invoice creation, management, customer handling, and includes pharmaceutical-specific fields like batch numbers, expiry dates, and regulatory compliance tracking.

## Prerequisites

1. Zoho Books account with API access
2. OAuth 2.0 credentials (client_id, client_secret)
3. Organization ID from Zoho Books
4. Python 3.9+ with required dependencies

## Configuration

### Step 1: Zoho Books API Setup

1. **Create Zoho API Application:**
   - Visit https://accounts.zoho.in/developerconsole
   - Create a new application
   - Select "Server-based Applications"
   - Add scopes: `ZohoBooks.fullaccess.all`
   - Note down `client_id` and `client_secret`

2. **Get Organization ID:**
   - Log into Zoho Books
   - Go to Settings → Organization
   - Copy the Organization ID from the URL or settings page

3. **Update temp.json:**
   ```json
   {
     "client_id": "YOUR_CLIENT_ID",
     "client_secret": "YOUR_CLIENT_SECRET",
     "api_domain": "https://www.zohoapis.in",
     "organization_id": "YOUR_ORGANIZATION_ID"
   }
   ```

### Step 2: OAuth Token Generation

**Option A: Using Zoho OAuth Playground (Recommended for Testing)**
1. Visit https://api-console.zoho.in/
2. Select your application
3. Generate access token and refresh token
4. Use the `/oauth/initialize` endpoint to set tokens

**Option B: Manual OAuth Flow**
1. Construct OAuth URL:
   ```
   https://accounts.zoho.in/oauth/v2/auth?scope=ZohoBooks.fullaccess.all&client_id=YOUR_CLIENT_ID&response_type=code&access_type=offline&redirect_uri=YOUR_REDIRECT_URI
   ```
2. Authorize and get authorization code
3. Exchange code for tokens:
   ```bash
   curl -X POST https://accounts.zoho.in/oauth/v2/token \
     -d "grant_type=authorization_code" \
     -d "client_id=YOUR_CLIENT_ID" \
     -d "client_secret=YOUR_CLIENT_SECRET" \
     -d "redirect_uri=YOUR_REDIRECT_URI" \
     -d "code=AUTHORIZATION_CODE"
   ```

### Step 3: Initialize OAuth Tokens

Once you have access_token and refresh_token:

```bash
curl -X POST http://localhost:8050/oauth/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "YOUR_ACCESS_TOKEN",
    "refresh_token": "YOUR_REFRESH_TOKEN",
    "expires_in": 3600
  }'
```

Or use the agent's health check to verify:
```bash
curl http://localhost:8050/health
```

## Running the Agent

### Development Mode

```bash
cd backend/agents
python zoho_books_agent.py
```

The agent will start on `http://localhost:8050`

### Production Mode

Use a process manager like systemd, supervisor, or PM2:

```bash
# Using uvicorn directly
uvicorn zoho_books_agent:app --host 0.0.0.0 --port 8050

# Using gunicorn with uvicorn workers
gunicorn zoho_books_agent:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8050
```

## API Rate Limits

The test system has a limit of **1000 API calls per day**. The agent:
- Tracks daily usage automatically
- Resets counter at midnight
- Returns 429 error when limit exceeded
- Logs warnings when approaching limit

Monitor usage via `/health` endpoint.

## Pharmaceutical-Specific Features

### Invoice Line Items

When creating invoices, include pharmaceutical fields:

```json
{
  "line_items": [
    {
      "name": "Paracetamol 500mg",
      "rate": 10.50,
      "quantity": 100,
      "unit": "tablets",
      "batch_number": "BATCH-2024-001",
      "expiry_date": "2025-12-31",
      "drug_name": "Paracetamol",
      "manufacturer": "ABC Pharmaceuticals",
      "regulatory_license": "FDA-12345",
      "storage_conditions": "Room temperature",
      "hazard_class": "None",
      "controlled_substance": false,
      "prescription_required": false
    }
  ]
}
```

### Customer Fields

```json
{
  "customer_name": "XYZ Medical Supplies",
  "license_number": "PHARMA-LIC-12345",
  "regulatory_authority": "FDA",
  "tax_id": "TAX-ID-12345"
}
```

## Common Operations

### Create Invoice

```bash
curl -X POST http://localhost:8050/invoices \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "123456789",
    "date": "2024-01-15",
    "due_date": "2024-02-15",
    "line_items": [
      {
        "name": "Medicine A",
        "rate": 25.00,
        "quantity": 50,
        "batch_number": "BATCH-001",
        "expiry_date": "2025-12-31"
      }
    ],
    "notes": "Pharmaceutical invoice with batch tracking"
  }'
```

### List Invoices

```bash
curl "http://localhost:8050/invoices?status=sent&page=1&per_page=50"
```

### Send Invoice

```bash
curl -X POST http://localhost:8050/invoices/123456789/send \
  -H "Content-Type: application/json" \
  -d '{
    "email_ids": ["customer@example.com"],
    "subject": "Invoice #INV-001",
    "send_attachment": true
  }'
```

## Error Handling

The agent implements fail-safe error handling:

1. **Automatic Retry:** Failed requests retry up to 3 times with exponential backoff
2. **Token Refresh:** Access tokens automatically refresh when expired
3. **Rate Limit Handling:** Graceful handling of 429 errors with wait times
4. **Network Errors:** Retry logic for network failures
5. **Validation:** Input validation with clear error messages

## Integration with Orchestrator

The agent is automatically discovered by the orchestrator when:
1. Agent JSON file is in `Agent_entries/`
2. Agent is synced to database using `python manage.py sync_agents`
3. Agent service is running on configured port (8050)

The orchestrator will:
- Match tasks to agent capabilities
- Handle parameter mapping
- Execute requests with proper error handling
- Track agent responses

## Troubleshooting

### Token Expired
- Check `/health` endpoint
- Re-initialize tokens using `/oauth/initialize`
- Ensure refresh_token is valid

### Rate Limit Exceeded
- Check daily usage in `/health`
- Wait until next day (resets at midnight)
- Optimize API calls (batch operations)

### Connection Errors
- Verify `api_domain` in temp.json
- Check network connectivity
- Ensure Zoho Books API is accessible

### Invalid Credentials
- Verify client_id and client_secret
- Check organization_id is correct
- Ensure OAuth tokens have proper scopes

## Security Best Practices

1. **Never commit temp.json** with real credentials
2. **Use environment variables** for production
3. **Rotate tokens** regularly
4. **Monitor API usage** for anomalies
5. **Use HTTPS** in production
6. **Implement token encryption** for stored tokens

## Environment Variables Configuration

For production deployments, use environment variables instead of temp.json:

```bash
# .env file (backend/.env)
ZOHO_CLIENT_ID=your_client_id
ZOHO_CLIENT_SECRET=your_client_secret
ZOHO_ORGANIZATION_ID=your_organization_id
ZOHO_API_DOMAIN=https://www.zohoapis.in
ZOHO_ACCESS_TOKEN=your_access_token
ZOHO_REFRESH_TOKEN=your_refresh_token
```

Then modify the agent to load from environment:
```python
from dotenv import load_dotenv
import os

def load_zoho_config_from_env():
    load_dotenv()
    return {
        "client_id": os.getenv("ZOHO_CLIENT_ID"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET"),
        "organization_id": os.getenv("ZOHO_ORGANIZATION_ID"),
        "api_domain": os.getenv("ZOHO_API_DOMAIN", "https://www.zohoapis.in"),
        "access_token": os.getenv("ZOHO_ACCESS_TOKEN"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN")
    }
```

## Testing the Setup

### 1. Verify Agent Health

```bash
curl -X GET http://localhost:8050/health
```

Expected response:
```json
{
  "status": "healthy",
  "api_calls_today": 5,
  "rate_limit": 1000,
  "token_expiry": "2024-01-15T10:30:00Z"
}
```

### 2. Test Customer Creation

```bash
curl -X POST http://localhost:8050/customers \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Test Pharmacy",
    "email": "test@pharmacy.com",
    "phone": "1234567890"
  }'
```

### 3. Test Invoice Creation

```bash
curl -X POST http://localhost:8050/invoices \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "123456789",
    "date": "2024-01-15",
    "due_date": "2024-02-15",
    "line_items": [
      {
        "name": "Test Medicine",
        "rate": 50.00,
        "quantity": 10
      }
    ]
  }'
```

### 4. Python Integration Test

```python
import requests
import json

BASE_URL = "http://localhost:8050"

# Test health
health = requests.get(f"{BASE_URL}/health").json()
print(f"Agent Status: {health['status']}")

# Create customer
customer_data = {
    "customer_name": "ABC Pharmacy",
    "email": "contact@abc-pharmacy.com",
    "phone": "+1234567890"
}
customer = requests.post(f"{BASE_URL}/customers", json=customer_data).json()
customer_id = customer['contact_id']

# Create invoice
invoice_data = {
    "customer_id": customer_id,
    "date": "2024-01-15",
    "due_date": "2024-02-15",
    "line_items": [
        {
            "name": "Aspirin 500mg",
            "rate": 15.00,
            "quantity": 100,
            "batch_number": "BATCH-2024-001",
            "expiry_date": "2025-12-31"
        }
    ]
}
invoice = requests.post(f"{BASE_URL}/invoices", json=invoice_data).json()
print(f"Invoice Created: {invoice['invoice_id']}")
```

## Monitoring and Logging

### Log Locations

- **Agent Logs:** `storage/zoho_books/agent.log`
- **Request Logs:** `storage/zoho_books/requests.log`
- **Error Logs:** `storage/zoho_books/errors.log`

### Enable Debug Logging

In `zoho_books_agent.py`, change logging level:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO
    handlers=[
        logging.FileHandler("storage/zoho_books/debug.log"),
        logging.StreamHandler()
    ],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Monitor API Usage

Check rate limit status:
```bash
curl http://localhost:8050/health | jq '.api_calls_today, .rate_limit'
```

Set up alerts for approaching limits (e.g., when usage > 80% of daily limit).

## Database Integration

If integrating with the orchestrator database:

```sql
-- Register agent in agents table
INSERT INTO agents (
    id, name, agent_type, status, base_url, capabilities, created_at
) VALUES (
    'zoho_books_agent',
    'Zoho Books Agent - Pharmaceutical',
    'http_rest',
    'active',
    'http://localhost:8050',
    '["create invoice", "list invoices", "send invoice", "create customer"]',
    NOW()
);

-- Create agent configuration
INSERT INTO agent_configs (
    agent_id, key, value
) VALUES
    ('zoho_books_agent', 'org_id', 'your_org_id'),
    ('zoho_books_agent', 'rate_limit', '1000'),
    ('zoho_books_agent', 'retry_attempts', '3');
```

## Deployment Strategies

### Local Development

```bash
python backend/agents/zoho_books/zoho_books_agent.py
```

### Docker Deployment

Create `Dockerfile` in `backend/agents/zoho_books/`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "zoho_books_agent:app", "--host", "0.0.0.0", "--port", "8050"]
```

Build and run:
```bash
docker build -t zoho-books-agent .
docker run -p 8050:8050 \
  -e ZOHO_CLIENT_ID=xxx \
  -e ZOHO_CLIENT_SECRET=xxx \
  zoho-books-agent
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zoho-books-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: zoho-books-agent
  template:
    metadata:
      labels:
        app: zoho-books-agent
    spec:
      containers:
      - name: agent
        image: zoho-books-agent:latest
        ports:
        - containerPort: 8050
        env:
        - name: ZOHO_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: zoho-creds
              key: client_id
        - name: ZOHO_CLIENT_SECRET
          valueFrom:
            secretKeyRef:
              name: zoho-creds
              key: client_secret
        livenessProbe:
          httpGet:
            path: /health
            port: 8050
          initialDelaySeconds: 10
          periodSeconds: 30
```

## Performance Optimization

### Connection Pooling

Enable connection pooling for better performance:

```python
import httpx

# Use persistent client for multiple requests
client = httpx.Client(
    limits=httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30.0
    ),
    timeout=30.0
)
```

### Batch Operations

Create multiple invoices efficiently:

```bash
curl -X POST http://localhost:8050/invoices/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "customer_id": "123",
      "date": "2024-01-15",
      "line_items": [{"name": "Item 1", "rate": 10}]
    },
    {
      "customer_id": "456",
      "date": "2024-01-15",
      "line_items": [{"name": "Item 2", "rate": 20}]
    }
  ]'
```

### Caching Strategies

Cache frequently accessed data:
- Customer information (24-hour TTL)
- Item catalog (12-hour TTL)
- Tax settings (7-day TTL)

## API Endpoint Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Check agent health and rate limit status |
| POST | `/oauth/initialize` | Initialize OAuth tokens |
| POST | `/customers` | Create new customer |
| GET | `/customers` | List all customers |
| GET | `/customers/{id}` | Get customer details |
| POST | `/invoices` | Create new invoice |
| GET | `/invoices` | List invoices with filters |
| GET | `/invoices/{id}` | Get invoice details |
| PUT | `/invoices/{id}` | Update invoice |
| DELETE | `/invoices/{id}` | Delete invoice |
| POST | `/invoices/{id}/send` | Send invoice via email |
| POST | `/invoices/batch` | Create multiple invoices |
| GET | `/items` | List items/products |
| POST | `/items` | Create new item |

## Support

For issues or questions:
- Check agent logs in `storage/zoho_books/`
- Review Zoho Books API documentation: https://www.zoho.com/books/api/v3/
- Verify OAuth token validity via `/health` endpoint
- Test with curl commands before integration
- Review recent changes in [ARCHITECTURE_IMPROVEMENTS.md](./ARCHITECTURE_IMPROVEMENTS.md)

