# Gmail Agent: Credentials & Communication Flow

**Last Updated:** March 9, 2026

This document details how the Gmail agent retrieves user credentials from the database and communicates with Composio's Gmail API.

---

## Table of Contents

1. [Overview](#overview)
2. [Credential Retrieval Flow](#credential-retrieval-flow)
3. [Database Schema](#database-schema)
4. [Encryption & Security](#encryption--security)
5. [Communication with Composio](#communication-with-composio)
6. [Connection Lifecycle](#connection-lifecycle)
7. [Error Handling & Recovery](#error-handling--recovery)
8. [Code Examples](#code-examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Gmail agent uses a **layered credential system** to securely access user Gmail accounts through Composio's OAuth integration:

```
User Request
    ↓
Gmail Agent Capability
    ↓
GmailService (user_id)
    ↓
ComposioToolManager (user_id)
    ↓
ComposioAuthManager.get_connection_for_agent(user_id, "gmail")
    ↓
Database Query (UserConnection table)
    ↓
Decrypt connection_id
    ↓
Composio SDK (execute action with connection_id)
    ↓
Gmail API
```

**Key Principle:** Credentials are never stored directly. Only OAuth connection IDs are stored (encrypted) and used to authenticate with Composio, which manages the actual OAuth tokens.

---

## Credential Retrieval Flow

### Step 1: User Identity Resolution

When a capability is invoked:

```python
# In base_agent_impl.py
@capability(name="search_emails", ...)
async def search_emails(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
    # Extract user_id from ExecutionContext (provided by orchestrator)
    user_id = context.user_id
    
    # Get service for this user
    service = self._get_service(user_id)
```

**Source of Truth:** `context.user_id` (from ExecutionContext)
- Set by orchestrator based on authenticated user
- Never hardcoded or defaulted to "default"
- Consistent across entire request lifecycle

### Step 2: Service Initialization

```python
# In service.py
class GmailService:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.tool_mgr = ComposioToolManager(user_id)  # Passes user_id to tool manager
```

### Step 3: Tool Manager Connection Lookup

```python
# In tools.py
class ComposioToolManager:
    def __init__(self, user_id: str):
        from services.integrations.composio_auth import get_auth_manager
        
        # Get singleton auth manager
        auth_mgr = get_auth_manager()
        
        # Look up connection for this user + app
        connection = auth_mgr.get_connection_for_agent(user_id, "gmail")
        
        if not connection:
            raise ValueError(f"User {user_id} not connected to Gmail")
        
        # Store decrypted connection_id for tool execution
        self.user_id = user_id
        self.connection_id = connection["connection_id"]  # Already decrypted
        self.composio = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))
```

**Key Point:** The `connection` dict returned by `get_connection_for_agent()` contains the **already-decrypted** connection ID, ready for use.

### Step 4: Database Query

```python
# In services/integrations/composio_auth.py
def get_connection_for_agent(self, user_id: str, app_slug: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as db:
        # Query UserConnection table
        connection = (
            db.query(UserConnection)
            .filter(
                or_(
                    UserConnection.user_id == user_id,
                    UserConnection.internal_user_id == user_id,  # Backward compat
                ),
                UserConnection.app_slug == self._normalize_app_slug(app_slug),
                UserConnection.status.in_(["active", "stale"])  # Allow stale for refresh
            )
            .first()
        )
        
        if not connection:
            return None
```

**Filters:**
- **user_id match:** Uses both `user_id` and `internal_user_id` for backward compatibility
- **app_slug match:** Normalized to lowercase (e.g., "GMAIL" → "gmail")
- **status filter:** Only `active` or `stale` connections (not `disabled` or `initiated`)

### Step 5: Connection Verification (if needed)

```python
# Check if connection needs verification (> 1 hour old)
needs_verification = False
if connection.last_verified:
    time_since_verification = datetime.now(timezone.utc).replace(tzinfo=None) - connection.last_verified
    needs_verification = time_since_verification > timedelta(hours=1)
else:
    needs_verification = True

# Verify connection if stale
if needs_verification:
    try:
        # Quick status check with Composio API
        decrypted_id = self._decrypt_connection_id(connection.connection_id)
        
        # Check all candidate entity IDs
        entity_ids = self._get_candidate_entity_ids(user_id, app_slug)
        all_connections = []
        for entity_id in entity_ids:
            try:
                connections = self._composio.connected_accounts.get(entity_ids=[entity_id])
                all_connections.extend(connections)
            except Exception as e:
                logger.debug(f"Could not fetch for entity {entity_id}: {e}")
        
        # Verify connection exists and is active
        conn_found = False
        for conn in all_connections:
            if conn.id == decrypted_id and self._normalize_status(conn.status) in ["active", "connected"]:
                conn_found = True
                break
        
        if conn_found:
            # Update verification timestamp
            connection.last_verified = datetime.now(timezone.utc).replace(tzinfo=None)
            connection.status = "active"
            db.commit()
        else:
            # Mark as stale
            connection.status = "stale"
            db.commit()
            return None
            
    except Exception as verify_error:
        # Mark stale on verification failure
        connection.status = "stale"
        db.commit()
        return None
```

**Verification Logic:**
- Only runs if `last_verified` > 1 hour ago
- Quick check with Composio to ensure connection still valid
- Updates status and timestamp in DB
- Marks stale on failure (doesn't crash the request)

### Step 6: Return Decrypted Connection

```python
# Decrypt connection ID for use
decrypted_id = self._decrypt_connection_id(connection.connection_id)

return {
    "connection_id": decrypted_id,  # DECRYPTED for immediate use
    "app_slug": connection.app_slug,
    "status": connection.status,
    "created_at": connection.created_at,
    "metadata": connection.app_metadata
}
```

---

## Database Schema

### UserConnection Table

```sql
CREATE TABLE user_connections (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    internal_user_id VARCHAR,  -- Backward compatibility
    
    -- App identification
    app_slug VARCHAR NOT NULL,  -- 'gmail', 'zohobooks', 'github'
    app_name VARCHAR,           -- 'Gmail', 'Zoho Books'
    
    -- Composio linkage
    composio_entity_id VARCHAR,      -- Composio's entity ID
    connection_id VARCHAR NOT NULL,  -- ENCRYPTED connection ID
    
    -- Status tracking
    status VARCHAR DEFAULT 'initiated',  -- 'active', 'stale', 'disabled', 'initiated'
    
    -- Timestamps
    auth_timestamp TIMESTAMP,    -- When OAuth completed
    last_verified TIMESTAMP,     -- Last connection check
    created_at TIMESTAMP,
    
    -- Metadata
    app_metadata JSON,  -- App-specific data (e.g., email address, scopes)
    
    -- Indexes
    INDEX idx_user_app (user_id, app_slug),
    INDEX idx_status (status)
);
```

**Key Fields:**

| Field | Purpose | Example |
|-------|---------|---------|
| `user_id` | Your app's user identifier | `"user_33Kk4V3..."` |
| `app_slug` | Composio app identifier | `"gmail"` |
| `connection_id` | **Encrypted** Composio connection ID | `"gAAAAABm..."` |
| `status` | Connection state | `"active"` |
| `last_verified` | Last health check | `"2026-03-09 10:30:00"` |
| `composio_entity_id` | Composio entity for API calls | `"user_33Kk4V3..."` |

### Example Record

```json
{
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "user_id": "user_33Kk4V3owJke6eWQjxySIc7aWWq",
  "app_slug": "gmail",
  "app_name": "Gmail",
  "composio_entity_id": "user_33Kk4V3owJke6eWQjxySIc7aWWq",
  "connection_id": "gAAAAABm5x2j...[encrypted]",
  "status": "active",
  "auth_timestamp": "2026-03-09T09:15:00",
  "last_verified": "2026-03-09T10:30:00",
  "app_metadata": {
    "email": "user@gmail.com",
    "scopes": ["https://www.googleapis.com/auth/gmail.readonly", "..."]
  }
}
```

---

## Encryption & Security

### Encryption Mechanism

**Fernet Symmetric Encryption** (cryptography.fernet)

```python
# In composio_auth.py initialization
def _init_encryption_key(self):
    encryption_key = os.getenv("CONNECTION_ENCRYPTION_KEY")
    
    if not encryption_key:
        # Generate from SECRET_KEY for consistency
        secret_key = os.getenv("SECRET_KEY", "orbimesh-default-secret-key")
        key_bytes = hashlib.sha256(secret_key.encode()).digest()
        encryption_key = base64.urlsafe_b64encode(key_bytes).decode()
    
    self._cipher = Fernet(encryption_key.encode())
```

### Encryption at Storage

```python
def _encrypt_connection_id(self, connection_id: str) -> str:
    """Encrypt connection ID before storing in database."""
    return self._cipher.encrypt(connection_id.encode()).decode()
```

**When:** Connection ID is encrypted immediately after OAuth callback, before DB insertion.

### Decryption at Retrieval

```python
def _decrypt_connection_id(self, encrypted_id: str) -> str:
    """Decrypt connection ID for API use."""
    return self._cipher.decrypt(encrypted_id.encode()).decode()
```

**When:** Connection ID is decrypted only in `get_connection_for_agent()`, just before returning to the tool manager.

### Security Boundaries

```
┌─────────────────────────────────────────────────────┐
│  Database                                           │
│  connection_id: "gAAAAABm5x2j..." (ENCRYPTED)      │
└─────────────────────────────────────────────────────┘
                    ↓ (Decrypt only at retrieval)
┌─────────────────────────────────────────────────────┐
│  ComposioAuthManager.get_connection_for_agent()     │
│  Returns: {"connection_id": "abc123..."} (PLAIN)    │
└─────────────────────────────────────────────────────┘
                    ↓ (Used immediately)
┌─────────────────────────────────────────────────────┐
│  ComposioToolManager.execute_tool()                 │
│  Passes to Composio SDK                             │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  Composio API (manages OAuth tokens)                │
└─────────────────────────────────────────────────────┘
```

**Never Logged:**
- Decrypted connection IDs
- OAuth tokens (managed by Composio)
- User passwords
- Sensitive PII from emails

---

## Communication with Composio

### Tool Execution Flow

```python
# In tools.py
async def execute_tool(self, tool_slug: str, parameters: Dict[str, Any], _max_retries: int = 3) -> Dict[str, Any]:
    for attempt in range(_max_retries):
        try:
            # Create action enum from slug
            action = Action(tool_slug)  # e.g., "GMAIL_FETCH_EMAILS"
            
            # Execute via Composio SDK
            response = self.composio.execute_action(
                action=action,
                params=parameters,
                entity_id=self.user_id,              # Composio entity identifier
                connected_account=self.connection_id  # OAuth connection ID (decrypted)
            )
            
            return {
                "success": True,
                "data": response,
                "action": tool_slug
            }
            
        except Exception as e:
            # Handle retryable errors
            error_str = str(e).lower()
            is_retryable = any(kw in error_str for kw in self._RETRYABLE_KEYWORDS)
            
            if is_retryable and attempt < _max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                await asyncio.sleep(wait_time)
                continue
            
            # Permanent failure
            return {
                "success": False,
                "error": str(e),
                "action": tool_slug
            }
```

### Composio SDK Parameters

| Parameter | Type | Purpose | Example |
|-----------|------|---------|---------|
| `action` | Action enum | Gmail operation to execute | `Action.GMAIL_FETCH_EMAILS` |
| `params` | Dict | Tool-specific parameters | `{"query": "label:inbox", "max_results": 10}` |
| `entity_id` | str | User identifier in Composio | `"user_33Kk4V3..."` |
| `connected_account` | str | OAuth connection ID | `"abc123def456..."` (decrypted) |

### Retryable Errors

```python
_RETRYABLE_KEYWORDS = (
    "rate limit", "ratelimit", "429", "503", "502", "504",
    "timeout", "timed out", "connection", "temporarily unavailable",
)
```

**Retry Strategy:**
- Max 3 attempts
- Exponential backoff: 1s → 2s → 4s
- Only for transient network/API errors
- 401/403/404 errors fail immediately (auth/permission issues)

### Communication Protocol

```
ComposioToolManager
    ↓ (HTTPS)
Composio API (api.composio.dev)
    ↓ (OAuth 2.0)
Gmail API (gmail.googleapis.com)
    ↓
User's Gmail Account
```

**Composio acts as a proxy:**
1. Manages OAuth tokens and refresh
2. Handles token expiration automatically
3. Provides unified API across multiple services
4. Enforces rate limits and quotas

---

## Connection Lifecycle

### 1. Initiation (User Connects Gmail)

```
User clicks "Connect Gmail" in UI
    ↓
Frontend: POST /api/connections/start-auth
    ↓
Backend: composio_auth.start_auth_flow(user_id, "gmail")
    ↓
Composio SDK: entity.initiate_connection(app_name="GMAIL")
    ↓
Returns: redirect_url (Composio OAuth flow)
    ↓
Frontend redirects user to Composio
    ↓
User authenticates with Google
    ↓
Composio redirects back to callback_url
    ↓
Backend: POST /webhooks/composio (Composio webhook)
    ↓
Save connection to DB with status="active"
```

**Database Entry Created:**
```python
UserConnection(
    id=uuid4(),
    user_id=user_id,
    app_slug="gmail",
    connection_id=encrypt(connection_id),
    status="active",
    auth_timestamp=now(),
    last_verified=now()
)
```

### 2. Active Use (Agent Executes Tools)

```
Gmail Agent capability invoked
    ↓
Gets connection: get_connection_for_agent(user_id, "gmail")
    ↓
If last_verified > 1 hour: verify with Composio API
    ↓
Execute tool with decrypted connection_id
    ↓
Update last_verified timestamp
```

### 3. Stale Detection

**Triggers:**
- Last verified > 1 hour ago
- Verification check fails
- 401 error from Composio API

**Actions:**
```python
connection.status = "stale"
connection.last_verified = now()
db.commit()

# Next request will attempt re-verification
# or prompt user to reconnect
```

### 4. Refresh/Reconnect

**User-Initiated:**
```
User clicks "Reconnect" in dashboard
    ↓
Start new OAuth flow (same as #1)
    ↓
Replace old connection_id with new one
    ↓
status = "active"
```

**Automatic (Planned):**
```
On 401 error:
    → Attempt token refresh via Composio
    → If refresh fails: mark stale, prompt user
```

### 5. Disconnection

```
User clicks "Disconnect Gmail"
    ↓
Backend: composio_auth.disconnect_app(user_id, "gmail")
    ↓
Delete connection from UserConnection table
    ↓
Composio SDK: revoke OAuth token
```

---

## Error Handling & Recovery

### Connection Not Found

**Scenario:** User never connected Gmail

```python
# In ComposioToolManager.__init__
connection = auth_mgr.get_connection_for_agent(user_id, "gmail")
if not connection:
    raise ValueError(f"User {user_id} not connected to Gmail")

# Agent catches this
except ValueError as e:
    return AgentResponse.error(
        error=str(e),
        suggestion="Please connect Gmail in the connections page"
    )
```

**Frontend Action:** Show OAuth redirect link

### Connection Stale/Expired

**Scenario:** OAuth token expired or connection inactive

```python
# Status check during verification
if not conn_found:
    connection.status = "stale"
    db.commit()
    return None  # Triggers ValueError in tool manager

# Agent response
return AgentResponse.error(
    error="Gmail connection expired",
    suggestion="Please reconnect Gmail"
)
```

**Frontend Action:** Prompt user to reconnect

### Rate Limit Exceeded

**Scenario:** Too many requests to Gmail API

```python
# In execute_tool retry logic
if "rate limit" in str(e).lower() or "429" in str(e):
    if attempt < max_retries - 1:
        await asyncio.sleep(2 ** attempt)  # Exponential backoff
        continue

# After max retries
return {
    "success": False,
    "error": "Rate limit exceeded. Please try again in a few minutes."
}
```

**User Action:** Wait and retry

### Network/API Errors

**Scenario:** Transient Composio/Gmail API failures

```python
# Retryable errors
_RETRYABLE_KEYWORDS = (
    "timeout", "503", "502", "connection",
    "temporarily unavailable"
)

# Auto-retry with backoff
for attempt in range(3):
    try:
        response = composio.execute_action(...)
        return response
    except Exception as e:
        if is_retryable(e) and attempt < 2:
            await asyncio.sleep(2 ** attempt)
            continue
        raise
```

**Transparent to user** if succeeds on retry

---

## Code Examples

### Example 1: Search Emails (Complete Flow)

```python
# 1. User request
user_input = "Find emails from john@example.com"

# 2. Orchestrator creates context
context = ExecutionContext(
    thread_id="thread_123",
    user_id="user_abc",
    task_id="task_456"
)

# 3. Gmail agent capability
@capability(name="search_emails", ...)
async def search_emails(self, params: Dict[str, Any], context: ExecutionContext):
    # Get service for this user
    service = self._get_service(context.user_id)  # "user_abc"
    
    # Execute search
    result = await service.search_emails(
        query=params["query"],
        max_results=10
    )
    
    return AgentResponse.success(result)

# 4. GmailService
class GmailService:
    def __init__(self, user_id: str):
        self.tool_mgr = ComposioToolManager(user_id)
    
    async def search_emails(self, query: str, max_results: int):
        # LLM optimizes query
        optimized = await llm_client.generate_optimized_query(query)
        
        # Execute via tool manager
        result = await self.tool_mgr.fetch_emails(
            query=optimized,
            max_results=max_results
        )
        return result

# 5. ComposioToolManager
class ComposioToolManager:
    def __init__(self, user_id: str):
        # GET CONNECTION FROM DB
        auth_mgr = get_auth_manager()
        connection = auth_mgr.get_connection_for_agent(user_id, "gmail")
        
        if not connection:
            raise ValueError(f"User {user_id} not connected to Gmail")
        
        self.connection_id = connection["connection_id"]  # Decrypted
        self.composio = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))
    
    async def fetch_emails(self, query: str, max_results: int):
        return await self.execute_tool(
            "GMAIL_FETCH_EMAILS",
            {"query": query, "max_results": max_results}
        )
    
    async def execute_tool(self, tool_slug: str, parameters: Dict):
        action = Action(tool_slug)
        
        # COMMUNICATE WITH COMPOSIO
        response = self.composio.execute_action(
            action=action,
            params=parameters,
            entity_id=self.user_id,
            connected_account=self.connection_id  # Decrypted connection ID
        )
        
        return {"success": True, "data": response}

# 6. Database query (inside get_connection_for_agent)
connection = db.query(UserConnection).filter(
    UserConnection.user_id == "user_abc",
    UserConnection.app_slug == "gmail",
    UserConnection.status.in_(["active", "stale"])
).first()

# 7. Decrypt connection ID
decrypted_id = cipher.decrypt(connection.connection_id.encode()).decode()

# 8. Return to tool manager
return {
    "connection_id": decrypted_id,  # "conn_xyz789" (plain)
    "app_slug": "gmail",
    "status": "active"
}
```

### Example 2: Send Email with Error Handling

```python
async def send_email(self, params: Dict[str, Any], context: ExecutionContext):
    try:
        # Get service (may raise ValueError if not connected)
        service = self._get_service(context.user_id)
        
        # Send email
        result = await service.send_email(
            to=params["to"],
            subject=params["subject"],
            body=params["body"]
        )
        
        if result["success"]:
            return AgentResponse.success(
                result=result["data"],
                summary=f"Email sent to {params['to']}"
            )
        else:
            return AgentResponse.error(error=result["error"])
    
    except ValueError as e:
        # Connection not found
        if "not connected" in str(e):
            return AgentResponse.error(
                error="Gmail not connected",
                suggestion="Please connect Gmail in the connections page"
            )
        raise
    
    except Exception as e:
        logger.error(f"Send email failed: {e}")
        return AgentResponse.error(error=str(e))
```

---

## Troubleshooting

### Issue: "User not connected to Gmail"

**Symptom:** `ValueError` raised when initializing ComposioToolManager

**Diagnosis:**
```sql
SELECT * FROM user_connections 
WHERE user_id = 'user_abc' AND app_slug = 'gmail';
```

**Solutions:**

1. **No records:** User never connected
   - Action: User must complete OAuth flow

2. **Record exists with status='disabled':**
   - Action: Re-enable or reconnect

3. **Record exists with status='stale':**
   - Action: Verification failed, prompt reconnect

4. **Record exists with old `user_id` format:**
   - Check `internal_user_id` field for mapping

### Issue: "Connection expired" / 401 Errors

**Symptom:** Tool execution returns 401 Unauthorized

**Diagnosis:**
1. Check `last_verified` timestamp
2. Check `status` field
3. Try manual verification:
   ```python
   auth_mgr.refresh_connection(user_id, "gmail")
   ```

**Solutions:**

1. **last_verified > 24 hours:**
   - Auto-verification should trigger
   - If fails: prompt user to reconnect

2. **Composio token expired:**
   - Composio should auto-refresh
   - If not: user must reconnect

### Issue: Rate-Limited

**Symptom:** `429 Too Many Requests`

**Diagnosis:**
- Check logs for retry attempts
- Check if hitting Gmail API quotas

**Solutions:**

1. **Transient rate limit:**
   - Auto-retry with backoff should handle
   - User can retry after ~1 minute

2. **Persistent rate limit:**
   - May need to implement request throttling
   - Consider batching requests

### Issue: Wrong User's Data Returned

**Symptom:** User A sees User B's emails

**Diagnosis:**
```python
# Check identity flow
logger.info(f"Capability user_id: {context.user_id}")
logger.info(f"Service user_id: {service.user_id}")
logger.info(f"Tool manager user_id: {tool_mgr.user_id}")
logger.info(f"DB query user_id: {connection.user_id}")
```

**Solutions:**

1. **Identity mismatch detected:**
   - Check `context.user_id` propagation
   - Never use hardcoded "default"
   - Ensure no cross-user service caching

2. **Entity ID confusion:**
   - Verify `composio_entity_id` matches `user_id`
   - Check `_get_candidate_entity_ids()` logic

### Issue: Decryption Failure

**Symptom:** `Fernet.DecryptionError`

**Diagnosis:**
- Check `CONNECTION_ENCRYPTION_KEY` env var
- Check if key changed after connections created

**Solutions:**

1. **Key mismatch:**
   - Connections encrypted with old key can't decrypt with new key
   - Must reconnect all users

2. **Corrupted data:**
   - Check if `connection_id` field intact
   - May need to prompt user to reconnect

---

## Security Best Practices

1. **Never log decrypted connection IDs**
   ```python
   # BAD
   logger.info(f"Connection: {connection_id}")
   
   # GOOD
   logger.info(f"Connection status: {status}")
   ```

2. **Always use parameterized queries**
   ```python
   # Already done in SQLAlchemy queries
   db.query(UserConnection).filter(UserConnection.user_id == user_id)
   ```

3. **Encrypt before storing**
   ```python
   connection.connection_id = encrypt(composio_connection_id)
   db.add(connection)
   db.commit()
   ```

4. **Decrypt only at execution boundary**
   ```python
   # Decrypt just before passing to Composio SDK
   decrypted = self._decrypt(connection.connection_id)
   composio.execute_action(..., connected_account=decrypted)
   ```

5. **Verify connection periodically**
   ```python
   # Auto-verify if > 1 hour old
   if needs_verification(connection):
       verify_and_update(connection)
   ```

6. **Handle auth failures gracefully**
   ```python
   if "401" in error or "unauthorized" in error:
       mark_stale(connection)
       return error_response("Please reconnect")
   ```

---

## Performance Considerations

### Connection Caching

```python
# In GmailAgent
self._services: Dict[str, GmailService] = {}

def _get_service(self, user_id: str) -> GmailService:
    if user_id not in self._services:
        self._services[user_id] = GmailService(user_id)
    return self._services[user_id]
```

**Benefit:** Avoids repeated DB queries for same user in same session

### Connection Pooling

```python
# SQLAlchemy automatically pools DB connections
SessionLocal = sessionmaker(bind=engine, pool_size=10, max_overflow=20)
```

### Lazy Verification

```python
# Only verify if > 1 hour old
if (now - last_verified) > timedelta(hours=1):
    verify_connection()
```

**Benefit:** Reduces unnecessary Composio API calls

---

## Related Documentation

- [Gmail Agent Architecture](./ARCHITECTURE.md) - Overall agent design
- [Composio Auth Manager](../../services/integrations/composio_auth.py) - Connection management
- [Connection System Generalization](../../CONNECTION_SYSTEM_GENERALIZATION.md) - Future improvements
- [Copilot Context: Integrations](../../.vscode/COPILOT_CONTEXT_INTEGRATIONS.md) - Development guidelines
