# Orbimesh Backend API Documentation
## Comprehensive API Endpoint & Integration Reference

**Generated:** December 8, 2025  
**Backend Framework:** FastAPI  
**Database:** PostgreSQL with pgvector  
**Auth:** Clerk (JWT tokens)

---

## Table of Contents
1. [HTTP REST Endpoints](#http-rest-endpoints)
2. [WebSocket Endpoints](#websocket-endpoints)
3. [External Service Integrations](#external-service-integrations)
4. [Database Models & Relationships](#database-models--relationships)
5. [File System Operations](#file-system-operations)
6. [Background Services](#background-services)

---

## HTTP REST Endpoints

### 📁 File Management
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `POST` | `/api/upload` | Upload files (images/documents) to storage | No |
| `GET` | `/api/files/{file_path:path}` | Serve uploaded files from storage | No |

**Upload Response:**
```json
[{
  "file_name": "example.png",
  "file_path": "storage/images/example.png",
  "file_type": "image"
}]
```

---

### 💬 Chat & Conversation Management
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `POST` | `/api/chat` | Start new conversation with agent orchestration | No |
| `POST` | `/api/chat/continue` | Continue paused conversation (user response) | No |
| `GET` | `/api/chat/status/{thread_id}` | Get conversation status | No |
| `GET` | `/api/chat/history/{thread_id}` | Load conversation from JSON file | No |
| `DELETE` | `/api/chat/{thread_id}` | Clear conversation from memory | No |
| `GET` | `/api/chat/debug/conversations` | Debug: view all active conversations | No |

**Chat Request:**
```json
{
  "prompt": "Create a weather dashboard",
  "thread_id": "optional-uuid",
  "files": [{"file_name": "...", "file_path": "...", "file_type": "..."}]
}
```

**Chat Response:**
```json
{
  "message": "Successfully processed the request.",
  "thread_id": "uuid",
  "task_agent_pairs": [...],
  "final_response": "...",
  "pending_user_input": false,
  "question_for_user": null,
  "has_canvas": true,
  "canvas_content": "<html>...",
  "current_view": "browser"
}
```

---

### 🗂️ Conversation History & Threads
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `GET` | `/api/conversations` | List all user conversations | Yes (Clerk) |
| `GET` | `/api/conversations/{thread_id}` | Get full conversation history | Yes (Clerk) |

**Conversations List Response:**
```json
[{
  "id": "thread-uuid",
  "thread_id": "thread-uuid",
  "title": "Weather Dashboard",
  "created_at": "2025-12-08T10:00:00",
  "updated_at": "2025-12-08T10:05:00",
  "last_message": "Here's your dashboard..."
}]
```

---

### 🖼️ Canvas Management (Browser Agent)
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `POST` | `/api/canvas/update` | Receive live browser screenshots/task plan | No |
| `POST` | `/api/canvas/toggle-view` | Switch between browser/plan view | No |

**Canvas Update:**
```json
{
  "thread_id": "uuid",
  "screenshot_data": "base64...",
  "url": "https://example.com",
  "step": 5,
  "task": "Fill form",
  "task_plan": [{"subtask": "...", "status": "completed"}],
  "current_action": "Clicking submit button"
}
```

---

### 🔄 Workflow Management
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `POST` | `/api/workflows` | Save conversation as reusable workflow | Yes (Clerk) |
| `GET` | `/api/workflows` | List user's workflows | Yes (Clerk) |
| `GET` | `/api/workflows/{workflow_id}` | Get workflow details | Yes (Clerk) |
| `POST` | `/api/workflows/{workflow_id}/execute` | Execute workflow (creates new thread) | Yes (Clerk) |
| `POST` | `/api/workflows/{workflow_id}/create-conversation` | Create conversation from workflow | Yes (Clerk) |
| `POST` | `/api/workflows/{workflow_id}/schedule` | Schedule workflow with cron | Yes (Clerk) |
| `POST` | `/api/workflows/{workflow_id}/webhook` | Create webhook trigger | Yes (Clerk) |

**Save Workflow:**
```json
POST /api/workflows?thread_id=uuid&name=Weather+Dashboard&description=...
Response: {
  "workflow_id": "uuid",
  "name": "Weather Dashboard",
  "status": "saved",
  "task_count": 3,
  "created_at": "2025-12-08T10:00:00"
}
```

---

### ⏰ Workflow Scheduling
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `GET` | `/api/schedules` | List all user's schedules | Yes (Clerk) |
| `GET` | `/api/schedules/{schedule_id}/executions` | Get schedule execution history | Yes (Clerk) |
| `PATCH` | `/api/schedules/{schedule_id}` | Update schedule (pause/resume/change cron) | Yes (Clerk) |
| `DELETE` | `/api/workflows/{workflow_id}/schedule/{schedule_id}` | Delete schedule | Yes (Clerk) |

**Schedule Workflow:**
```json
POST /api/workflows/{id}/schedule
{
  "cron_expression": "0 9 * * *",
  "input_template": {"location": "NYC"}
}
```

---

### 🔗 Webhook Triggers
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `POST` | `/webhooks/{webhook_id}?webhook_token=...` | Trigger workflow via webhook | Token-based |

---

### 📊 Plan & Execution
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `GET` | `/api/plan/{thread_id}` | Get markdown execution plan | No |

---

### 🤖 Agent Management
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `POST` | `/api/agents/register` | Register/update agent | No |
| `GET` | `/api/agents/search?capabilities=...` | Search agents by capabilities | No |
| `GET` | `/api/agents/all` | Get all active agents | No |
| `GET` | `/api/agents/{agent_id}` | Get agent details | No |
| `POST` | `/api/agents/{agent_id}/rate` | Rate agent (1-5 stars) | No |
| `POST` | `/api/agents/by-name/{agent_name}/rate` | Rate agent by name | No |

**Agent Search:**
```
GET /api/agents/search?capabilities=weather&capabilities=visualization&max_price=0.05
```

---

### 🔐 Credentials Management
*Router: `credentials_router.py`*

| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `GET` | `/api/credentials/status` | Get credential status for all agents | Yes (Clerk) |
| `GET` | `/api/credentials/{agent_id}` | Get configured credentials for agent | Yes (Clerk) |
| `POST` | `/api/credentials/{agent_id}` | Save/update agent credentials | Yes (Clerk) |
| `DELETE` | `/api/credentials/{agent_id}` | Delete agent credentials | Yes (Clerk) |
| `POST` | `/api/credentials/{agent_id}/test` | Test credential validity | Yes (Clerk) |

**Save Credentials:**
```json
POST /api/credentials/{agent_id}
{
  "agent_id": "gmail-agent",
  "credentials": [
    {"field_name": "api_key", "value": "sk-..."},
    {"field_name": "connection_id", "value": "conn_123"}
  ]
}
```

---

### 🔌 MCP Connection Management
*Router: `connect_router.py`*

| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `POST` | `/api/connect/probe` | Probe MCP server for auth requirements | No |
| `POST` | `/api/connect/ingest` | Connect and ingest MCP server | No |
| `GET` | `/api/connect/list?user_id=...` | List user's MCP connections | No |
| `DELETE` | `/api/connect/{agent_id}?user_id=...` | Delete MCP connection | No |
| `GET` | `/api/connect/integrations` | Get pre-configured integration templates | No |

**Probe MCP:**
```json
POST /api/connect/probe
{
  "url": "https://mcp.supabase.com/mcp"
}
Response: {
  "status": "auth_required",
  "type": "api_key",
  "header": "Authorization",
  "message": "API key authentication required"
}
```

**Ingest MCP:**
```json
POST /api/connect/ingest
{
  "url": "https://mcp.supabase.com/mcp",
  "credentials": {"Authorization": "Bearer sk-..."},
  "user_id": "user_123",
  "agent_name": "Supabase DB",
  "agent_description": "Database operations"
}
```

---

### 📦 Content Management (Unified)
*Router: `content_router.py`*

| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `POST` | `/api/content/upload` | Upload files (returns content metadata) | No |
| `POST` | `/api/content/store` | Store artifact (auto-compresses large content) | No |
| `GET` | `/api/content/download/{content_id}` | Download content | No |
| `GET` | `/api/content/{content_id}` | Get content metadata | No |
| `GET` | `/api/content/{content_id}/reference` | Get content reference (summary only) | No |
| `GET` | `/api/content/{content_id}/full` | Get full content | No |
| `GET` | `/api/content/list/all` | List all content | No |
| `GET` | `/api/content/thread/{thread_id}` | List content by thread | No |
| `GET` | `/api/content/type/{content_type}` | List content by type | No |
| `GET` | `/api/content/search/tags` | Search content by tags | No |
| `GET` | `/api/content/{content_id}/agent-mapping/{agent_id}` | Get agent content mapping | No |
| `POST` | `/api/content/{content_id}/upload-to-agent/{agent_id}` | Upload content to agent | No |
| `POST` | `/api/content/optimize-context` | Get optimized context for thread | No |
| `DELETE` | `/api/content/{content_id}` | Delete content | No |
| `POST` | `/api/content/cleanup/expired` | Clean up expired content | No |
| `POST` | `/api/content/cleanup/session/{thread_id}` | Clean up thread content | No |
| `GET` | `/api/content/stats/overview` | Get storage statistics | No |
| `POST` | `/api/content/files/upload` | Upload files (alternative endpoint) | No |
| `GET` | `/api/content/files/download/{file_id}` | Download file by ID | No |
| `GET` | `/api/content/files/{file_id}` | Get file metadata | No |

**Store Artifact:**
```json
POST /api/content/store
{
  "content": {"data": "large json..."},
  "name": "API Response",
  "content_type": "data",
  "description": "User data from API",
  "thread_id": "uuid",
  "priority": "high",
  "tags": ["api", "user-data"],
  "ttl_hours": 24
}
```

---

### 📈 Analytics & Metrics
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `GET` | `/api/metrics/dashboard` | Get comprehensive dashboard metrics | Yes (X-User-ID header) |

**Dashboard Metrics Response:**
```json
{
  "total_conversations": 42,
  "total_workflows": 8,
  "total_agents": 15,
  "recent_activity": 5,
  "conversation_trend": [{"date": "Dec 01", "count": 3}],
  "cost_metrics": {
    "today": 0.45,
    "week": 2.34,
    "month": 8.92,
    "total": 24.56,
    "avg_per_conversation": 0.58
  },
  "performance_metrics": {
    "total_tasks": 156,
    "successful_tasks": 149,
    "success_rate": 95.51,
    "avg_response_time": 2.5
  },
  "top_agents": [{
    "name": "Gmail Agent",
    "calls": 23,
    "cost": 1.15,
    "cost_per_call": 0.05
  }]
}
```

---

### 🏥 System Health & Admin
| Method | Path | Purpose | Auth Required |
|--------|------|---------|---------------|
| `GET` | `/api/health` | Health check | No |
| `GET` | `/api/agent-servers/status` | Get status of all agent servers | No |
| `POST` | `/api/admin/reload-schedules` | Reload schedules from DB | No |

---

## WebSocket Endpoints

### 🔌 Real-time Chat
**Endpoint:** `/ws/chat`

**Client → Server:**
```json
{
  "prompt": "Create dashboard",
  "thread_id": "optional-uuid",
  "user_response": "optional-for-continuation",
  "files": [...],
  "owner": {"user_id": "clerk_user_id", "email": "user@example.com"},
  "planning_mode": false
}
```

**Server → Client Events:**
```json
// Start
{"node": "__start__", "thread_id": "uuid", "message": "Starting..."}

// Node updates
{"node": "analyze_request", "data": {...}, "progress_percentage": 10}
{"node": "parse_prompt", "data": {"tasks_identified": 3, "task_names": [...]}}
{"node": "agent_directory_search", "data": {"agents_found": 5}}
{"node": "execute_batch", "data": {...}}

// Live canvas updates (browser agent)
{"node": "__live_canvas__", "data": {"canvas_content": "<html>..."}}

// Real-time task events
{"node": "task_started", "task_name": "Fetch weather", "agent_name": "Weather API"}
{"node": "task_completed", "task_name": "Fetch weather", "execution_time": 1.23}
{"node": "task_failed", "task_name": "...", "error": "..."}

// User input required (approval/question)
{"node": "__user_input_required__", "data": {
  "question_for_user": "Approve execution?",
  "approval_required": true,
  "estimated_cost": 0.15,
  "task_count": 3,
  "task_plan": [...]
}}

// Completion
{"node": "__end__", "data": {...}, "status": "completed"}

// Errors
{"node": "__error__", "error": "...", "error_type": "...", "error_category": "..."}
```

**Node Sequence:**
1. `analyze_request` - Analyze user request
2. `parse_prompt` - Extract tasks
3. `agent_directory_search` - Find capable agents
4. `rank_agents` - Rank and select best agents
5. `plan_execution` - Create execution plan
6. `validate_plan_for_execution` - Validate plan
7. `execute_batch` - Execute tasks
8. `evaluate_agent_response` - Evaluate results
9. `generate_final_response` - Generate response
10. `save_history` - Save conversation

---

### 🔄 Workflow Execution
**Endpoint:** `/ws/workflow/{workflow_id}/execute?token=...`

**Client → Server:**
```json
{
  "inputs": {"location": "NYC"}
}
```

**Server → Client:**
Similar to `/ws/chat` but executes saved workflow plan.

---

## External Service Integrations

### 🤖 LLM Providers

#### Cerebras (Primary)
- **Service:** llama-3.3-70b
- **Used For:** All agent orchestration, planning, parsing
- **API Key:** `CEREBRAS_API_KEY` env var
- **Features:** JSON mode, structured outputs
- **Classes:** `ExtendedChatCerebras`, `ChatCerebras`

#### Groq (Backup/Fast)
- **Service:** llama-3.3-70b-versatile
- **Used For:** Fast inference when needed
- **API Key:** `GROQ_API_KEY`

#### OpenAI (Vision)
- **Service:** GPT-4 Vision
- **Used For:** Image analysis tasks
- **API Key:** `OPENAI_API_KEY`

---

### 📧 Email Integration (Composio)

**Environment Variables:**
```bash
COMPOSIO_API_KEY=your_key
GMAIL_CONNECTION_ID=conn_xxxxx
GMAIL_MCP_URL=https://...
```

**Setup:**
1. Get Composio API key from https://app.composio.dev/
2. Add to backend/.env
3. Connect Gmail account via Composio dashboard
4. Configure connection ID in .env

**Usage in Agents:**
```python
from composio_langchain import ComposioToolSet, App
toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))
tools = toolset.get_tools(apps=[App.GMAIL], entity_id=connection_id)
```

---

### 🌐 MCP (Model Context Protocol) Servers

**Service Module:** `services/mcp_service.py`

**Supported MCP Servers:**
- Supabase (`mcp.supabase.com`)
- Linear
- GitHub
- GitLab
- Notion
- Custom MCP servers

**Functions:**
- `probe_mcp_url(url)` - Detect auth requirements
- `ingest_mcp_agent(db, url, user_id, credentials)` - Connect & discover tools
- `list_user_connections(db, user_id)` - Get user's MCP agents
- `delete_user_connection(db, user_id, agent_id)` - Remove connection

**MCP Discovery:**
- Connects to `/sse` endpoint
- Lists tools via `tools/list` method
- Creates `AgentEndpoint` for each tool
- Stores encrypted credentials per user

---

### 🎯 Agent Servers (Local)

**Location:** `backend/agents/`

**Active Agents:**
- `browser_automation_agent.py` - Playwright browser automation (Port: 8001)
- `document_analysis_agent.py` - PDF/document processing (Port: 8002)
- `finance_agent.py` - Financial data APIs (Port: 8003)
- `groq_search_agent.py` - Web search via Groq (Port: 8004)
- `image_analysis_agent.py` - Image analysis (Port: 8005)
- `mail_agent.py` - Gmail via Composio (Port: 8006)
- `news_agent.py` - News APIs (Port: 8007)
- `wiki_agent.py` - Wikipedia search (Port: 8008)

**Lifecycle:**
- Started on FastAPI startup via `start_agents_async()`
- Health checked via background task
- Logs: `backend/logs/{agent_name}.log`
- Status: `/api/agent-servers/status`

---

### 📊 Vector Search (pgvector)

**Library:** `sentence-transformers` (all-mpnet-base-v2)

**Usage:**
- Agent capability search
- Semantic matching of tasks to agents
- 768-dimensional embeddings
- Cosine similarity search

**Models:**
- `AgentCapability` - Stores capability embeddings
- Query: `embedding.cosine_distance(query_vector) < threshold`

---

## Database Models & Relationships

### Core Agent Models

#### `Agent`
```python
id: str (primary key)
owner_id: str
name: str
description: text
capabilities: JSON (array of strings)
price_per_call_usd: float
status: enum (active, inactive, deprecated)
rating: float
rating_count: int
created_at: datetime

# MCP Support
agent_type: str (http_rest, mcp_http)
connection_config: JSON
requires_credentials: bool
credential_fields: JSON

# Relationships
capability_vectors: List[AgentCapability]
endpoints: List[AgentEndpoint]
credentials: List[AgentCredential]
```

#### `AgentCapability`
```python
id: int (primary key)
agent_id: str (foreign key → agents.id)
capability_text: str
embedding: Vector(768)
```

#### `AgentEndpoint`
```python
id: int (primary key)
agent_id: str (foreign key → agents.id)
endpoint: str
http_method: str
description: text

# Relationships
parameters: List[EndpointParameter]
```

#### `EndpointParameter`
```python
id: int (primary key)
endpoint_id: int (foreign key → agent_endpoints.id)
name: str
description: text
param_type: str
required: bool
default_value: str
```

---

### Conversation Models

#### `UserThread`
```python
id: int (primary key)
user_id: str (index)
thread_id: str (unique, index)
title: str
created_at: datetime
updated_at: datetime
```

#### `ConversationPlan`
```python
plan_id: str (unique)
thread_id: str (foreign key → user_threads.thread_id)
user_id: str
plan_version: int
task_agent_pairs: JSON
task_plan: JSON
plan_graph: JSON
status: str (draft, executing, completed, failed)
execution_time_ms: int
```

#### `ConversationSearch`
```python
thread_id: str (foreign key)
user_id: str
message_index: int
message_content: text
message_role: str
message_timestamp: datetime
```

#### `ConversationTag`
```python
tag_id: str (unique)
user_id: str
tag_name: str
tag_color: str (hex)
is_system: bool
```

---

### Workflow Models

#### `Workflow`
```python
workflow_id: str (unique)
user_id: str
name: str
description: text
blueprint: JSON  # Full workflow structure
plan_graph: JSON  # Visualization
version: int
status: str (active, archived)
created_at: datetime
updated_at: datetime
```

#### `WorkflowExecution`
```python
execution_id: str (unique)
workflow_id: str (foreign key)
user_id: str
status: str (queued, running, completed, failed)
inputs: JSON
outputs: JSON
error: text
started_at: datetime
completed_at: datetime
```

#### `WorkflowSchedule`
```python
schedule_id: str (unique)
workflow_id: str (foreign key)
user_id: str
cron_expression: str (UTC)
input_template: JSON
is_active: bool
last_run_at: datetime
next_run_at: datetime
```

#### `WorkflowWebhook`
```python
webhook_id: str (unique)
workflow_id: str (foreign key)
user_id: str
webhook_token: str
is_active: bool
```

---

### Credentials Model

#### `AgentCredential`
```python
id: str (primary key)
user_id: str (Clerk ID)
agent_id: str (foreign key → agents.id)
encrypted_credentials: JSON  # {"api_key": "enc...", "connection_id": "enc..."}
auth_type: str (none, api_key, oauth2)
is_active: bool
created_at: datetime
updated_at: datetime
```

**Encryption:**
- Uses `cryptography.fernet`
- Key from `ENCRYPTION_KEY` env var
- All credential values encrypted before storage

---

### Analytics Models

#### `ConversationAnalytics`
```python
thread_id: str (unique, foreign key)
user_id: str
total_messages: int
total_agents_used: int
plan_attempts: int
successful_plans: int
total_execution_time_ms: int
avg_response_time_ms: float
```

#### `AgentUsageAnalytics`
```python
analytics_id: str (unique)
user_id: str
agent_id: str (foreign key)
execution_count: int
success_count: int
failure_count: int
avg_execution_time_ms: float
last_used_at: datetime
```

#### `UserActivitySummary`
```python
user_id: str
activity_date: str (YYYY-MM-DD)
total_conversations_started: int
total_workflows_executed: int
total_plans_created: int
successful_executions: int
api_calls_made: int
```

---

## File System Operations

### 📂 Directory Structure
```
backend/
├── conversation_history/      # Saved conversation JSON files
│   └── {thread_id}.json
├── agent_plans/               # Markdown execution plans
│   └── {thread_id}-plan.md
├── storage/
│   ├── images/                # Uploaded images
│   ├── documents/             # Uploaded documents
│   └── vector_store/          # Future: embeddings
├── logs/
│   ├── orchestrator_temp.log  # Last conversation logs
│   └── {agent_name}_agent.py.log
└── Agent_entries/             # Agent definition JSONs
    ├── browser_automation_agent.json
    ├── gmail_agent.json
    └── ...
```

---

### 📝 Conversation History Format
**File:** `conversation_history/{thread_id}.json`

```json
{
  "thread_id": "uuid",
  "original_prompt": "Create dashboard",
  "task_agent_pairs": [...],
  "task_plan": [[{"primary": {...}, "task": {...}}]],
  "messages": [
    {
      "type": "user",
      "content": "Create dashboard",
      "timestamp": 1733645678.123,
      "id": "msg_abc123"
    },
    {
      "type": "assistant",
      "content": "I'll create that for you...",
      "timestamp": 1733645680.456,
      "id": "msg_def456"
    }
  ],
  "completed_tasks": [...],
  "final_response": "Here's your dashboard...",
  "pending_user_input": false,
  "status": "completed",
  "metadata": {
    "original_prompt": "...",
    "parsed_tasks": [...]
  },
  "uploaded_files": [...]
}
```

---

### 📋 Agent Execution Plan
**File:** `agent_plans/{thread_id}-plan.md`

```markdown
# Task Execution Plan
**Thread ID:** {thread_id}

## Batch 1
### Task: Fetch weather data
- **Agent:** Weather API Agent
- **Endpoint:** GET /weather
- **Expected Output:** Temperature, conditions

### Task: Visualize data
- **Agent:** Chart Generator
- **Endpoint:** POST /chart
```

---

### 🗃️ Agent Entry Files
**Location:** `Agent_entries/{agent_name}.json`

```json
{
  "id": "weather-api-agent",
  "owner_id": "system",
  "name": "Weather API Agent",
  "description": "Fetch weather data",
  "capabilities": [
    "fetch current weather",
    "get weather forecast",
    "historical weather data"
  ],
  "price_per_call_usd": 0.01,
  "status": "active",
  "agent_type": "http_rest",
  "connection_config": {
    "base_url": "https://api.weather.com"
  },
  "endpoints": [
    {
      "endpoint": "/weather",
      "http_method": "GET",
      "description": "Get current weather",
      "parameters": [
        {
          "name": "location",
          "param_type": "query",
          "required": true,
          "description": "City name or coordinates"
        }
      ]
    }
  ]
}
```

**Sync to DB:**
```bash
python manage.py sync
# Or automatic on startup via startup_event()
```

---

## Background Services

### ⏰ Workflow Scheduler
**Module:** `services/workflow_scheduler.py`  
**Class:** `WorkflowScheduler`

**Features:**
- APScheduler with cron expressions
- UTC timezone for all schedules
- Background execution via asyncio
- Automatic schedule loading on startup

**Key Methods:**
```python
add_schedule(schedule_id, workflow_id, cron_expression, input_template, user_id, db_session_factory)
remove_schedule(schedule_id)
load_active_schedules(db)
```

**Cron Format:** `minute hour day month day_of_week` (UTC)
- `0 9 * * *` - Daily at 9 AM UTC
- `0 */6 * * *` - Every 6 hours
- `0 0 * * 1` - Every Monday at midnight

**Execution Flow:**
1. Scheduler fires job
2. Create `WorkflowExecution` record
3. Execute via `_async_execute_workflow()`
4. Run orchestration with workflow blueprint
5. Update execution status (completed/failed)

---

### 🏥 Agent Health Checker
**Function:** `check_agent_health_background()`

**Features:**
- Runs every 3 seconds
- Checks `/health` or `/` endpoints
- Updates `agent_status` dict
- States: `starting`, `ready`, `failed`

**Status API:** `GET /api/agent-servers/status`

---

### 📊 Artifact Cleanup
**Module:** `services/artifact_service.py`

**Features:**
- TTL-based expiration
- Automatic cleanup of expired artifacts
- Compression for large content (>10KB)
- Reference counting

**Endpoint:** `POST /api/artifacts/cleanup/expired`

---

### 💾 Conversation State Management
**In-Memory Store:** `conversation_store: Dict[str, Dict[str, Any]]`

**Features:**
- Thread-safe with `store_lock`
- Persisted to JSON files
- LangGraph checkpointer for state recovery
- Message deduplication via `MessageManager`

**State Keys:**
```python
{
  "thread_id": str,
  "original_prompt": str,
  "messages": List[BaseMessage],
  "task_plan": List[List[Dict]],
  "task_agent_pairs": List[Dict],
  "completed_tasks": List[Dict],
  "final_response": str,
  "pending_user_input": bool,
  "question_for_user": str,
  "uploaded_files": List[Dict],
  "needs_approval": bool,
  "plan_approved": bool,
  "planning_mode": bool
}
```

---

## Orchestrator Architecture

### 🧠 LangGraph Workflow
**Module:** `orchestrator/graph.py`

**Nodes:**
1. **analyze_request** - Determine if complex processing needed
2. **parse_prompt** - Extract tasks from user prompt
3. **agent_directory_search** - Find capable agents via vector search
4. **rank_agents** - Rank agents by relevance and cost
5. **plan_execution** - Create batched execution plan
6. **validate_plan_for_execution** - Ensure plan is executable
7. **execute_batch** - Execute tasks with agents
8. **evaluate_agent_response** - Validate responses
9. **generate_final_response** - Create user-facing response
10. **save_history** - Persist conversation

**Execution Subgraph:**
- Used for post-approval execution
- Skips planning phase
- Starts at `execute_batch`

**Checkpointer:** `MemorySaver` for conversation persistence

---

### 📝 Message Management
**Module:** `orchestrator/message_manager.py`  
**Class:** `MessageManager`

**Features:**
- Deduplication via content hashing
- Timestamp and ID preservation
- Message ordering validation
- Safe merging of message lists

---

### 🎯 State Management
**Module:** `orchestrator/state.py`

**Reducers:**
- `or_overwrite` - Boolean values
- `concat_reducer` - String concatenation
- `overwrite_reducer` - Replace values
- `add_messages` - LangGraph message addition

---

## Error Handling & Logging

### 📋 Logging Configuration
```python
# Backend logs (main.py)
logger = logging.getLogger("uvicorn.error")

# Orchestrator logs (separate file)
orchestrator_logger = logging.getLogger("AgentOrchestrator")
# File: logs/orchestrator_temp.log (overwrites per conversation)
```

### ⚠️ WebSocket Error Categories
```python
{
  "database": "Database connection error",
  "authorization": "Permission denied",
  "timeout": "Request timeout",
  "resource_not_found": "Resource not found",
  "validation": "Invalid parameters",
  "unknown": "Unexpected error"
}
```

### 🔒 Safety Features
- WebSocket connection state checking
- Graceful disconnection handling
- Automatic reconnection support
- Error response truncation (max 150 chars)

---

## Authentication & Security

### 🔐 Clerk Integration
**Header:** `Authorization: Bearer {jwt_token}`

**Functions:**
- `get_user_from_request(request)` - Extract user from JWT
- `get_current_user_id()` - FastAPI dependency
- `verify_clerk_token(token)` - Verify JWT signature

**User ID Sources:** `sub`, `user_id`, or `id` from JWT

### 🔒 Credential Encryption
**Module:** `utils/encryption.py`

```python
from utils.encryption import encrypt, decrypt

encrypted = encrypt("my_api_key")
decrypted = decrypt(encrypted)
```

**Key:** `ENCRYPTION_KEY` environment variable (Fernet)

### 🛡️ CORS Configuration
```python
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

---

## Environment Variables

### Required
```bash
DATABASE_URL=postgresql://user:pass@host/db
CEREBRAS_API_KEY=sk-...
ENCRYPTION_KEY=...  # For credential storage
```

### Optional
```bash
GROQ_API_KEY=...
OPENAI_API_KEY=...
COMPOSIO_API_KEY=...
COMPOSIO_EMAIL_CONNECTION_ID=conn_...
```

---

## Startup Sequence

1. **Database Migrations** - `alembic upgrade head`
2. **Agent Sync** - Sync `Agent_entries/*.json` to DB
3. **Agent Servers** - Start all agent servers (`start_agents_async()`)
4. **Health Checker** - Background task for agent health
5. **Workflow Scheduler** - Load active schedules from DB
6. **Ready** - FastAPI app accepting requests

**Log Output:**
```
✅ Database migrations applied successfully
✅ Agent sync completed successfully
✓ Agents started in background
✓ Health checker started
✓ Workflow scheduler initialized with 3 jobs loaded
APPLICATION STARTUP COMPLETED
```

---

## Performance Considerations

### 💡 Optimizations
- **Lazy Model Loading** - SentenceTransformer loaded on first use
- **Eager Loading** - SQLAlchemy `joinedload` for related data
- **Compression** - Artifacts >10KB auto-compressed
- **Streaming** - WebSocket for real-time updates
- **Batching** - Tasks grouped for parallel execution
- **Caching** - Agent status cached in memory

### 📊 Token Budgets
- Orchestrator: 1,000,000 tokens
- Context optimization for large states
- Artifact references instead of full content

---

## Testing & Development

### 🧪 Testing Endpoints
```bash
# Health check
curl http://localhost:8000/api/health

# Agent status
curl http://localhost:8000/api/agent-servers/status

# Search agents
curl "http://localhost:8000/api/agents/search?capabilities=weather"

# WebSocket test (wscat)
wscat -c ws://localhost:8000/ws/chat
> {"prompt": "test", "owner": {"user_id": "test"}}
```

### 🔧 Development Mode
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Features:**
- Auto-reload on file changes
- WebSocket ping/pong (20s intervals)
- Agent server auto-restart
- Detailed error tracebacks

---

## Common Workflows

### 1. New Conversation
```
Client → POST /api/chat
  ↓
Backend → Orchestrator (LangGraph)
  ↓ (parse → search → rank → plan → execute)
Backend → Save to conversation_history/{id}.json
  ↓
Client ← Response with task_agent_pairs + final_response
```

### 2. WebSocket Streaming
```
Client → WS /ws/chat {"prompt": "...", "owner": {...}}
  ↓
Backend → Stream node updates
  ↓ analyze_request (10%)
  ↓ parse_prompt (20%)
  ↓ agent_directory_search (40%)
  ↓ execute_batch (70%)
  ↓ generate_final_response (90%)
  ↓ __end__ (100%)
Client ← Complete state
```

### 3. Save & Execute Workflow
```
Client → POST /api/workflows?thread_id=...
Backend → Extract blueprint from conversation
  ↓
Client → POST /api/workflows/{id}/schedule
Backend → Add to APScheduler
  ↓
Scheduler → Cron triggers
Backend → Execute via orchestrator
  ↓
Backend → Save results to WorkflowExecution
```

### 4. MCP Connection
```
Client → POST /api/connect/probe {"url": "..."}
Backend → Check auth requirements
  ↓
Client → POST /api/connect/ingest {"url": "...", "credentials": {...}}
Backend → Discover tools via MCP protocol
Backend → Create Agent + AgentEndpoint records
Backend → Save encrypted credentials
  ↓
Client → Use agent in conversations
```

---

## Troubleshooting

### Common Issues

**Agent Not Found:**
- Check `Agent_entries/*.json` exists
- Run `python manage.py sync`
- Verify agent status: `GET /api/agents/all`

**WebSocket Disconnect:**
- Check ping/pong settings (20s timeout)
- Verify JWT token not expired
- Check browser console for errors

**Schedule Not Running:**
- Verify cron expression (use UTC time)
- Check `is_active=true` in database
- Restart scheduler: `POST /api/admin/reload-schedules`

**Credentials Not Working:**
- Verify `ENCRYPTION_KEY` env var
- Check encrypted_credentials in database
- Test with `GET /api/credentials/status`

---

## API Rate Limits & Costs

### Agent Costs (Configurable)
- Weather API: $0.01/call
- Gmail Agent: $0.05/call
- Browser Agent: $0.10/call
- Custom agents: defined in `Agent_entries/`

### LLM Costs (Approximate)
- Cerebras: ~$0.001/request
- Planning: ~$0.002/conversation
- Execution: ~$0.003/task

### Database Queries
- Optimized with indexes on: `user_id`, `thread_id`, `agent_id`, `status`
- Vector search: cosine distance on 768D embeddings

---

## Future Enhancements

### Planned Features
- [ ] Real-time collaboration (multi-user threads)
- [ ] Agent marketplace with ratings
- [ ] Custom agent creation UI
- [ ] Advanced analytics dashboard
- [ ] Multi-LLM support (Anthropic, Gemini)
- [ ] Voice input/output
- [ ] Mobile app API
- [ ] Enterprise SSO

### Performance Improvements
- [ ] Redis caching for agent search
- [ ] CDN for static files
- [ ] Database read replicas
- [ ] Horizontal scaling with load balancer

---

**Last Updated:** December 8, 2025  
**Version:** 1.0  
**Maintainer:** Orbimesh Team

