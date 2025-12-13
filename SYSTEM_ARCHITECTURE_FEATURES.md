# Orbimesh System Architecture - Features & Interactions
**Updated:** December 10, 2025  
**Purpose:** Plain text documentation for optimization planning

**Related Documentation:**
- `BACKEND_API_DOCUMENTATION.md` - Complete REST API and WebSocket endpoint reference (updated)
- `FRONTEND_DOCUMENTATION.md` - Frontend architecture, components, state management (updated)

---

## CORE FEATURES OVERVIEW

### 1. CONVERSATIONAL ORCHESTRATION
**What it does:**
- User sends text prompt
- System breaks down into tasks
- Finds appropriate AI agents to execute tasks
- Executes tasks in parallel batches
- Combines results into human response

**Key capabilities:**
- Multi-agent coordination
- Task dependency management
- Real-time progress tracking
- Error handling and retry logic
- Cost estimation before execution

---

### 2. PLANNING MODE WITH APPROVAL
**What it does:**
- User enables "Planning Mode" toggle
- System creates execution plan but pauses
- Shows user estimated cost and task breakdown
- Waits for approval/modification/cancellation
- Executes only after user confirmation

**User actions:**
- Approve - Execute as planned
- Modify - Add instructions and re-plan
- Cancel - Stop and start over

---

### 3. SAVED WORKFLOWS
**What it does:**
- User completes a conversation successfully
- Can save entire conversation as reusable workflow
- Workflow stores: original prompt, task plan, agent selections
- Can execute saved workflow with same/different inputs
- Can schedule workflows or trigger via webhooks

**Workflow capabilities:**
- Name and describe workflows
- View execution history
- Clone workflows
- Schedule with cron expressions
- Create webhook triggers

---

### 4. REAL-TIME TASK TRACKING
**What it does:**
- Shows live status of each task during execution
- Displays running/completed/failed states
- Shows execution time and cost per task
- Updates UI in real-time via WebSocket
- Allows monitoring of multi-agent workflows

**Visual indicators:**
- Spinner for running tasks
- Green checkmark for completed
- Red X for failed
- Progress bar for overall completion

---

### 5. FILE UPLOAD & PROCESSING
**What it does:**
- User attaches files (images, PDFs, documents) to prompts
- System uploads files to backend storage
- Distributes files to agents that need them
- Agents process files (OCR, analysis, extraction)
- Results include processed file data

**File flow:**
- Frontend upload to `/api/upload`
- Backend stores in `storage/images` or `storage/documents`
- File metadata sent with WebSocket message
- Agents receive file paths or upload files themselves
- Content orchestrator manages file lifecycle

---

### 6. CANVAS SYSTEM (BROWSER AGENT)
**What it does:**
- Browser agent performs web automation
- Captures live screenshots during execution
- Streams HTML/visual results to frontend
- Shows interactive preview of web pages
- Allows browser/plan view toggle

**Browser capabilities:**
- Navigate websites
- Fill forms
- Click buttons
- Extract data
- Screenshot capture
- Live streaming to UI

---

### 7. CONTENT ORCHESTRATOR
**What it does:**
- Manages large content (artifacts) efficiently
- Compresses content >2KB into artifacts
- Provides content references instead of full data
- Expands artifacts when needed for LLM context
- Handles file uploads to agents automatically

**Artifact types:**
- task_result - Large task outputs
- canvas_content - Browser screenshots/HTML
- screenshot - Images from agents
- conversation - Full conversation snapshots

**Content lifecycle:**
1. Agent produces large output
2. Content service creates artifact
3. Stores in database with metadata
4. Returns reference ID to orchestrator
5. Orchestrator uses reference in state
6. Expands artifact when needed for LLM

---

### 8. AGENT MARKETPLACE
**What it does:**
- Browse all available AI agents
- Filter by capabilities, price, rating
- Search by keywords
- View agent details and endpoints
- Rate agents after use

**Agent types:**
- HTTP REST agents (custom APIs)
- MCP agents (Model Context Protocol)
- Built-in agents (browser, finance, mail, etc.)

---

### 9. MCP CONNECTIONS
**What it does:**
- Connect to external MCP servers
- Discover available tools automatically
- Store credentials per user
- Create agent endpoints from MCP tools
- Manage connections (add/delete)

**Supported MCP servers:**
- Supabase
- GitHub
- Linear
- Notion
- GitLab
- Custom MCP servers

**Connection flow:**
1. User provides MCP URL
2. System probes for auth requirements
3. User provides credentials
4. System connects and lists tools
5. Creates agent endpoints in database
6. Credentials encrypted and stored per user

---

### 10. CREDENTIAL MANAGEMENT
**What it does:**
- Store API keys securely per user
- Encrypt credentials using Fernet
- Associate credentials with agents
- Validate credentials before use
- Support multiple auth types (API key, OAuth2)

**Security features:**
- Per-user credential isolation
- Encryption at rest
- Clerk user authentication
- No credential sharing between users

---

## INTERACTION FLOWS

### FLOW 1: NEW CONVERSATION START
```
User types prompt + attaches files + enables planning mode
  ↓
Frontend: interactive-chat-interface.tsx
  - Calls uploadFiles() to upload attachments
  - Gets FileObject[] with backend file paths
  ↓
Frontend: conversation-store.ts
  - startConversation(prompt, files, planningMode, owner)
  - Checks for WebSocket connection
  - Retries up to 50 times if not connected
  ↓
WebSocket: Sends message to ws://localhost:8000/ws/chat
  {
    "prompt": "user text",
    "files": [FileObject],
    "planning_mode": true,
    "owner": { "user_id": "clerk_id" }
  }
  ↓
Backend: WebSocket handler in main.py
  - Generates thread_id (UUID)
  - Creates LangGraph State
  - Invokes graph.invoke(state, config)
  ↓
Backend: LangGraph Orchestrator (graph.py)
  Node 1: analyze_request
    - Determines if needs multi-agent processing
    - Sends WebSocket event: {"node": "analyze_request", "progress": 10}
  ↓
  Node 2: parse_prompt
    - Extracts tasks from user prompt using LLM
    - Returns ParsedRequest with tasks list
    - Sends event: {"node": "parse_prompt", "data": {tasks}}
  ↓
  Node 3: agent_directory_search
    - Searches database for agents matching task capabilities
    - Uses vector similarity (SentenceTransformer embeddings)
    - Returns list of candidate agents
    - Sends event: {"node": "agent_directory_search", "data": {agents}}
  ↓
  Node 4: rank_agents
    - Ranks agents by relevance, cost, rating
    - Selects best agent for each task
    - Returns TaskAgentPair[] mapping
    - Sends event: {"node": "rank_agents"}
  ↓
  Node 5: plan_execution
    - Creates batched execution plan (parallel-safe groups)
    - Estimates total cost
    - Sends event: {"node": "plan_execution", "data": {plan}}
  ↓
  Node 6: validate_plan_for_execution
    - If planning_mode=true: PAUSE HERE
    - Sets approval_required=true
    - Sends event: {"node": "__user_input_required__", "data": {
        "approval_required": true,
        "task_plan": [...],
        "estimated_cost": 0.12,
        "task_count": 3
      }}
    - Waits for user_response="approve" or "cancel"
  ↓
Frontend: WebSocket receives __user_input_required__
  - Updates store: approval_required=true
  - PlanApprovalModal opens automatically
  - Shows task breakdown, cost, agent assignments
  ↓
User clicks "Accept & Execute"
  ↓
Frontend: handleAcceptAndExecute()
  - Calls continueConversation('approve')
  - Sends user_response via WebSocket
  ↓
Backend: Receives user_response="approve"
  - Updates state with user's response
  - Continues graph execution
  ↓
  Node 7: execute_batch
    - For each batch (parallel group):
      - Prepares content for agents (content_orchestrator)
      - Calls agents via HTTP POST
      - Sends task_started events
      - Waits for responses
      - Captures agent outputs
      - Sends task_completed events
    - Stores results in state.completed_tasks
  ↓
  Node 8: evaluate_agent_response
    - Validates agent responses
    - Checks for errors
    - Aggregates results
  ↓
  Node 9: generate_final_response
    - Expands artifacts for LLM context
    - Summarizes all task results
    - Uses LLM to create human-friendly response
    - Returns final_response string
  ↓
  Node 10: save_history
    - Compresses state using content_orchestrator
    - Saves to conversation_history/{thread_id}.json
    - Sends event: {"node": "__end__", "data": {final_response}}
  ↓
Frontend: WebSocket receives __end__
  - Updates store with final state
  - Shows final response in chat
  - Status changes to 'completed'
  - Thread ID saved to localStorage
```

---

### FLOW 2: SAVED WORKFLOW EXECUTION
```
User navigates to /workflows
  ↓
Frontend: workflows/page.tsx
  - Fetches workflows from GET /api/workflows
  - Displays list with name, description, created date
  ↓
User clicks "Execute" on a workflow
  ↓
Frontend: Redirects to /?threadId=new&workflow_id={id}&executeNow=true
  ↓
Frontend: page.tsx useEffect
  - Detects workflow_id and executeNow params
  - Calls POST /api/workflows/{id}/create-conversation
  ↓
Backend: Creates thread with pre-seeded plan
  - Loads workflow blueprint from database
  - Creates new thread_id
  - Stores original_prompt, task_plan, task_agent_pairs in thread
  - Sets status='planning_complete'
  - Returns thread_id
  ↓
Frontend: Receives thread_id
  - Calls loadConversation(thread_id)
  - Shows plan in UI with "Accept & Execute" and "Modify" buttons
  ↓
User clicks "Accept & Execute"
  ↓
Frontend: handleAcceptPlan()
  - Calls continueConversation('approve')
  ↓
Backend: Receives approve on pre-seeded thread
  - Skips planning nodes (already have plan)
  - Goes directly to execute_batch node
  - Executes saved task_agent_pairs
  - Generates final response
  ↓
Frontend: Receives results via WebSocket
  - Shows execution progress
  - Displays final response
```

---

### FLOW 3: WORKFLOW MODIFICATION
```
User has saved workflow loaded (planning_complete status)
  ↓
User types modification in text box: "also include weather forecast"
  ↓
User clicks "Modify" button
  ↓
Frontend: handleSubmit() detects planning_complete status
  - Calls continueConversation(modification_text)
  - Sends to backend as user message
  ↓
Backend: Receives modification request
  - Current state has original_prompt + task_plan
  - User's modification triggers re-planning
  - Combines original_prompt + modification
  - Runs through parse_prompt again
  - Creates new task list
  - Searches for agents again
  - Creates updated plan
  - Sends new plan for approval
  ↓
Frontend: Receives new __user_input_required__
  - Shows updated plan with new tasks
  - User can approve/modify/cancel again
```

---

### FLOW 4: FILE UPLOAD TO AGENTS
```
User attaches PDF document to prompt
  ↓
Frontend: Uploads via POST /api/upload
  - Sends FormData with file
  - Receives FileObject: {file_name, file_path, file_type}
  ↓
Backend: Stores file in storage/documents/
  ↓
WebSocket: Sends prompt with uploaded_files metadata
  ↓
Backend: Orchestrator node: execute_batch
  ↓
Content Orchestrator: prepare_content_for_agent()
  - Checks if agent requires file upload
  - Looks for agent's /upload endpoint
  - Checks if endpoint needs file_id parameter
  ↓
If agent needs upload:
  - Reads file from storage/documents/
  - POSTs file to agent's /upload endpoint
  - Receives agent's file_id
  ↓
Content Orchestrator: inject_content_id_into_payload()
  - Adds file_id to agent request payload
  ↓
Agent receives request with file_id
  - Processes file (OCR, analysis, etc.)
  - Returns results
  ↓
Content Orchestrator: capture_agent_outputs()
  - Detects large outputs (>2KB)
  - Creates artifacts for large content
  - Stores artifact in database
  - Returns artifact reference
  ↓
Backend: Stores task result with artifact reference
  - Instead of full data: {"_artifact_ref": {"id": "art_123"}}
  ↓
Frontend: Receives final response
  - Artifact data expanded for display
  - Shows processed file results
```

---

### FLOW 5: REAL-TIME TASK TRACKING
```
Backend: execute_batch node
  - For each task in batch:
  
  Send task_started event:
    {
      "node": "task_started",
      "task_name": "Fetch weather data",
      "agent_name": "Weather API Agent"
    }
  ↓
Frontend: WebSocket receives task_started
  - Updates task_statuses[taskName] = {status: 'running', start_time}
  - UI shows spinner next to task
  ↓
Backend: Agent completes task
  
  Send task_completed event:
    {
      "node": "task_completed",
      "task_name": "Fetch weather data",
      "execution_time": 1.23,
      "cost": 0.002,
      "result": {...}
    }
  ↓
Frontend: WebSocket receives task_completed
  - Updates task_statuses[taskName] = {
      status: 'completed',
      execution_time: 1.23,
      cost: 0.002
    }
  - UI shows green checkmark
  - Displays execution time
  ↓
If task fails:
  
  Send task_failed event:
    {
      "node": "task_failed",
      "task_name": "Fetch weather data",
      "error": "API key invalid"
    }
  ↓
Frontend: Updates task_statuses
  - Shows red X
  - Displays error message
```

---

### FLOW 6: BROWSER AGENT CANVAS
```
Task assigned to browser automation agent
  ↓
Backend: Calls browser agent at localhost:8001/browse
  - Sends URL and actions to perform
  ↓
Browser Agent (Playwright):
  - Launches browser
  - Navigates to URL
  - Performs actions (click, type, scroll)
  - Takes screenshots at each step
  ↓
Browser Agent: Sends live updates to backend
  - POSTs to /api/canvas/update
  - Payload: {
      thread_id,
      canvas_content: "<html>...</html>",
      screenshot_base64: "data:image/png;base64,...",
      current_action: "Clicking submit button"
    }
  ↓
Backend: Receives canvas update
  - Stores in conversation_store[thread_id]
  - Broadcasts via WebSocket
  
  Event: {"node": "__live_canvas__", "data": {
    canvas_content,
    screenshot_base64,
    current_action
  }}
  ↓
Frontend: WebSocket receives __live_canvas__
  - Updates browser_view in store
  - Renders in OrchestrationDetailsSidebar
  - Shows live screenshot
  - Shows current action text
  - User can toggle between browser view / plan view
  ↓
Browser Agent: Completes task
  - Returns final HTML or extracted data
  - Sends task_completed event
  ↓
Frontend: Shows final result in chat
```

---

### FLOW 7: MCP CONNECTION
```
User navigates to /connections
  ↓
User enters MCP URL: https://mcp.supabase.com/mcp
  ↓
Frontend: Calls POST /api/connect/probe
  - Sends: {url: "https://mcp.supabase.com/mcp"}
  ↓
Backend: Probes MCP server
  - Attempts connection to /sse endpoint
  - Checks for auth requirements
  - Returns: {
      status: "auth_required",
      auth_types: ["api_key"],
      fields_needed: [{name: "api_key", type: "string"}]
    }
  ↓
Frontend: Shows credential input form
  - User enters API key
  ↓
Frontend: Calls POST /api/connect/ingest
  - Sends: {
      url,
      credentials: {api_key: "..."},
      agent_name: "Supabase",
      user_id: "clerk_id"
    }
  ↓
Backend: Connects to MCP server
  - Establishes SSE connection with credentials
  - Sends tools/list request
  - Receives list of available tools
  ↓
Backend: Creates agent endpoints
  - For each MCP tool:
    - Creates AgentEndpoint in database
    - Stores tool name, description, parameters
    - Associates with agent_id
  ↓
Backend: Stores credentials
  - Encrypts credentials using Fernet
  - Stores in AgentCredential table
  - Links to user_id + agent_id
  ↓
Backend: Returns success
  ↓
Frontend: Shows connection in list
  - User can now use MCP tools in conversations
  - Tools appear in agent search results
```

---

### FLOW 8: SCHEDULED WORKFLOW
```
User has saved workflow
  ↓
User clicks "Schedule" button
  ↓
Frontend: Opens ScheduleWorkflowDialog
  - Shows cron expression builder
  - Input template fields
  ↓
User configures schedule:
  - Cron: "0 9 * * *" (daily at 9 AM UTC)
  - Inputs: {location: "NYC"}
  ↓
Frontend: Calls POST /api/workflows/{id}/schedule
  - Sends: {
      cron_expression: "0 9 * * *",
      input_template: {location: "NYC"}
    }
  ↓
Backend: Creates schedule
  - Stores in WorkflowSchedule table
  - Adds to APScheduler
  - Calculates next_run_at
  - Returns schedule_id
  ↓
APScheduler: At scheduled time (9 AM UTC)
  - Triggers job
  - Calls _async_execute_workflow()
  ↓
Backend: Executes workflow
  - Creates new WorkflowExecution record
  - Loads workflow blueprint
  - Merges input_template with blueprint
  - Runs orchestration
  - Saves results to execution record
  ↓
Backend: Updates schedule
  - Sets last_run_at
  - Calculates next_run_at
  ↓
User can view execution history:
  - GET /api/schedules/{id}/executions
  - Shows all past runs with status, outputs, errors
```

---

## STATE MANAGEMENT

### FRONTEND STATE (Zustand)
**Location:** lib/conversation-store.ts

**Core state fields:**
```typescript
thread_id: string | undefined
status: 'idle' | 'processing' | 'completed' | 'waiting_for_user' | 'error' | 'planning_complete'
messages: Message[]
task_agent_pairs: TaskAgentPair[]
final_response: string | undefined
metadata: {
  currentStage: string        // analyze_request, parse_prompt, etc.
  stageMessage: string         // "Breaking down your request..."
  progress: number             // 0-100
  original_prompt: string      // For saved workflows
  completed_tasks: any[]
  parsed_tasks: any[]
}
uploaded_files: FileObject[]
approval_required: boolean
estimated_cost: number
task_count: number
task_plan: any[]
task_statuses: Record<string, TaskStatus>  // Real-time task tracking
canvas_content: string
canvas_type: 'html' | 'markdown'
has_canvas: boolean
browser_view: string
current_view: 'browser' | 'plan'
```

**Actions:**
- startConversation()
- continueConversation()
- loadConversation()
- resetConversation()
- _setConversationState() - Internal, called by WebSocket

**State updates triggered by:**
- User actions (send message, approve plan, etc.)
- WebSocket events (node updates, task events, canvas updates)
- HTTP responses (load conversation)

---

### BACKEND STATE (LangGraph)
**Location:** orchestrator/state.py

**State dict structure:**
```python
{
  "messages": List[BaseMessage],           # LangChain messages
  "user_prompt": str,
  "parsed_request": ParsedRequest | None,
  "agents": List[AgentCard],
  "task_agent_pairs": List[TaskAgentPair],
  "execution_plan": ExecutionPlan | None,
  "completed_tasks": List[CompletedTask],
  "final_response": str | None,
  "uploaded_files": List[FileObject],
  "thread_id": str,
  "user_id": str,
  "planning_mode": bool,
  "approval_required": bool,
  "user_response": str | None,
  "task_plan": List[PlannedTask],
  "estimated_cost": float,
  "canvas_content": str | None,
  "canvas_data": dict | None,
  "requires_approval": bool,
  "error": str | None
}
```

**State persistence:**
- MemorySaver checkpointer (in-memory during execution)
- JSON file after completion (conversation_history/)
- Compressed using content_orchestrator before saving
- Artifacts stored separately in database

---

## WEBSOCKET PROTOCOL

### CLIENT TO SERVER MESSAGES
**Format:**
```json
{
  "prompt": "user input text",
  "thread_id": "uuid-or-null",
  "planning_mode": true,
  "owner": {
    "user_id": "clerk_user_id",
    "email": "user@example.com"
  },
  "uploaded_files": [
    {
      "file_name": "doc.pdf",
      "file_path": "storage/documents/uuid.pdf",
      "file_type": "application/pdf"
    }
  ],
  "user_response": "approve"  // For continuing conversations
}
```

---

### SERVER TO CLIENT EVENTS

**Start Event:**
```json
{
  "node": "__start__",
  "thread_id": "uuid",
  "message": "Starting workflow orchestration...",
  "timestamp": 1702234567
}
```

**Progress Events:**
```json
{
  "node": "parse_prompt",
  "thread_id": "uuid",
  "data": {
    "tasks_identified": 3,
    "task_names": ["Fetch weather", "Create chart", "Send email"]
  },
  "progress_percentage": 20
}
```

**Task Events:**
```json
// Started
{
  "node": "task_started",
  "thread_id": "uuid",
  "task_name": "Fetch weather data",
  "agent_name": "Weather API Agent"
}

// Completed
{
  "node": "task_completed",
  "thread_id": "uuid",
  "task_name": "Fetch weather data",
  "execution_time": 1.23,
  "cost": 0.002,
  "result": {...}
}

// Failed
{
  "node": "task_failed",
  "thread_id": "uuid",
  "task_name": "Fetch weather data",
  "error": "API key invalid",
  "error_category": "authentication"
}
```

**Approval Request:**
```json
{
  "node": "__user_input_required__",
  "thread_id": "uuid",
  "data": {
    "approval_required": true,
    "question_for_user": "Approve execution?",
    "task_plan": [...],
    "task_agent_pairs": [...],
    "estimated_cost": 0.12,
    "task_count": 3
  }
}
```

**Canvas Update:**
```json
{
  "node": "__live_canvas__",
  "thread_id": "uuid",
  "data": {
    "canvas_content": "<html>...</html>",
    "screenshot_base64": "data:image/png;base64,...",
    "current_action": "Filling login form"
  }
}
```

**Completion:**
```json
{
  "node": "__end__",
  "thread_id": "uuid",
  "data": {
    "final_response": "Here are your results...",
    "completed_tasks": [...],
    "task_agent_pairs": [...],
    "metadata": {...}
  },
  "status": "completed"
}
```

**Error:**
```json
{
  "node": "__error__",
  "thread_id": "uuid",
  "error": "Failed to connect to agent",
  "error_type": "AgentConnectionError",
  "error_category": "network",
  "details": {...}
}
```

---

## OPTIMIZATION OPPORTUNITIES

### PERFORMANCE BOTTLENECKS

**1. WebSocket Reconnection**
- Current: Retries up to 50 times with delays
- Issue: Can delay message sending
- Optimization: Use exponential backoff, connection pooling

**2. File Upload Flow**
- Current: Sequential upload then WebSocket send
- Issue: Delays conversation start
- Optimization: Parallel upload, progressive enhancement

**3. Artifact Expansion**
- Current: Expands all artifacts for LLM context
- Issue: Can be slow for many large artifacts
- Optimization: Selective expansion, caching, lazy loading

**4. Agent Search**
- Current: Vector similarity on every request
- Issue: Sentence transformer model loading
- Optimization: Cache embeddings, pre-compute similarities

**5. State Compression**
- Current: Compresses entire state before saving
- Issue: Blocking operation on main thread
- Optimization: Background worker, incremental compression

---

### SCALABILITY CONCERNS

**1. In-Memory Conversation Store**
- Current: All active conversations in RAM
- Limit: Memory grows with concurrent users
- Solution: Redis/distributed cache

**2. WebSocket Connections**
- Current: One connection per user
- Limit: Server connection limits
- Solution: Connection pooling, sticky sessions

**3. Agent Health Checking**
- Current: Polls every 3 seconds
- Limit: Network overhead with many agents
- Solution: Event-driven health updates

**4. File Storage**
- Current: Local filesystem
- Limit: Not cloud-scalable
- Solution: S3/blob storage

**5. Schedule Execution**
- Current: APScheduler in-process
- Limit: Single server instance
- Solution: Distributed task queue (Celery/Redis)

---

### CODE REDUNDANCIES

**1. API Clients (CRITICAL)**
- api-client.ts (active, with auth)
- api-client-new.ts (duplicate, no auth)
- api-unified.ts (incomplete class)
- **Action:** Remove duplicates, keep api-client.ts

**2. Workflow Pages**
- /workflows (full featured)
- /saved-workflows (similar features)
- **Action:** Merge into single page

**3. Mock Data**
- lib/mock-data.ts with hardcoded agents
- **Action:** Remove if not used for testing

**4. Message Deduplication**
- Implemented in both conversation-store.ts and message_manager.py
- **Action:** Unify logic or clarify responsibilities

---

### MISSING FEATURES

**1. Offline Support**
- No service worker
- No local caching
- **Add:** PWA capabilities, IndexedDB caching

**2. Error Recovery**
- Limited retry logic
- No graceful degradation
- **Add:** Exponential backoff, fallback agents

**3. Real-time Collaboration**
- No multi-user conversation sharing
- **Add:** Shared thread access, presence indicators

**4. Analytics**
- Basic metrics only
- No user behavior tracking
- **Add:** Detailed analytics, performance monitoring

**5. Testing**
- No unit tests visible
- No integration tests
- **Add:** Jest/Vitest tests, E2E with Playwright

---

### SECURITY IMPROVEMENTS

**1. Credential Storage**
- Currently encrypted but in same database
- **Add:** Separate secrets manager (AWS Secrets, Vault)

**2. Rate Limiting**
- Not implemented
- **Add:** Per-user, per-endpoint rate limits

**3. Input Validation**
- Basic Pydantic validation
- **Add:** Comprehensive sanitization, XSS prevention

**4. CORS**
- Currently allows all origins
- **Add:** Whitelist specific domains

**5. WebSocket Authentication**
- No explicit token validation
- **Add:** Token-based WS auth, session management

---

## TECHNICAL DEBT

**Priority 1 (High Impact):**
1. Remove duplicate API clients
2. Add comprehensive error handling
3. Implement proper TypeScript strict mode
4. Add unit tests for core functions
5. Document environment variables

**Priority 2 (Medium Impact):**
1. Consolidate workflow pages
2. Optimize WebSocket reconnection
3. Add request caching layer
4. Improve state serialization
5. Add logging standards

**Priority 3 (Low Impact):**
1. Clean up unused components
2. Standardize naming conventions
3. Add JSDoc comments
4. Optimize bundle size
5. Add performance monitoring

---

## DEPENDENCIES & VERSIONS

**Frontend:**
- Next.js: 15.2.4
- React: 19
- Zustand: 5.0.8
- Clerk: 6.34.0
- Radix UI: ~1.x (30+ components)
- Tailwind CSS: 4.1.9
- React Hook Form: 7.60.0
- Zod: 3.25.67

**Backend:**
- FastAPI: (version from requirements.txt)
- SQLAlchemy: (ORM)
- LangGraph: (orchestration)
- LangChain: (LLM integration)
- Cerebras: (primary LLM)
- APScheduler: (cron jobs)
- pgvector: (vector search)

---

## FILE STRUCTURE SUMMARY

**Frontend:**
```
app/
  page.tsx                    # Main chat interface
  (dashboard)/
    agents/                   # Agent marketplace
    metrics/                  # Analytics
    credentials/              # API key management
  workflows/                  # Workflow management
  saved-workflows/            # Workflow library
  schedules/                  # Scheduled workflows
  connections/                # MCP connections

components/
  interactive-chat-interface.tsx    # Main chat UI
  orchestration-details-sidebar.tsx # Right sidebar
  plan-approval-modal.tsx           # Approval dialog
  task-builder.tsx                  # Task construction
  ui/ (30+ shadcn components)

lib/
  conversation-store.ts       # Zustand state
  api-client.ts              # HTTP client (ACTIVE)
  auth-fetch.ts              # Clerk auth wrapper
  types.ts                   # TypeScript interfaces

hooks/
  use-websocket-conversation.ts  # WebSocket manager
```

**Backend:**
```
main.py                      # FastAPI app, 3671 lines
orchestrator/
  graph.py                   # LangGraph workflow, 5547 lines
  state.py                   # State management
  content_orchestrator.py    # Content/artifact management, 797 lines
  message_manager.py         # Message deduplication

models.py                    # SQLAlchemy models
schemas.py                   # Pydantic schemas
database.py                  # DB connection

routers/
  connect_router.py          # MCP connections
  credentials_router.py      # Credential management

services/
  unified_content_service.py # Artifact storage
  workflow_scheduler.py      # APScheduler

agents/
  browser_automation_agent.py   # Port 8001
  document_analysis_agent.py    # Port 8002
  finance_agent.py              # Port 8003
  (8 total agents)

Agent_entries/
  *.json                     # Agent definitions
```

---

## KEY METRICS

**Lines of Code:**
- Backend main.py: 3,671
- Backend orchestrator/graph.py: 5,547
- Frontend conversation-store.ts: 749
- Frontend use-websocket-conversation.ts: 983
- Total estimated: 25,000+ lines

**Database Tables:**
- Agents: 10 tables (Agent, AgentCapability, AgentEndpoint, etc.)
- Workflows: 4 tables (Workflow, WorkflowExecution, etc.)
- Conversations: 5 tables (UserThread, ConversationPlan, etc.)
- Total: 20+ tables

**API Endpoints:**
- REST: 40+ endpoints
- WebSocket: 2 endpoints (/ws/chat, /ws/workflow/execute)

**Agent Servers:**
- 8 standalone agents on ports 8001-8008
- Each with /health, /upload, and task endpoints

---

## SUMMARY

**System Architecture:**
Multi-tier conversational AI orchestration platform with real-time WebSocket communication, file processing, workflow automation, and external integrations.

**Core Strength:**
Flexible agent-based task execution with planning, approval, and real-time tracking.

**Main Challenge:**
State management complexity across WebSocket, LangGraph, and Zustand with artifact compression.

**Optimization Focus:**
- Remove redundant code (API clients, workflows pages)
- Improve WebSocket reliability (reconnection, auth)
- Optimize artifact expansion (caching, selective loading)
- Add comprehensive testing
- Enhance error handling

**Next Steps:**
1. Remove api-client-new.ts and api-unified.ts
2. Consolidate workflow management pages
3. Add unit tests for core flows
4. Implement connection pooling
5. Add performance monitoring
