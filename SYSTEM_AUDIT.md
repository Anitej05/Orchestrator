# 🔍 Orbimesh — Scoped System Audit

**Date:** February 10, 2026  
**Scope:** Integrations layer, Gmail Agent, Zoho Tools, Frontend (full)  
**Out of Scope:** Orchestrator/Backend core (work in progress separately)  
**Target Deployment:** Azure (Docker)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Integrations Layer (Composio)](#2-integrations-layer-composio)
3. [Gmail Agent Audit](#3-gmail-agent-audit)
4. [Zoho Tools Audit](#4-zoho-tools-audit)
5. [Frontend Audit](#5-frontend-audit)
6. [Cloud Deployment Notes (Frontend + Integrations)](#6-cloud-deployment-notes)
7. [Action Plan](#7-action-plan)

---

## 1. Executive Summary

| Area | Critical | High | Medium | Low |
|------|----------|------|--------|-----|
| Integrations Layer | 2 | 3 | 4 | 1 |
| Gmail Agent | 1 | 3 | 3 | — |
| Zoho Tools | 2 | 2 | 2 | — |
| Frontend | 4 | 5 | 3 | 2 |
| **Total** | **9** | **13** | **12** | **3** |

### Top 5 Risks if Deployed Today

1. **Connections page is completely broken** — Frontend calls `/api/integrations/*` but no router exists in backend
2. **Hardcoded `localhost` in ~80% of frontend API calls** — Nothing works outside dev machine
3. **Zoho tools reference a directory that doesn't exist** — `agents/zoho_books/` is missing → runtime ImportError
4. **Gmail agent is invisible to orchestrator** — No SKILL.md, no registry entry, non-standard response format
5. **3 dead API client files shipping in bundle** — Wasted build size, developer confusion

---

## 2. Integrations Layer (Composio)

### 2.1 What Exists Today

```
backend/services/integrations/
├── __init__.py              → exports get_auth_manager, get_tool_manager
├── composio_auth.py (1044 lines) → ComposioAuthManager singleton
│   ├── start_auth_flow()          — Initiates OAuth via Composio SDK
│   ├── check_connection_status()  — Polls Composio + syncs to local DB
│   ├── get_connection_for_agent() — Decrypts connection ID for agent use
│   ├── disconnect_app()           — Removes connection from Composio + DB
│   ├── refresh_connection()       — Token refresh
│   ├── enable/disable_connection()— Pause/resume
│   └── _save_connection_to_db()   — Encrypted storage in UserConnection table
│
└── composio_tools.py (274 lines) → ComposioToolManager singleton
    ├── get_tools_for_user()       — Returns LangChain BaseTool list
    ├── execute_tool()             — Direct tool execution
    └── get_invoice_tools_for_user() — Convenience for Zoho Books
```

### 2.2 Issues Found

#### 🔴 CRITICAL

| # | Issue | Detail | What To Build Instead |
|---|-------|--------|----------------------|
| 1 | **No integrations router mounted in backend** | Frontend `connections/page.tsx` calls 8 `/api/integrations/*` endpoints. None exist. The `connect_router` at `/api/connect` handles MCP connections only, NOT Composio OAuth. | **Build `routers/integrations_router.py`** — a thin FastAPI router that wraps every `ComposioAuthManager` method as an endpoint. Mount it in `main.py` with prefix `/api/integrations`. Needs 8 endpoints (see §2.3 below). |
| 2 | **Logger used before it's defined** | `composio_auth.py` line 29 does `logger.warning(...)` in an `except ImportError` block, but `logger` isn't created until line 35. Same issue in `composio_tools.py` line 19. If Composio SDK is missing → instant `NameError` crash on import. | **Move `logger = logging.getLogger(...)` to the top of each file**, before any try/except that references it. 2-line fix per file. |

#### 🟠 HIGH

| # | Issue | Detail | What To Build Instead |
|---|-------|--------|----------------------|
| 3 | **Duplicate method definitions in `composio_auth.py`** | `refresh_connection()` is defined twice — once taking `(user_id, app_slug)` (line ~590) and again taking `(connection_id)` (line ~920). Same for `disable_connection()` and `enable_connection()`. Python's second definition silently overwrites the first. | **merge the `(user_id, app_slug)` and `(connection_id)` versions** (they're more useful for the router).|
| 4 | **Encryption key fallback is insecure** | If `CONNECTION_ENCRYPTION_KEY` is not set, a deterministic key is derived from `COMPOSIO_API_KEY` via SHA-256. Anyone with the API key can decrypt all stored connection IDs. | **In production, require the key explicitly.** Add an env check: if `ENV=production` and no `CONNECTION_ENCRYPTION_KEY` → fail startup with a clear error message and a command to generate one (`Fernet.generate_key()`). |
| 5 | **`_get_auth_config_id` is hardcoded to 5 apps** | Only gmail, zohobooks, github, slack, notion. Adding a new integration requires a code change. | **Load the mapping from `integrations.json` or a DB table.** The file already exists at `backend/data/integrations.json` — extend its schema to include a `composio_auth_config_env_var` field per integration. Then the method just loops through the config. |

#### 🟡 MEDIUM

| # | Issue | Detail | What To Build Instead |
|---|-------|--------|----------------------|
| 6 | **No connection expiry/TTL** | Connections stored as "active" forever in DB. If Composio revokes a token, local DB still says "active" → agent calls fail silently with cryptic errors. | **Add a `last_verified_at` timestamp** to `UserConnection`. On every `get_connection_for_agent()` call, if `last_verified_at` is older than 1 hour, do a quick `check_connection_status()` ping. This prevents stale connections from accumulating. |
| 7 | **Singletons not thread-safe** | `_auth_manager` and `_tool_manager` use simple global variable pattern. Concurrent first-access could create duplicates. | **Initialize both in FastAPI's lifespan handler** (`app.on_startup`) and store on `app.state`. Or use `threading.Lock()` around the `if None` check. |
| 8 | **DB sessions leak risk** | Multiple methods do `db = SessionLocal()` then `try/finally: db.close()`. If exception occurs between creation and the try → session leaks. | **Use `contextmanager` pattern**: `with SessionLocal() as db:` everywhere. Or better: pass `db` as a parameter (injected by FastAPI `Depends(get_db)` in the router layer). |
| 9 | **No retry on Composio API calls** | `start_auth_flow`, `check_connection_status`, etc. make HTTP calls to Composio with no retry logic. Network blips → user sees error. | **Add `tenacity` retry decorator** on the methods that call Composio SDK: 3 retries, exponential backoff, only for network/timeout errors. |

#### 🟢 LOW

| # | Issue | Detail | What To Build Instead |
|---|-------|--------|----------------------|
| 10 | **Auth config map has `zoho_books` AND `zohobooks`** | Both map to the same env var. Not harmful but confusing. | **Normalize to one slug** (`zohobooks`) and document it. Add a deprecation warning for `zoho_books`. |

### 2.3 Integrations Router — What To Build

The frontend expects these 8 endpoints. Here's the mapping to existing `ComposioAuthManager` methods:

```
Router: /api/integrations (prefix)
Auth: All endpoints need Clerk JWT (get_current_user)

GET  /status/{user_id}
  → auth_manager.check_connection_status(user_id)
  → Returns: {connected_apps: [], pending_apps: [], all_toolkits: []}

POST /auth/start
  → Body: {user_id, app_slug, callback_url?}
  → auth_manager.start_auth_flow(user_id, app_slug, callback_url)
  → Returns: {redirect_url, connection_id}

POST /sync/{user_id}
  → auth_manager.check_connection_status(user_id)  # same as status but forces re-sync
  → Returns: same as /status

GET  /available
  → Read from integrations.json + merge with Composio supported apps
  → Returns: [{id, name, description, icon, is_connected}]

DELETE /{user_id}/{app_slug}
  → auth_manager.disconnect_app(user_id, app_slug)
  → Returns: {success, message}

POST /{user_id}/{app_slug}/refresh
  → auth_manager.refresh_connection(user_id, app_slug)
  → Returns: {success, refreshed_at}

POST /{user_id}/{app_slug}/enable
  → auth_manager.enable_connection(user_id, app_slug)
  → Returns: {success, status}

POST /{user_id}/{app_slug}/disable
  → auth_manager.disable_connection(user_id, app_slug)
  → Returns: {success, status}
```

**Ideation:** This router is purely a thin wrapper. No business logic. Just: authenticate user → call auth_manager → return JSON. The `ComposioAuthManager` already has all the logic — it just doesn't have an HTTP surface.

---

## 3. Gmail Agent Audit

### 3.1 What Exists Today

Two separate Gmail agents exist:

| | `mail_agent` (existing, integrated) | `gmail_agent` (new, standalone) |
|---|------|------|
| **Location** | `backend/agents/mail_agent/` | `backend/agents/gmail_agent/` |
| **Port** | 8040 | 8003 |
| **Registered in orchestrator** | ✅ Has SKILL.md, in agent registry | ❌ Not registered |
| **Auth model** | Composio MCP (single-tenant, hardcoded connection_id) | Composio SDK (multi-tenant, per-user) |
| **LLM** | Centralized `InferenceService` | Own `LLMClient` with manual Cerebras→NVIDIA→Groq failover |
| **Response format** | `AgentResponse` (UAP-compliant) | Raw dict |
| **CMS hooks** | ✅ | ❌ |
| **Canvas support** | ✅ | ❌ |
| **Memory** | CMS-backed + local cache | In-memory only (`AgentMemory` dict) |
| **Tools** | Raw HTTP POST to Composio MCP endpoint | Native Composio SDK — 23 wrapped tool methods |

> The duplicate is fine — you built `gmail_agent` separately for a reason (better SDK, per-user auth). The goal is to bring it into the platform properly.

### 3.2 What Needs To Happen (Recommended: Merge Strategy)

**The gmail_agent has the better tool layer and auth model.** The mail_agent has the better platform integration. Merge the best of both:

#### Step 1: Make `gmail_agent` Orchestrator-Compatible

| What | Current | What To Build Instead |
|------|---------|----------------------|
| **Response format** | Returns raw `{"success": True, "messages": [...]}` | **Wrap every endpoint's return** in `AgentResponse(status=COMPLETE/ERROR, result=data)`. Import from shared `schemas.py`. The orchestrator's Hands module parses `AgentResponse` — without it, results get silently dropped. |
| **SKILL.md** | Doesn't exist | **Create `agent_entries/templates/gmail_agent/SKILL.md`** describing: what the agent does, what endpoints it exposes, what input/output formats it uses, and what auth it needs. This is how the Brain discovers agent capabilities during planning. |
| **Agent registry entry** | Not in DB | **Add a registration call** in the agent's startup (or a migration script) that inserts into the `agents` table with `connection_config.base_url = "http://gmail-agent:8003"` (Docker service name). |
| **`/execute` endpoint** | Returns raw dict based on keyword matching | **Rebuild `/execute`** to return `AgentResponse`. The current keyword-matching approach (`if "search" in prompt`) is too brittle. Instead: accept the UAP payload format `{prompt, payload, task_id}` and use the existing granular endpoints internally. |

#### Step 2: Fix Auth Model

| What | Current | What To Build Instead |
|------|---------|----------------------|
| **`entity_id` parameter** | `tools.py` line 68: `entity_id=self.connection_id` | **Change to `entity_id=self.user_id`**. Composio SDK's `execute_action` uses `entity_id` to look up the user's connection automatically. Passing the connection_id instead of user_id breaks multi-account scenarios. |
| **Connection verification** | Calls `auth_mgr.get_connection_for_agent()` to get connection, then creates a new `Composio` client with the raw API key | **Simplify:** Just pass `user_id` to the Composio SDK. It handles connection resolution internally. The manual connection lookup is redundant when using `entity_id=user_id`. |

#### Step 3: Replace Standalone LLM Client

| What | Current | What To Build Instead |
|------|---------|----------------------|
| **`llm.py`** (330 lines) | Own `AsyncOpenAI` clients with manual failover chain (Cerebras→NVIDIA→Groq), own `strip_think_tags()`, own prompt templates | **Replace with centralized `InferenceService`**. The orchestrator already has a battle-tested LLM service with provider routing, retry logic, and cost tracking. Import and use it. Delete `llm.py`. The `strip_think_tags()` utility is useful — move it to `backend/utils/` as a shared utility. |

#### Step 4: Fix Memory Persistence

| What | Current | What To Build Instead |
|------|---------|----------------------|
| **`memory.py`** | Pure in-memory dict per user. Lost on restart. No size limits. | **Two options:** (a) Use the existing CMS (Content Management Service) that `mail_agent` uses — it persists conversation context. (b) If you want simpler: use Redis with TTL. Key: `gmail:{user_id}:context`, TTL: 24 hours. The `AgentMemory` class API can stay the same — just swap the backing store. |

#### Step 5: Service Cache Eviction

| What | Current | What To Build Instead |
|------|---------|----------------------|
| **`_service_cache`** in `agent.py` | `Dict[str, GmailService]` — grows forever, one entry per user who ever made a request | **Use `functools.lru_cache(maxsize=100)`** or a TTL cache (`cachetools.TTLCache`). Evict after 30 minutes of inactivity. This prevents memory growth in a long-running container. |

### 3.3 Gmail Agent Issues Summary

| # | Issue | Severity | What To Build Instead |
|---|-------|----------|----------------------|
| 1 | Not registered in orchestrator (no SKILL.md, no DB entry) | 🔴 Critical | Create SKILL.md + registration migration |
| 2 | Non-UAP `/execute` response format | 🟠 High | Wrap returns in `AgentResponse` |
| 3 | `entity_id=connection_id` instead of `user_id` | 🟠 High | Change to `entity_id=user_id` in `tools.py` |
| 4 | Own standalone LLM client (330 lines) | 🟠 High | Replace with centralized `InferenceService` |
| 5 | In-memory only state (lost on restart) | 🟡 Medium | Back with Redis or CMS |
| 6 | Service cache grows unbounded | 🟡 Medium | Add TTL eviction (cachetools) |
| 7 | Own `strip_think_tags()` duplicated | 🟡 Medium | Move to shared `backend/utils/text.py` |

---

## 4. Zoho Tools Audit

### 4.1 What Exists Today

```
backend/tools/zoho_books_helpers.py
├── get_zoho_books_tools()                → Uses composio_tools.get_tools_for_user() ✅ Works
├── check_zoho_books_connection()         → Uses composio_auth.get_auth_manager() ✅ Works
├── get_zoho_books_connect_url()          → Imports from agents.zoho_books.composio_client ❌ DOES NOT EXIST
├── disconnect_zoho_books()               → Imports from agents.zoho_books.composio_client ❌ DOES NOT EXIST
├── _check_approval_needed()              → Approval gate for destructive operations ✅ Good pattern
├── _execute_zoho_action()                → Executes via composio_tools ✅ Works
└── 9 @tool wrapper functions:
    ├── create_zoho_books_invoice
    ├── list_zoho_books_invoices
    ├── get_zoho_books_invoice
    ├── update_zoho_books_invoice
    ├── delete_zoho_books_invoice
    ├── void_zoho_books_invoice
    ├── manage_zoho_books_contacts
    ├── manage_zoho_books_items
    └── manage_zoho_books_bank_transactions
```

### 4.2 Issues Found

#### 🔴 CRITICAL

| # | Issue | Detail | What To Build Instead |
|---|-------|--------|----------------------|
| 1 | **Dead import: `agents.zoho_books.composio_client`** | `get_zoho_books_connect_url()` and `disconnect_zoho_books()` import from `agents.zoho_books.composio_client`. That directory **does not exist** in the codebase. Calling either function → `ImportError` every time. | **Rewrite both functions to use `ComposioAuthManager` directly.** The auth manager already has `start_auth_flow()` (for getting connect URLs) and `disconnect_app()`. Just import `get_auth_manager` and delegate to it. Example flow: `get_auth_manager().start_auth_flow(user_id, "zohobooks")` returns `{redirect_url}`. |
| 2 | **`zoho_books` agent spawn reference in `main.py`** | Line 822 in `main.py` has `("zoho_books", "zoho_books/zoho_books_agent.py")` in the agent subprocess list. The file doesn't exist → startup error or silent failure. | **Remove the dead reference from `main.py`'s agent spawn list.** Zoho Books doesn't need its own microservice — the tools in `zoho_books_helpers.py` are called directly by the orchestrator's Hands, not via HTTP. |

#### 🟠 HIGH

| # | Issue | Detail | What To Build Instead |
|---|-------|--------|----------------------|
| 3 | **Zoho tools invisible to orchestrator** | The 9 `@tool` decorated functions in `zoho_books_helpers.py` are never registered in `ToolRegistryService`. The orchestrator's Brain sees available tools from the registry — these aren't there. | **Two approaches:** (a) **SKILL.md approach** — Create `agent_entries/templates/zoho_books/SKILL.md` that describes all Zoho capabilities. The Brain reads SKILL.md files to understand what's available during planning. (b) **Direct tool registration** — In `ToolRegistryService.__init__()`, import and register the 9 tool functions so they appear in the Brain's tool list. Approach (a) is better because it lets the Brain understand the tools' purpose at planning time. |
| 4 | **No SKILL.md for Zoho Books** | Brain can't plan any Zoho operation because it doesn't know the agent exists. | **Create the SKILL.md with:** agent name, description ("Manages Zoho Books accounting — invoices, contacts, items, bills, bank transactions"), list of operations with input/output descriptions, and required auth ("Requires active Zoho Books connection via /connections page"). |

#### 🟡 MEDIUM

| # | Issue | Detail | What To Build Instead |
|---|-------|--------|----------------------|
| 5 | **Approval flow not wired to orchestrator** | `_execute_zoho_action()` returns `AgentResponse(status=NEEDS_INPUT)` for destructive operations, but this goes into the void — the orchestrator's human-in-the-loop mechanism expects the approval flag in its own state (`pending_approval` field), not in a tool's return value. | **Emit the approval request through the orchestrator's state mechanism.** When a Zoho tool needs approval: (1) Set `state.pending_approval = True` with the approval message, (2) Return `ActionResult(status="needs_approval", ...)` from Hands, (3) Let the graph's conditional routing handle the pause/resume. The Brain already knows how to handle this — the Zoho tool just needs to speak the same language. |
| 6 | **Import path only works from `backend/` CWD** | `from schemas import AgentResponse` is a relative-from-CWD import. In Docker, if `WORKDIR` is `/app` and the module is at `/app/backend/schemas.py`, this breaks. | **Use explicit paths:** `from backend.schemas import AgentResponse` or set `PYTHONPATH=/app/backend` in the Dockerfile. The latter is simpler and matches how other agents import. |

### 4.3 Ideation: What the Zoho Integration Should Look Like

```
Current (broken):
  tools/zoho_books_helpers.py → imports dead module → crashes

Target architecture:
  tools/zoho_books_helpers.py
    ├── get_zoho_books_tools()           → composio_tools.get_tools_for_user()
    ├── get_zoho_books_connect_url()     → composio_auth.get_auth_manager().start_auth_flow()  ← FIX
    ├── disconnect_zoho_books()          → composio_auth.get_auth_manager().disconnect_app()    ← FIX
    ├── check_zoho_books_connection()    → composio_auth.get_auth_manager().check_connection_status()
    └── 9 @tool wrappers                → _execute_zoho_action() → composio_tools

  agent_entries/templates/zoho_books/SKILL.md  ← NEW (so Brain can plan with it)

  main.py agent spawn list               → remove dead "zoho_books" reference
```

The core tool execution path (`get_zoho_books_tools` → `_execute_zoho_action`) is actually solid. The problems are at the edges: connection management (dead imports) and discoverability (no SKILL.md).

---

## 5. Frontend Audit

### 5.1 Dead Files — Delete Immediately

| File | Lines | Why Dead | What To Do |
|------|-------|----------|------------|
| `lib/api-client-new.ts` | ~300 | **Zero imports anywhere** in the codebase. Uses raw `fetch` with no auth. Endpoints partially wrong. | **Delete entirely.** Nothing depends on it. |
| `lib/api-unified.ts` | ~200 | **Zero imports anywhere.** Class-based `ApiClient` pattern, raw fetch, no auth. Exported singleton `apiClient` never used. | **Delete entirely.** |
| `lib/mock-data.ts` | ~150 | **Zero imports anywhere.** Contains hardcoded mock conversation data. | **Delete entirely.** |
| `lib/mock-conversation.ts` | ~100 | Only imported by `mock-data.ts` (also dead). | **Delete entirely.** |
| `components/workflow-manager.tsx` | ~200 | **Never imported by any page or component.** | **Delete entirely.** |

**Total: ~950 lines of dead code to remove.** Cleaner codebase, smaller bundle, less confusion.

### 5.2 Dead Functions in Active API Client

`lib/api-client.ts` is the **only active** API client. But many of its exports are never called:

| Function | Why Dead | What To Do |
|----------|----------|------------|
| `sendMessage()` | All chat uses WebSocket, not HTTP POST | **Remove.** If you ever need HTTP chat fallback, re-add later. |
| `continueChat()` | Same — WebSocket handles continuations | **Remove.** |
| `getConversation()` | No component calls it | **Remove.** |
| `deleteConversation()` | `conversations-dropdown` uses inline fetch instead | **Remove**, or better: refactor `conversations-dropdown` to use this function, then keep it. |
| `getAgentById()` | Not called anywhere | **Remove.** |
| `rateAgent()` / `rateAgentByName()` | `star-rating.tsx` uses its own inline fetch | **Pick one pattern:** Either refactor `star-rating.tsx` to use these functions (preferred — centralizes API calls), or delete the functions. Don't have both. |
| `getWebSocketUrl()` | Legacy helper, not called | **Remove.** |
| `getDashboardMetrics()` | Metrics page uses inline fetch | **Same as rating — either refactor page to use this, or delete it.** |
| `healthCheck()` | Not called | **Remove.** |

**Ideation:** The goal is **one API client file → all API calls go through it**. Currently, many components bypass `api-client.ts` and do their own `fetch()`. This creates a maintenance nightmare. The fix pattern:
1. Keep `api-client.ts` as the single source of truth
2. Every API function uses `authFetch()` internally
3. Components import from `api-client.ts`, never call `fetch()` directly
4. This makes URL changes a one-file edit, auth changes a one-file edit

### 5.3 Hardcoded URLs — 🔴 Cloud Deployment Blocker

This is the single biggest blocker for Azure deployment. ~80% of API calls use `'http://localhost:8000'` as a string literal.

| Pattern | Count | Example Files |
|---------|-------|---------------|
| `'http://localhost:8000'` hardcoded | 15+ files | api-client.ts, conversation-store.ts, workflow-execution-chat.tsx, save-workflow-button.tsx, conversations-dropdown.tsx, saved-workflows pages, credentials page |
| `process.env.NEXT_PUBLIC_API_URL` (correct!) | 5 files | connections/page.tsx, star-rating.tsx, schedule pages, metrics page |
| `'ws://localhost:8000'` hardcoded | 4 files | use-websocket-conversation.ts, workflow-orchestration.tsx, workflow-execution-chat.tsx, page_new.tsx |

#### What To Build Instead

**Create `lib/config.ts`** — single source of truth for all runtime configuration:

```
Exports:
  - API_BASE_URL  → reads from NEXT_PUBLIC_API_URL, falls back to localhost for dev
  - WS_BASE_URL   → derives from API_BASE_URL (http→ws, https→wss)
  - getApiUrl(path) → helper: joins API_BASE_URL + path
  - getWsUrl(path)  → helper: joins WS_BASE_URL + path
```

Then search-and-replace across the entire frontend: every `'http://localhost:8000'` becomes `API_BASE_URL`, every `'ws://localhost:8000'` becomes `WS_BASE_URL`.

**Also needed:** Create `frontend/.env.example` documenting all required env vars so developers (and Docker builds) know what to set.

### 5.4 Frontend Calls to Missing Backend Endpoints

These API calls go nowhere — the backend routes don't exist:

| Frontend Location | Endpoint Called | Why It's Missing | What To Build |
|-------------------|----------------|------------------|---------------|
| `connections/page.tsx` | `GET /api/integrations/status/{userId}` | No integrations router (see §2.3) | Build the integrations router |
| `connections/page.tsx` | `POST /api/integrations/auth/start/{userId}` | Same | Same |
| `connections/page.tsx` | `POST /api/integrations/sync/{userId}` | Same | Same |
| `connections/page.tsx` | `GET /api/integrations/available` | Same | Same |
| `connections/page.tsx` | `DELETE /api/integrations/{userId}/{appSlug}` | Same | Same |
| `connections/page.tsx` | `POST .../refresh`, `.../enable`, `.../disable` | Same | Same |
| Unknown component | `GET /api/conversations/search` | Endpoint not implemented in any router | Either implement conversation search or remove the frontend call |
| `saved-workflows/` | `POST /api/workflows/{id}/clone` | No clone endpoint in workflows_router | Implement clone (copy workflow + generate new ID) or remove UI button |
| `saved-workflows/` | `PUT /api/workflows/{id}` | No PUT endpoint (only POST/GET) | Add update endpoint to workflows_router or change frontend to PATCH |

### 5.5 Unauthenticated Frontend API Calls

These pages make `fetch()` without sending the Clerk JWT token. Even if the backend enforces auth, these calls will get 401 errors:

| Page/Component | What It Calls | Risk | What To Do |
|----------------|--------------|------|------------|
| `star-rating.tsx` | `POST /api/agents/{id}/rate` | Rating manipulation | **Use `authFetch()`** from `lib/auth-fetch.ts` instead of raw `fetch()` |
| `PlanGraph.tsx` | `POST /api/orchestrator/action/approve` | Action approval bypass | **Use `authFetch()`** — critical, approving actions must be gated |
| Schedule pages | `GET/PATCH/DELETE /api/workflows/schedules/*` | Schedule tampering | **Use `authFetch()`** |
| Metrics page | `GET /api/metrics/dashboard` | Data leak | **Use `authFetch()`** |
| Credentials page | `GET/POST /api/credentials/*` | Credential theft | **Use `authFetch()`** — backend requires JWT but frontend doesn't send it |

**Pattern:** `lib/auth-fetch.ts` already exists and correctly attaches Clerk JWT. Fix is mechanical: replace `fetch(url, options)` with `authFetch(url, options)` in each file. Same API signature.

### 5.6 WebSocket Issues

| Issue | Detail | What To Build Instead |
|-------|--------|----------------------|
| **Hardcoded `ws://localhost:8000`** | All 4 WebSocket connections use hardcoded URLs | **Derive from `API_BASE_URL`** in `lib/config.ts`: `http` → `ws`, `https` → `wss`. One line. |
| **Multiple WebSocket implementations** | `use-websocket-conversation.ts` (hook), `workflow-orchestration.tsx` (inline), `workflow-execution-chat.tsx` (inline), `page_new.tsx` (inline) — 4 separate WebSocket managers | **Consolidate into the hook.** `use-websocket-conversation.ts` is the best implementation. Other components should use it instead of creating their own. Fewer open connections = cheaper on Azure. |
| **No reconnection on disconnect** | If WebSocket drops (common behind Azure load balancers), no auto-reconnect | **Add to the hook:** reconnection with exponential backoff (1s, 2s, 4s, max 30s). Library `reconnecting-websocket` wraps this cleanly, or hand-roll in the hook with `setTimeout`. |

### 5.7 Heavy/Unused Dependencies

| Package | Issue | What To Do |
|---------|-------|------------|
| `kokoro-js` | Very large TTS model, ships client-side | **If TTS needed:** move to server-side endpoint. **If not:** remove from production bundle, add only when feature activated. |
| `react-icons` (~500KB) | Only 3 icons used, in 1 file. `lucide-react` already installed + used everywhere else. | **Replace 3 imports** with `lucide-react` equivalents. Remove `react-icons` from `package.json`. |
| `marker` | **Not imported anywhere** in the codebase. | **Remove from `package.json`.** |
| `geist` | Font package, not imported in TS/TSX. | **Check `globals.css` / `layout.tsx`** — if not referenced, remove. |
| `sonner` + `use-toast.ts` (dual toast) | Two notification systems. Inconsistent UX. | **Standardize on `sonner`** — simpler, no provider needed. Remove shadcn toast. |

### 5.8 Frontend Optimization Summary

| Action | Estimated Savings |
|--------|-------------------|
| Delete 4 dead lib files + 1 dead component | ~950 lines, ~40KB source |
| Remove dead functions from api-client.ts | ~200 lines, ~8KB source |
| Remove `react-icons` (use lucide-react) | ~500KB from bundle |
| Remove `marker` | Package size varies |
| Code-split `reactflow` (PlanGraph) and `recharts` (metrics) | ~650KB lazy-loaded instead of upfront |
| Next.js `output: 'standalone'` | Docker image: ~1GB → ~100MB |

---

## 6. Cloud Deployment Notes (Frontend + Integrations)

> Only covering what's relevant to integrations/agents/frontend. Backend core deployment is separate workstream.

### 6.1 What's Needed for Azure Deployment

| Need | Status | What To Build |
|------|--------|---------------|
| **Frontend env-var config** | ❌ Hardcoded localhost | `lib/config.ts` + `.env.example` + replace all hardcoded URLs |
| **Integrations router** | ❌ Missing | `routers/integrations_router.py` (see §2.3) |
| **Frontend Dockerfile** | ❌ Missing | Multi-stage: `npm ci` → `npm run build` → standalone serve |
| **Next.js standalone output** | ❌ Not configured | Add `output: 'standalone'` to `next.config.mjs` |
| **WebSocket in cloud** | ⚠️ Hardcoded ws:// | Derive from API URL, enable WebSocket on Azure App Service |
| **CORS for frontend domain** | ❌ `allow_origins=["*"]` | Read allowed origins from env var, set to frontend Azure URL |
| **Frontend `.env.example`** | ❌ Missing | Document all required vars (see below) |

### 6.2 Frontend `.env.example` (To Create)

```bash
# Required
NEXT_PUBLIC_API_URL=https://your-backend.azurewebsites.net
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_xxx
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up

# Optional
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/
```

### 6.3 Azure WebSocket Considerations

Azure App Service supports WebSocket but requires:
1. **Configuration → General Settings → Web Sockets → ON**
2. WSS (not WS) in production — URL derivation must handle `https` → `wss`
3. Azure Front Door / Application Gateway needs WebSocket affinity enabled
4. Connection timeout: Azure defaults to 230 seconds idle → add ping/pong heartbeat to the WebSocket hook (send ping every 60 seconds)

### 6.4 Azure Environment Variables for Integrations

```bash
# These are specific to the integrations/agents scope
COMPOSIO_API_KEY=your_composio_key
CONNECTION_ENCRYPTION_KEY=your_fernet_key
COMPOSIO_AUTH_CONFIG_GMAIL=your_gmail_auth_config
COMPOSIO_AUTH_CONFIG_ZOHOBOOKS=your_zoho_auth_config
COMPOSIO_AUTH_CONFIG_GITHUB=your_github_auth_config
```

---

## 7. Action Plan

### Phase 1: Unblock Deployment — ~3-4 days

| # | Task | Scope | Impact |
|---|------|-------|--------|
| 1 | **Fix hardcoded URLs in frontend** | Create `lib/config.ts`, replace in ~20 files | Unblocks cloud deployment |
| 2 | **Create integrations router** | New `routers/integrations_router.py`, add to `main.py` | Unblocks connections page |
| 3 | **Fix logger-before-definition** | 2-line fix in `composio_auth.py` + `composio_tools.py` | Prevents import crash |
| 4 | **Fix dead Zoho imports** | Rewrite 2 functions in `zoho_books_helpers.py` | Prevents runtime crash |
| 5 | **Remove dead `zoho_books` spawn reference** | Delete 1 line in `main.py` | Prevents startup error |
| 6 | **Delete 5 dead frontend files** | Remove `api-client-new.ts`, `api-unified.ts`, `mock-data.ts`, `mock-conversation.ts`, `workflow-manager.tsx` | Cleaner codebase |
| 7 | **Create frontend `.env.example`** | New file | Dev onboarding + Docker builds |

### Phase 2: Gmail Agent Integration — ~3-4 days

| # | Task | Scope | Impact |
|---|------|-------|--------|
| 8 | **Create SKILL.md for gmail_agent** | New file in `agent_entries/templates/` | Brain discovers Gmail tasks |
| 9 | **Wrap gmail_agent responses in AgentResponse** | Edit `agent.py` — wrap all returns | Orchestrator parses results |
| 10 | **Fix `entity_id` usage** | Edit `tools.py` — `connection_id` → `user_id` | Multi-user auth works |
| 11 | **Replace standalone LLM client** | Delete `llm.py`, import `InferenceService` | Unified LLM management |
| 12 | **Add TTL to service cache** | Edit `agent.py` — use `cachetools.TTLCache` | Prevent memory leak |

### Phase 3: Zoho + Polish — ~2-3 days

| # | Task | Scope | Impact |
|---|------|-------|--------|
| 13 | **Create SKILL.md for Zoho Books** | New file in `agent_entries/templates/` | Brain plans accounting tasks |
| 14 | **Register Zoho tools in orchestrator** | SKILL.md approach or tool registry | Tools become usable |
| 15 | **Fix composio_auth duplicate methods** | Delete 3 duplicate method defs | Clean API surface |
| 16 | **Standardize frontend fetch → authFetch** | Edit ~10 components | Security + consistent auth |
| 17 | **Remove unused npm packages** | Edit `package.json` | Smaller builds |
| 18 | **Consolidate WebSocket implementations** | Refactor 3 inline WSs → shared hook | Fewer connections |

### Phase 4: Frontend Optimization — ~2 days

| # | Task | Scope | Impact |
|---|------|-------|--------|
| 19 | **Remove dead api-client.ts functions** | Edit `lib/api-client.ts` | Smaller bundle |
| 20 | **Replace react-icons with lucide-react** | Edit 1 file + remove package | ~500KB savings |
| 21 | **Add Next.js standalone output** | Edit `next.config.mjs` | 10x smaller Docker image |
| 22 | **Add WebSocket reconnection** | Edit `use-websocket-conversation.ts` | Reliable cloud connections |
| 23 | **Code-split PlanGraph + metrics** | Dynamic imports for reactflow + recharts | ~650KB deferred |
| 24 | **Standardize toast system** | Pick sonner, remove shadcn toast | Consistent UX |

---

*Scoped audit complete. Backend/orchestrator core excluded — separate workstream.*
