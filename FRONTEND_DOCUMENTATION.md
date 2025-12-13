# Frontend Architecture Documentation - Current Implementation

**Generated:** December 8, 2025  
**Project:** Orbimesh App - Ambitious  
**Framework:** Next.js 15.2.4 with React 19  

---

## 📁 1. PAGES & ROUTES

### Root Routes
| Path | File | Purpose | Key Features |
|------|------|---------|---------------|
| `/` | `app/page.tsx` | Home/Main Chat Interface | Interactive chat, conversation state management, WebSocket integration, file uploads, plan approval |
| `/c/[thread_id]` | `app/c/[thread_id]/page.tsx` | Shareable Conversation URLs | Redirects to home with threadId query param |
| `/sign-in/*` | `app/sign-in/[[...sign-in]]/` | Authentication | Clerk sign-in pages |
| `/sign-up/*` | `app/sign-up/[[...sign-up]]/` | Registration | Clerk sign-up pages |

### Dashboard Routes (Protected)
| Path | File | Purpose | Key Features |
|------|------|---------|---------------|
| `/agents` | `app/(dashboard)/agents/page.tsx` | Agent Marketplace | Browse, search, filter agents by category |
| `/metrics` | `app/(dashboard)/metrics/page.tsx` | Analytics Dashboard | Conversation stats, cost metrics, agent usage, performance metrics |
| `/profile` | `app/(dashboard)/profile/page.tsx` | User Profile | Clerk UserProfile component, preferences |
| `/credentials` | `app/(dashboard)/credentials/page.tsx` | API Keys & Credentials | User credential management |

### Workflow Routes
| Path | File | Purpose | Key Features |
|------|------|---------|---------------|
| `/workflows` | `app/workflows/page.tsx` | Workflow Management | List saved workflows, execute, schedule, create webhooks |
| `/saved-workflows` | `app/saved-workflows/page.tsx` | Saved Workflows Library | View, execute, delete, clone workflows |
| `/saved-workflows/[workflow_id]` | `app/saved-workflows/[workflow_id]/` | Individual Workflow | Workflow details and execution |
| `/schedules` | `app/schedules/page.tsx` | Scheduled Workflows | View, pause/resume, delete schedules |
| `/schedules/[schedule_id]/executions` | `app/schedules/[schedule_id]/executions/` | Schedule Execution History | View past executions |

### Other Routes
| Path | File | Purpose | Key Features |
|------|------|---------|---------------|
| `/register-agent` | `app/register-agent/page.tsx` | Agent Registration | Form to register new agents with endpoints |
| `/connections` | `app/connections/page.tsx` | External Connections | MCP server connections, integrations management |

---

## 🔌 2. API CLIENT FILES

### Primary API Clients

#### `lib/api-client.ts` (ACTIVE - WITH AUTH)
**Purpose:** Main API client using Clerk authentication  
**Base URL:** `http://localhost:8000`  
**Auth Method:** `authFetch()` wrapper with Clerk JWT

**Functions:**
- **Agent Management:**
  - `fetchAllAgents()` - Get all agents
  - `fetchFilteredAgents(options)` - Filter by price, rating, status
  - `searchAgents(options)` - Search by capabilities with similarity
  - `rateAgent(agentId, rating)` - Rate agent by ID
  - `rateAgentByName(agentName, rating)` - Rate agent by name
  - `fetchAgentById(agentId)` - Get single agent
  - `registerAgent(agentData)` - Register new agent

- **Conversation Management:**
  - `startConversation(prompt, thread_id?, uploadedFiles?)` - POST `/api/chat`
  - `continueConversation(response, threadId, uploadedFiles?)` - POST `/api/chat/continue`
  - `getConversationStatus(threadId)` - GET `/api/chat/status/:threadId`
  - `clearConversation(threadId)` - DELETE `/api/chat/:threadId`

- **File Management:**
  - `uploadFiles(files)` - POST `/api/upload` with FormData

- **Content Management:**
  - Uses unified content API (`/api/content/*`)
  - Supports file uploads, artifact storage, and retrieval
  - Integrated with content orchestrator for agent file distribution

- **Plan Management:**
  - `fetchPlanFile(threadId)` - GET `/api/plan/:threadId`

- **Utilities:**
  - `processPrompt(request)` - Legacy wrapper for startConversation
  - `healthCheck()` - GET `/api/health`

**Exports:**
- `frameworks` - Array of supported frameworks
- `capabilities` - Array of agent capabilities

---

#### `lib/api-client-new.ts` (REDUNDANT - NO AUTH)
**Status:** ⚠️ **DUPLICATE** - Same functionality as api-client.ts but WITHOUT authentication  
**Base URL:** `http://localhost:8000`  
**Auth Method:** None (plain fetch)

**Identical Functions to api-client.ts:**
- All agent management functions
- All conversation functions
- `healthCheck()`

**Additional Exports:**
- `fallbackAgents` - Hardcoded fallback agent data

**Redundancy Note:** This appears to be an older version before auth was implemented. Should be removed or consolidated.

---

#### `lib/api-unified.ts` (INCOMPLETE - CLASS-BASED)
**Status:** ⚠️ **INCOMPLETE** - Partial class-based implementation  
**Base URL:** `process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'`  
**Pattern:** Class with instance methods

**Class: `UnifiedAPIClient`**
- `getAllAgents()` - GET `/api/agents/all`
- `getFilteredAgents(options)` - GET `/api/agents/all` with params
- `searchAgents(params)` - GET `/agents/search`
- `getAgent(agentId)` - GET `/agents/:id`
- `registerAgent(agentData)` - POST `/agents/register`
- `rateAgent(agentId, rating)` - POST `/agents/:id/rate`
- `processPrompt(request)` - POST `/api/chat`
- `healthCheck()` - GET `/api/health`

**Export:** Single instance `api`

**Redundancy Note:** Partially implemented alternative to api-client.ts. Not used in current codebase.

---

#### `lib/auth-fetch.ts` (AUTH UTILITY - ACTIVE)
**Purpose:** Clerk JWT authentication wrapper  
**Used By:** api-client.ts

**Functions:**
- `getClerkToken()` - Retrieves JWT from Clerk session
  - Tries template-based token first (from env: `NEXT_PUBLIC_CLERK_JWT_TEMPLATE`)
  - Falls back to default token
  - Client-side only (checks `window.Clerk`)

- `authFetch(url, options)` - Wrapper around fetch with Authorization header
  - Automatically adds `Bearer ${token}` to headers
  - Logs all requests for debugging

- `getOwnerFromClerk()` - Extract user ID and email from Clerk
  - Returns: `{ user_id: string, email?: string }`

---

### API Redundancy Summary

| Feature | api-client.ts | api-client-new.ts | api-unified.ts |
|---------|---------------|-------------------|----------------|
| Agent CRUD | ✅ With Auth | ✅ No Auth | ⚠️ Partial |
| Conversations | ✅ With Auth | ✅ No Auth | ⚠️ Partial |
| File Upload | ✅ | ❌ | ❌ |
| Plan Fetch | ✅ | ❌ | ❌ |
| Pattern | Functions | Functions | Class |
| Status | **ACTIVE** | **REDUNDANT** | **INCOMPLETE** |

**Recommendation:** Remove `api-client-new.ts` and `api-unified.ts`. Use only `api-client.ts` with `auth-fetch.ts`.

---

## 🗄️ 3. STATE MANAGEMENT

### `lib/conversation-store.ts` (Zustand Store)
**Purpose:** Global conversation state management  
**Library:** Zustand 5.0.8

**State Interface: `ConversationStore`**
```typescript
{
  // Core State
  thread_id?: string
  status: 'idle' | 'processing' | 'completed' | 'waiting_for_user' | 'error'
  messages: Message[]
  isLoading: boolean
  isWaitingForUser: boolean
  currentQuestion?: string
  
  // Agent & Task Data
  task_agent_pairs: TaskAgentPair[]
  final_response?: string
  metadata: {
    currentStage: string
    stageMessage: string
    progress: number
    original_prompt?: string
    completed_tasks?: any[]
    parsed_tasks?: any[]
  }
  
  // File Management
  uploaded_files: FileObject[]
  
  // Plan Data
  plan: any[]
  approval_required: boolean
  estimated_cost: number
  task_count: number
  task_plan?: any[]
  
  // Real-time Task Tracking
  task_statuses: Record<string, TaskStatus>
  current_executing_task: string | null
  
  // Canvas Feature
  canvas_content?: string
  canvas_type?: 'html' | 'markdown'
  has_canvas: boolean
  browser_view?: string
  plan_view?: string
  current_view: 'browser' | 'plan'
}
```

**Actions:**
- `startConversation(input, files?, planningMode?, owner?)` - Start new conversation via WebSocket
- `continueConversation(input, files?, planningMode?, owner?)` - Continue existing conversation
- `loadConversation(threadId)` - Load conversation from backend
- `resetConversation()` - Clear all state
- `_setConversationState(newState)` - Internal state updater (used by WebSocket)

**Key Features:**
- Deterministic message ID generation (matches backend)
- File upload integration
- WebSocket message queuing with retry logic (up to 50 attempts)
- Approval response handling (doesn't show "approve"/"cancel" as messages)
- Message deduplication (content + type based)
- LocalStorage persistence for thread_id

---

## 🎣 4. HOOKS

### `hooks/use-websocket-conversation.ts`
**Purpose:** WebSocket connection management and message handling  
**Pattern:** React hook returning connection status

**Hook: `useWebSocketManager(props?)`**
- **URL:** `ws://localhost:8000/ws/chat` (default)
- **Returns:** `{ isConnected: boolean }`

**Features:**
- Auto-connect on mount
- Auto-reconnect on disconnect (2 second delay)
- Exposes WebSocket to window: `window.__websocket`
- Processes WebSocket events and updates Zustand store
- Handles orchestration stage tracking with progress
- Real-time task status updates (`task_started`, `task_completed`, `task_failed`)
- Error handling with user-friendly messages
- Plan approval detection
- Canvas live updates

**WebSocket Event Handlers:**
| Event Node | Action | Progress |
|------------|--------|----------|
| `__start__` | Set status to processing | 0% |
| `analyze_request` | Analyzing stage | 10% |
| `parse_prompt` | Breaking down tasks | 20% |
| `agent_directory_search` | Searching agents | 35% |
| `rank_agents` | Ranking agents | 50% |
| `plan_execution` | Creating plan | 60% |
| `validate_plan_for_execution` | Validating plan | 70% |
| `execute_batch` | Executing tasks | 80% |
| `aggregate_responses` | Generating response | 95% |
| `__end__` | Complete, update all state | 100% |
| `__error__` | Error handling | - |
| `__user_input_required__` | Wait for user | - |
| `__live_canvas__` | Canvas update | - |
| `task_started` | Update task status | - |
| `task_completed` | Update task status | - |
| `task_failed` | Update task status | - |

**Message Processing:**
- Filters empty assistant messages
- Detects HTML content for canvas
- Prevents HTML from appearing in chat
- Merges backend messages with frontend
- Handles plan approval without adding system message

---

### `hooks/use-toast.ts`
**Purpose:** Toast notification management  
**Pattern:** shadcn/ui toast hook

---

### `hooks/use-mobile.ts`
**Purpose:** Responsive design detection  
**Pattern:** Media query hook

---

## 🧩 5. COMPONENTS

### Major Feature Components

#### Chat & Conversation
- **`interactive-chat-interface.tsx`** - Main chat UI with:
  - Message display with Markdown rendering
  - File attachments (images, documents)
  - Plan approval modal
  - **Accept & Modify buttons** (inline with Send Response)
  - **Workflow modification support** - detects saved workflow state and sends modifications
  - Context-aware button text ("Modify" for saved workflows, "Send Response" otherwise)
  - Real-time task tracking display
  - Canvas view integration
- **`websocket-workflow.tsx`** - Alternative WebSocket-based workflow interface
- **`workflow-execution-chat.tsx`** - Workflow-specific chat interface

#### Workflow Management
- **`workflow-orchestration.tsx`** - Orchestration visualization with phases, data sources, task analysis
- **`workflow-manager.tsx`** - Workflow CRUD operations
- **`task-builder.tsx`** - Task construction interface
- **`save-workflow-button.tsx`** - Save conversation as workflow
- **`schedule-workflow-dialog.tsx`** - Schedule creation dialog

#### Planning & Approval
- **`plan-approval-modal.tsx`** - Plan review and approval UI with cost estimate
- **`plan-review-modal.tsx`** - Detailed plan review
- **`PlanGraph.tsx`** - Visual graph of plan structure using ReactFlow
- **`orchestration-progress.tsx`** - Progress visualization with stage animations

#### Agent Management
- **`agent-card.tsx`** - Individual agent display card
- **`agent-grid.tsx`** - Grid layout for agents
- **`agent-preview.tsx`** - Preview during registration
- **`agent-registration-form.tsx`** - Agent registration form with endpoint management

#### Navigation & Layout
- **`app-sidebar.tsx`** - Main sidebar with conversation history
- **`navbar.tsx`** - Top navigation bar
- **`page-layout-wrapper.tsx`** - Page layout wrapper
- **`conversations-dropdown.tsx`** - Conversation selector dropdown
- **`orchestration-details-sidebar.tsx`** - Right sidebar with orchestration details

#### Utilities
- **`theme-provider.tsx`** - Next-themes wrapper
- **`theme-toggle.tsx`** - Theme switcher
- **`dark-mode-toggle.tsx`** - Dark mode toggle
- **`CollapsibleSection.tsx`** - Collapsible UI section

### UI Components (shadcn/ui)
Located in `components/ui/`:
- accordion, alert-dialog, avatar, badge, button, calendar, card, carousel
- checkbox, collapsible, command, context-menu, dialog, drawer, dropdown-menu
- form, hover-card, input, label, menubar, navigation-menu, popover
- progress, radio-group, resizable, scroll-area, select, separator, sidebar
- skeleton, slider, switch, table, tabs, textarea, toast, toggle, tooltip
- **markdown.tsx** - Custom Markdown renderer with remark-gfm, rehype-raw

---

## 🔗 6. EXTERNAL INTEGRATIONS

### Authentication
- **Library:** `@clerk/nextjs` v6.34.0
- **Setup:** ClerkProvider wraps entire app in `app/layout.tsx`
- **Middleware:** `middleware.ts` protects all routes except `/sign-in`, `/sign-up`, `/api/webhooks`
- **JWT Template:** Configurable via `NEXT_PUBLIC_CLERK_JWT_TEMPLATE`
- **Usage:**
  - `useUser()` - Get current user
  - `useAuth()` - Get auth methods (getToken)
  - `authFetch()` - Custom wrapper for authenticated requests

### UI Framework
- **Library:** Radix UI primitives (v1.x)
- **Components:** 30+ accessible, unstyled components
- **Styling:** Tailwind CSS 4.1.9 with tailwindcss-animate

### Data Visualization
- **Library:** Recharts 2.15.4
- **Used In:** Metrics dashboard for charts (bar, line, pie)

### Markdown Rendering
- **Library:** react-markdown 10.1.0
- **Plugins:**
  - `remark-gfm` - GitHub Flavored Markdown
  - `remark-breaks` - Line break support
  - `rehype-raw` - Raw HTML support

### Flow Diagrams
- **Library:** ReactFlow 11.11.4
- **Used In:** PlanGraph component for visual plan representation

### Others
- **Framer Motion** 12.23.24 - Animations
- **Zustand** 5.0.8 - State management
- **React Hook Form** 7.60.0 - Form handling
- **Zod** 3.25.67 - Schema validation
- **Sonner** 1.7.4 - Toast notifications

---

## 🌐 7. WEBSOCKET CONNECTION MANAGEMENT

### Connection Flow
1. **Initialization:** `useWebSocketManager` hook connects on mount
2. **URL:** `ws://localhost:8000/ws/chat`
3. **State:** Exposed to window as `window.__websocket`
4. **Auto-reconnect:** 2 second delay on abnormal disconnect
5. **Cleanup:** Connection persists across component remounts

### Message Flow
```
User Input (Chat Interface)
  ↓
conversation-store.ts (startConversation/continueConversation)
  ↓
WebSocket Send (with retry logic up to 50 attempts)
  ↓
Backend Processing
  ↓
WebSocket Events (__start__, parse_prompt, execute_batch, __end__, etc.)
  ↓
use-websocket-conversation.ts (onmessage handler)
  ↓
_setConversationState (Zustand store update)
  ↓
UI Updates (React re-render)
```

### Connection States
- **CONNECTING:** WebSocket initializing
- **OPEN:** Ready to send/receive
- **CLOSING:** Disconnect initiated
- **CLOSED:** Disconnected, attempting reconnect

### Error Handling
- **Parse Errors:** Logged, user notified via system message
- **Connection Errors:** Auto-reconnect, user notified
- **Timeout:** Handled by backend, frontend waits for `__end__` or `__error__`

---

## 📤 8. FILE UPLOAD/DOWNLOAD FLOWS

### Upload Flow
```
User selects files
  ↓
InteractiveChatInterface.tsx (attachedFiles state)
  ↓
startConversation/continueConversation
  ↓
uploadFiles(files) - POST /api/upload with FormData
  ↓
Backend stores files, returns FileObject[]
  ↓
WebSocket message includes file metadata
  ↓
Files stored in conversation-store.uploaded_files
```

### File Types
**Supported:**
- Images (image/*) - Shown as base64 data URL previews
- Documents (application/*) - File icon shown

### FileObject Structure
```typescript
{
  file_name: string
  file_path: string  // Backend storage path
  file_type: string  // MIME type
}
```

### Attachment Display
- **Images:** Rendered inline with data URL in chat
- **Documents:** File icon with name
- **Removal:** X button to remove before sending

### Download Flow
(Not explicitly implemented in frontend - would require backend endpoint)

---

## 🔄 9. DATA FLOW PATTERNS

### Conversation Lifecycle
```
1. User lands on home page
   ↓
2. Check localStorage for thread_id
   ↓
3. If exists: loadConversation(thread_id)
   ↓
4. User sends message
   ↓
5. startConversation (if new) or continueConversation
   ↓
6. WebSocket sends message to backend
   ↓
7. Backend processes through orchestrator
   ↓
8. WebSocket events stream back
   ↓
9. Store updates via _setConversationState
   ↓
10. UI reflects changes
   ↓
11. On complete: thread_id saved to localStorage
```

### Plan Approval Flow
```
1. Backend sends __user_input_required__ with approval_required: true
   ↓
2. WebSocket handler detects approval request
   ↓
3. Sets approval_required: true in store (no message added)
   ↓
4. PlanApprovalModal opens
   ↓
5. User clicks Approve/Cancel
   ↓
6. Sets approval_required: false (modal closes)
   ↓
7. continueConversation('approve' or 'cancel')
   ↓
8. Backend continues or cancels execution
```

### Workflow Modification Flow
```
1. User loads saved workflow (status: planning_complete)
   ↓
2. Send Response button displays as "Modify"
   ↓
3. User types modifications in text box
   ↓
4. User clicks "Modify" button
   ↓
5. handleSubmit detects currentStage === 'validating' OR status === 'planning_complete'
   ↓
6. Calls continueConversation(userInput, files, planningMode, owner)
   ↓
7. Backend combines original_prompt with modifications
   ↓
8. Re-executes orchestration with updated prompt
   ↓
9. New plan generated and displayed
```

### Real-time Task Tracking
```
1. Backend sends task_started event
   ↓
2. WebSocket handler updates task_statuses[taskName] = { status: 'running' }
   ↓
3. UI shows running indicator
   ↓
4. Backend sends task_completed or task_failed
   ↓
5. WebSocket updates task_statuses[taskName] with result
   ↓
6. UI shows completed/failed state with metrics
```

---

## ⚠️ 10. REDUNDANCIES & DUPLICATES FOUND

### Critical Redundancies

#### 1. API Client Files (HIGH PRIORITY)
**Issue:** Three API client implementations with overlapping functionality

| File | Status | Recommendation |
|------|--------|----------------|
| `api-client.ts` | ✅ Active, with auth | **KEEP** |
| `api-client-new.ts` | ⚠️ Duplicate, no auth | **DELETE** |
| `api-unified.ts` | ⚠️ Incomplete class | **DELETE** |

**Action:** Remove `api-client-new.ts` and `api-unified.ts`, use only `api-client.ts`

---

#### 2. Mock Data Files
**File:** `lib/mock-data.ts`  
**Status:** Contains hardcoded agents and mock functions  
**Issue:** May conflict with real API data  
**Recommendation:** Remove if not used for testing/demo

---

#### 3. Conversation Management
**Pattern:** Two conversation interfaces overlap
- `interactive-chat-interface.tsx` - Full-featured, actively used
- `websocket-workflow.tsx` - Simplified version, less featured

**Recommendation:** Consolidate or clarify use cases

---

#### 4. Workflow Pages
**Duplicate Routes:**
- `/workflows` - Full workflow management
- `/saved-workflows` - Similar functionality

**Recommendation:** Merge into single unified workflow management page

---

#### 5. Page Files
**Found:** `saved-workflows/[workflow_id]/page_new.tsx` alongside `page.tsx`  
**Recommendation:** Remove `page_new.tsx` if not in use

---

### Environment Variables
**Required:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
NEXT_PUBLIC_CLERK_JWT_TEMPLATE=your-backend-template
```

---

## 📊 11. KEY METRICS

### File Counts
- **TypeScript files:** 15+ in lib/
- **TSX components:** 96+ files
- **Pages:** 15+ routes
- **Hooks:** 3 custom hooks
- **API functions:** 30+ across all clients

### Dependencies
- **Total:** 50+ production dependencies
- **Dev Dependencies:** 6
- **Framework:** Next.js 15.2.4
- **React:** Version 19
- **UI Components:** 30+ Radix UI primitives

### Code Patterns
- **State Management:** Zustand (global), useState (local)
- **Data Fetching:** Server-side API calls with auth
- **Styling:** Tailwind CSS with CSS variables
- **TypeScript:** Strict typing with interfaces
- **Authentication:** Clerk with middleware protection

---

## 🎯 12. RECOMMENDATIONS

### Immediate Actions
1. ✅ **Remove duplicate API clients** (`api-client-new.ts`, `api-unified.ts`)
2. ✅ **Consolidate workflow pages** (merge `/workflows` and `/saved-workflows`)
3. ✅ **Remove unused mock data** if not needed
4. ✅ **Clean up `page_new.tsx` files**

### Code Quality
1. Add JSDoc comments to all API functions
2. Create consistent error handling pattern
3. Implement proper TypeScript strict mode
4. Add unit tests for utilities and hooks

### Performance
1. Implement React.memo for expensive components
2. Add code splitting for large pages
3. Optimize WebSocket reconnection logic
4. Add request caching layer

### Documentation
1. Add API endpoint documentation
2. Create component usage examples
3. Document WebSocket message schema
4. Add architecture decision records (ADRs)

---

## 📝 SUMMARY

**Frontend Architecture:** Next.js 15 with App Router, React 19, TypeScript, Tailwind CSS

**Key Strengths:**
- ✅ Comprehensive WebSocket integration with real-time updates
- ✅ Robust state management with Zustand
- ✅ Strong authentication with Clerk
- ✅ Rich UI component library (Radix UI + shadcn/ui)
- ✅ File upload/attachment support
- ✅ Plan approval workflow
- ✅ Real-time task tracking

**Areas for Improvement:**
- ⚠️ Remove duplicate API client files
- ⚠️ Consolidate redundant pages
- ⚠️ Improve error handling consistency
- ⚠️ Add comprehensive testing
- ⚠️ Document environment variables

**Total Lines of Code (estimate):** 15,000+ lines across all TypeScript/TSX files

---

**End of Documentation**
