# Gmail Agent Architecture & Operations

## Overview

The Gmail Agent is a specialized agent within the Orbimesh system that provides comprehensive Gmail operations through a modular, multi-layered architecture. It combines Composio's Gmail tools with an internal LLM for intelligent query optimization, summarization, and email analysis.

**Key Capabilities:**
- Email search with natural language processing
- Email sending and replying
- Draft management
- Attachment handling (up to 25MB)
- Email summarization
- Action item extraction
- Label management

---

## Architecture Layers

### Layer 1: Service Layer (`service.py`)

**GmailService** is the core business logic handler that orchestrates all Gmail operations.

```
User Request
    ↓
GmailService.__init__(user_id)
    ├─→ ComposioToolManager (Composio SDK wrapper)
    ├─→ LLMClient (Reasoning & optimization)
    └─→ Memory (Session cache)
```

**Key Responsibilities:**
- **Search Emails**: Natural language → Gmail syntax optimization
- **Get Email**: Fetch single email by ID with full payload
- **Send/Reply**: Compose and dispatch emails
- **Draft Management**: Create, retrieve, update drafts
- **Attachments**: Download/upload files up to 25MB
- **Summarization**: Recursive map-reduce for long emails
- **Action Extraction**: Parse emails for actionable items

### Layer 2: Tool Manager (`tools.py`)

**ComposioToolManager** wraps the Composio SDK and handles:

```
ComposioToolManager
    ├─→ Authentication: Gets connection from DB
    ├─→ Tool Execution: Calls Composio Gmail API
    └─→ Error Handling: Retries transient errors (Rate limits, timeouts, 5xx)
```

**Key Methods:**
- `execute_tool(tool_slug, parameters)`: Generic tool wrapper with retry logic
- `fetch_emails()`: Search emails with pagination
- `fetch_message_by_id()`: Get single email details
- `send_email()`: Dispatch new email
- `reply_to_email()`: Reply in thread
- `create_draft()`: Save draft without sending
- `get_attachments()`: List all attachments
- `download_attachment()`: Save attachment locally

**Authentication Flow:**
```
ComposioToolManager.__init__(user_id)
    ↓
get_auth_manager().get_connection_for_agent(user_id, "gmail")
    ├─→ Query DB: UserConnection table
    ├─→ WHERE user_id = ? AND app_slug = 'gmail'
    ├─→ Decrypt connection_id
    ├─→ [Optional] Verify if > 1 hour old
    └─→ Return connection for tool use
```

### Layer 3: LLM Layer (`llm.py`)

**LLMClient** provides intelligent reasoning for email operations.

```
LLMClient
    ├─→ Provider Fallback: Cerebras → NVIDIA → Groq
    ├─→ Methods:
    │   ├─→ generate_optimized_query()
    │   ├─→ summarize_email_content()
    │   ├─→ draft_email_reply()
    │   └─→ extract_actions()
    └─→ strip_think_tags(): Remove internal reasoning from output
```

**System Prompts (Domain Expertise):**

1. **Query Generation** (Temperature: 0.5)
   - Converts "emails from John about project" → `from:John project`
   - Understands Gmail operators: `from:`, `to:`, `subject:`, `has:attachment`, etc.
   - Handles complex multi-operator queries

2. **Summarization** (Temperature: 0.5)
   - Recursive map-reduce for emails > 4KB
   - Focuses on facts, actions, context, urgency
   - Preserves implicit meanings and relationships

3. **Reply Drafting** (Temperature: 0.7)
   - Analyzes thread context
   - Matches sender tone and formality
   - Generates professional HTML emails

4. **Action Extraction** (Temperature: 0.4)
   - Identifies deadlines, requests, meetings
   - Infers priority levels from context
   - Distinguishes FYI from actionable items

### Layer 4: Base Agent (`base_agent_impl.py`)

**GmailAgent** extends BaseAgent framework:

```python
class GmailAgent(BaseAgent):
    - Registers capabilities (search, send, reply, etc.)
    - Manages per-user GmailService instances
    - Handles request routing and context extraction
    - Implements lifecycle (initialize, cleanup)
```

**Capability Methods:**
```python
@capability(name="search_emails", ...)
async def search_emails(
    params: Dict[str, Any],
    context: ExecutionContext
) -> AgentResponse:
    # Extracts user_id from context
    # Gets service for user
    # Calls service method
    # Returns AgentResponse with summary
```

---

## Data Flow Examples

### Example 1: Email Search with Optimization

```
1. User: "Show me recent emails from John about the Q1 project"
   ↓
2. BaseAgent receives request
   ├─→ Extracts user_id from ExecutionContext
   ├─→ Creates/gets GmailService for user
   └─→ Calls search_emails(query="...", context=ctx)
   ↓
3. GmailService.search_emails()
   ├─→ LLM optimizes query (0.5 temperature)
   │   "from:John Q1 project"
   ├─→ ComposioToolManager executes
   │   GMAIL_FETCH_EMAILS with optimized_query
   ├─→ Saves results to memory
   └─→ Returns { messages, total_count, query_used }
   ↓
4. Response to orchestrator with email list
```

### Example 2: Draft Email Reply

```
1. User: "Reply to John's latest email about meeting schedule"
   ↓
2. Service finds the email thread
   ↓
3. LLM summarizes thread (if > 6KB)
   ├─→ Reduces complexity for context understanding
   └─→ Maintains key details
   ↓
4. LLM drafts reply
   ├─→ System prompt: tone matching, business etiquette
   ├─→ Returns HTML-formatted email
   └─→ Returns as draft (not sent yet)
   ↓
5. User reviews and approves
   ↓
6. GmailService.send_email() dispatches
```

---

## Memory System (`memory.py`)

**AgentMemory** tracks per-user session state:

```python
agent_memory.save_search_results(user_id, message_ids)
# Later: retrieve for action extraction on original emails

agent_memory.save_draft_intent(user_id, draft_data)
# Track drafts across conversation turns
```

**Use Cases:**
- Remember last search results for follow-ups
- Track draft states (created, reviewed, sent)
- Store conversation context for multi-turn interactions

---

## Configuration (`config.py`)

```python
# Composio API
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")

# Storage
ATTACHMENT_DIR = "storage/gmail_agent/attachments/"
PROFILE_DIR = "storage/gmail_agent/profiles/"

# Limits
MAX_CONCURRENT_FETCHES = 5
MAX_SEARCH_RESULTS = 50
MAX_ATTACHMENT_SIZE = 25 * 1024 * 1024  # 25MB

# LLM Providers
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

---

## Error Handling Strategy

### Transient Errors (Retryable)
```python
Rate limits (429), Timeouts, Server errors (502, 503, 504)
    → Retry with exponential backoff (max 3 attempts)
```

### Permanent Errors (Non-Retryable)
```python
Invalid query (400), Not found (404), Auth failure (401)
    → Return error to user immediately
```

### Connection Errors
```python
User not connected to Gmail → Redirect to integrations_agent
    → Show OAuth link on frontend
    → Refresh credentials after OAuth callback
```

---

## Security & Privacy

### Credential Encryption
```
Connection ID stored in DB
    ↓ (Fernet cipher)
    ↓
Encrypted in database
    ↓ (Retrieved and decrypted on demand)
    ↓
Used by Composio SDK
    ↓
Never logged or exposed
```

### Sensitive Data Filtering
```python
# Logs never contain:
- Email bodies
- Attachment contents
- OAuth tokens
- User passwords
- Sensitive PII
```

### Scope Limitations
```
Gmail Agent can only:
✓ Read/search emails
✓ Create drafts
✓ Send emails (user-initiated)
✓ Manage labels
✗ Permanently delete emails
✗ Access other users' accounts
✗ Export all data
```

---

## Performance Optimizations

### LLM Query Optimization
```
"Show me recent emails"
    ↓ (LLM with 0.5 temperature)
    ↓
"label:inbox newer_than:7d"
    → Faster search, fewer false positives
```

### Recursive Map-Reduce for Large Emails
```
Email: 15,000 characters
    ↓ (Split into 3 chunks of 5,000 chars)
    ↓ (Summarize each chunk in parallel)
    ↓ (Recursively summarize summaries)
    ↓
Final summary: 500 characters
→ Fits in context window, maintains key info
```

### Connection Caching
```
First request: user_id="john_123", app="gmail"
    → Query DB, decrypt, verify
Second request: same user
    → Use cached connection (if < 1 hour)
    → No DB query
```

### Concurrent Tool Execution
```
Multiple fetch_message_by_id() calls
    ↓ (Async gather with MAX_CONCURRENT_FETCHES=5)
    ↓
Parallel execution: 5 emails at a time
    → Faster batch operations
    → Respects rate limits
```

---

## Testing Strategy

### Unit Tests (`tests/agents/test_gmail_agent.py`)
```
Layer 1: GmailService methods
    - search_emails() variations
    - get_email() error cases
    - send_email() validation
    
Layer 2: ComposioToolManager
    - Tool execution with mocks
    - Error handling + retries
    - Connection handling
    
Layer 3: LLMClient
    - Query generation accuracy
    - Summarization quality
    - Action extraction correctness
```

### Integration Tests (`tests/integration/gmail_agent/`)
```
Live Composio API tests (marked @pytest.mark.skip)
    - Actual Gmail searches
    - Real email sending (to test addresses only)
    - Attachment upload/download
    - Connection lifecycle
```

### Connection Tests
```
Connection verification flow
Draft lifecycle (create → review → send)
Error recovery (rate limits, auth failures)
```

---

## Troubleshooting Guide

### "User not connected to Gmail"
```
Issue: ValueError raised in ComposioToolManager.__init__
Fix:
  1. User must visit /connections page
  2. Click "Connect Gmail"
  3. OAuth redirect completes
  4. Check UserConnection table: status should be "active"
```

### "Query too vague or malformed"
```
Issue: LLM generated invalid Gmail syntax
Fix:
  Check llm.py:generate_optimized_query()
  - If query > 800 chars: reject (too complex)
  - If query contains malformed operators: retry with different provider
  - If all providers fail: use original query
```

### "Email too large to summarize"
```
Issue: Recursive summarization hit depth limit
Fix:
  Existing: Map-reduce handles up to 100KB emails
  If still failing: stream processing not yet implemented
```

### "Rate limit exceeded"
```
Issue: Composio API rate limit 429
Fix:
  - ComposioToolManager auto-retries (up to 3 times)
  - With exponential backoff
  - If all retries fail: return error to user
  - User can retry after ~1 minute
```

---

## Extensions & Future Work

### Planned Features
1. **Email Categorization**: Auto-tag emails by category
2. **Spam Detection**: Identify suspicious emails
3. **Follow-up Reminders**: Track emails needing replies
4. **Template Library**: Common reply templates
5. **Multi-language Support**: Compose/summarize in other languages

### Integration Points
```
Brain (orchestrator) ← GmailAgent
                    ↓
Integrations Agent ← OAuth/connections
                    ↓
Database ← UserConnection, ConversationHistory
```

---

## Configuration & Deployment

### Environment Variables
```bash
# Composio
COMPOSIO_API_KEY=your_api_key

# LLM Providers (optional fallback chain)
CEREBRAS_API_KEY=...
NVIDIA_API_KEY=...
GROQ_API_KEY=...

# Database
DATABASE_URL=postgresql://...
```

### Running the Agent

**Standalone:**
```bash
cd backend/agents/gmail_agent
python -c "from __init__ import run_agent; run_agent()"
# Listens on http://localhost:8003
```

**With Orchestrator:**
```bash
cd backend
python main.py
# Brain routes to gmail_agent as needed
```

---

## References

### Internal Documentation
- [Credentials & Communication](./CREDENTIALS_AND_COMMUNICATION.md) - Detailed credential retrieval and Composio communication flow
- [BaseAgent Framework](../base/README.md)
- [Connection System Generalization](../../CONNECTION_SYSTEM_GENERALIZATION.md)
- [Orchestrator Brain](../../orchestrator/brain.py)

### External References
- [Composio Gmail Docs](https://docs.composio.dev/docs/api/apps/gmail)
- [Gmail API Operators](https://support.google.com/mail/answer/7190)
