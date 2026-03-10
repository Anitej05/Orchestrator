# Gmail Agent

**Clean Composio-native Gmail agent using official Composio SDK with 23 tools + 2 triggers**

## Overview

Gmail Agent is a next-generation email management agent built exclusively with Composio's official Gmail tools. It replaces the legacy mail_agent's custom MCP implementation with direct SDK integration for better performance, reliability, and maintainability.

### 📚 Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Complete architecture, data flows, LLM system, security, and performance
- **[CREDENTIALS_AND_COMMUNICATION.md](./CREDENTIALS_AND_COMMUNICATION.md)** - Detailed credential retrieval, database schema, encryption, and Composio communication
- **[SKILL.md](./SKILL.md)** - API reference and capability specifications
- **[QUICK_START.md](./QUICK_START.md)** - Setup and testing guide

## Architecture

```
Gmail Agent
├── config.py         # Configuration and settings
├── memory.py         # Conversation state management
├── llm.py           # LLM client (summarization, drafting, extraction)
├── tools.py         # ComposioToolManager - wraps all 23 Gmail tools
├── service.py       # GmailService - core business logic
├── schemas.py       # Pydantic models for requests/responses
└── agent.py         # FastAPI application with 30+ endpoints
```

### Key Components

**ComposioToolManager** (`tools.py`)
- Wraps Composio SDK with per-user authentication
- Handles all 23 Gmail tools (fetch, send, draft, labels, attachments, etc.)
- Error handling and logging

**GmailService** (`service.py`)
- Core business logic layer
- Email operations: search, send, reply, delete
- Draft management: create, list, send, delete
- Label operations: add, remove, list, create
- LLM-enhanced features: summarize, draft replies, extract actions
- Contact & thread operations
- Attachment handling with file storage

**LLMClient** (`llm.py`)
- Query optimization (natural language → Gmail syntax)
- Email summarization (with recursive map-reduce for large content)
- Smart reply drafting
- Action item extraction
- Multi-provider support (Cerebras, NVIDIA, Groq)

**AgentMemory** (`memory.py`)
- Per-user conversation history
- Search result caching for follow-up actions
- Draft context storage

## Available Tools (23)

### Email Operations (8)
- `GMAIL_FETCH_EMAILS` - Search/list emails
- `GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID` - Get single email
- `GMAIL_FETCH_MESSAGE_BY_THREAD_ID` - Get thread messages
- `GMAIL_SEND_EMAIL` - Send email with attachments
- `GMAIL_REPLY_TO_THREAD` - Reply in thread
- `GMAIL_DELETE_MESSAGE` - Permanent delete
- `GMAIL_MOVE_TO_TRASH` - Soft delete
- `GMAIL_GET_ATTACHMENT` - Download attachment

### Draft Operations (4)
- `GMAIL_CREATE_EMAIL_DRAFT` - Create draft
- `GMAIL_LIST_DRAFTS` - List drafts
- `GMAIL_SEND_DRAFT` - Send draft
- `GMAIL_DELETE_DRAFT` - Delete draft

### Label Management (6)
- `GMAIL_ADD_LABEL_TO_EMAIL` - Add labels
- `GMAIL_MODIFY_THREAD_LABELS` - Modify thread labels
- `GMAIL_CREATE_LABEL` - Create custom label
- `GMAIL_LIST_LABELS` - List all labels
- `GMAIL_REMOVE_LABEL` - Delete label
- `GMAIL_PATCH_LABEL` - Update label properties

### Contact Operations (3)
- `GMAIL_GET_CONTACTS` - List contacts
- `GMAIL_SEARCH_PEOPLE` - Search contacts
- `GMAIL_GET_PEOPLE` - Get person details

### Thread & Profile (2)
- `GMAIL_LIST_THREADS` - List threads
- `GMAIL_GET_PROFILE` - Get Gmail profile

## API Endpoints

### Email Operations
```
POST   /search                    # Search emails
POST   /send                      # Send email
POST   /reply                     # Reply to email/thread
GET    /message/{user_id}/{id}    # Get single email
DELETE /message/{user_id}/{id}    # Delete email (permanent)
POST   /trash/{user_id}/{id}      # Move to trash
```

### Draft Operations
```
POST   /draft/create              # Create draft
GET    /drafts/{user_id}          # List drafts
POST   /draft/{user_id}/{id}/send # Send draft
DELETE /draft/{user_id}/{id}      # Delete draft
```

### Label Operations
```
POST   /labels/add                # Add labels to email
GET    /labels/{user_id}          # List all labels
POST   /labels/create/{user_id}   # Create label
```

### LLM-Enhanced Operations
```
POST   /summarize                 # Summarize emails (LLM)
POST   /draft-reply               # AI-generated reply
POST   /extract-actions           # Extract action items (LLM)
```

### Attachments
```
POST   /attachments/download      # Download attachments
```

### Contacts & Threads
```
GET    /contacts/{user_id}        # List contacts
POST   /contacts/search/{user_id} # Search contacts
GET    /threads/{user_id}         # List threads
GET    /thread/{user_id}/{id}     # Get thread messages
```

### Profile
```
GET    /profile/{user_id}         # Get Gmail profile
```

### Orchestrator Integration
```
POST   /execute                   # Execute natural language action
```

## Usage Examples

### Search Emails
```python
POST /search
{
  "user_id": "user123",
  "query": "unread emails from boss",
  "max_results": 10
}

Response:
{
  "success": true,
  "messages": [...],
  "total_count": 5,
  "query_used": "is:unread from:boss"
}
```

### Summarize Emails
```python
POST /summarize
{
  "user_id": "user123",
  "message_ids": ["msg1", "msg2", "msg3"]
}

Response:
{
  "success": true,
  "summary": "=== SUMMARY ===\n- Email 1: Project deadline extended...",
  "emails_summarized": 3
}
```

### Draft Smart Reply
```python
POST /draft-reply
{
  "user_id": "user123",
  "message_id": "msg123",
  "user_instructions": "Thank them for the update"
}

Response:
{
  "success": true,
  "message": "Smart reply draft created",
  "draft": {...},
  "generated_content": {
    "subject": "Re: Project Update",
    "body": "<p>Thank you for the update...</p>"
  }
}
```

### Send Email
```python
POST /send
{
  "user_id": "user123",
  "to": "colleague@example.com",
  "subject": "Project Status",
  "body": "Here's the status update...",
  "is_html": false
}

Response:
{
  "success": true,
  "message": "Email sent successfully",
  "data": {...}
}
```

## Per-User Authentication

Gmail Agent uses **per-user connections** via the Composio integration system:

1. User connects Gmail via OAuth (handled by `services/integrations/composio_auth.py`)
2. Connection ID stored encrypted in database
3. Each request includes `user_id`
4. Agent retrieves user's connection: `get_auth_manager().get_connection_for_agent(user_id, "gmail")`
5. All Composio tools executed with user's connection

**Error Handling:**
If user not connected → `ValueError: User {user_id} not connected to Gmail`

## LLM Features

### Query Optimization
Converts natural language to Gmail search syntax:
- "unread emails from John" → `is:unread from:John`
- "Demo Request" → `subject:"Demo Request"`

### Summarization
- Recursive map-reduce for large emails (>4000 chars)
- Hierarchical summarization for batches
- High-density bullet points

### Smart Reply
- Context-aware draft generation
- Professional tone
- HTML output

### Action Extraction
- Deadlines, requests, meetings, tasks
- Priority classification (high/medium/low)
- Source tracking

## Configuration

### Environment Variables
```bash
# Required
COMPOSIO_API_KEY=your_composio_key

# LLM Providers (at least one required)
GROQ_API_KEY=your_groq_key
CEREBRAS_API_KEY=your_cerebras_key
NVIDIA_API_KEY=your_nvidia_key

# Optional
MAX_SEARCH_RESULTS=50
DEFAULT_PAGE_SIZE=10
MAX_CONCURRENT_FETCHES=5
```

### Storage
Attachments stored in: `storage/gmail_agent/attachments/{user_id}/`

## Running the Agent

### Standalone
```bash
cd backend/agents/gmail_agent
python agent.py
# Runs on http://0.0.0.0:8003
```

### Via Orchestrator
```python
# agents/orchestrator/agent_manager.py
AGENT_CONFIGS = {
    "gmail_agent": {
        "name": "Gmail Agent",
        "base_url": "http://localhost:8003",
        "capabilities": ["email", "gmail", "draft", "summarize"]
    }
}
```

## Migration from mail_agent

### Key Differences

| Aspect | mail_agent | gmail_agent |
|--------|------------|-------------|
| **Architecture** | Custom MCP HTTP client | Direct Composio SDK |
| **Tools** | ~10 custom wrappers | 23 official tools |
| **Authentication** | MCP server | Per-user OAuth |
| **Code Lines** | ~2500 | ~1500 (40% reduction) |
| **Maintenance** | Custom updates needed | SDK auto-updated |

### Migration Steps

1. **User connections preserved** - No user action needed (same OAuth system)
2. **Update orchestrator** - Point to `gmail_agent` instead of `mail_agent`
3. **Test endpoints** - All functionality maintained
4. **Gradual rollout** - Run both agents in parallel initially

## Error Handling

### Common Errors

**User Not Connected**
```json
{
  "detail": "User user123 not connected to Gmail"
}
```
**Solution:** User must connect Gmail via `/api/integrations/auth/start/{user_id}?apps=gmail`

**Tool Execution Failed**
```json
{
  "success": false,
  "error": "Invalid message ID"
}
```
**Solution:** Verify message/draft/thread IDs are valid

**LLM Unavailable**
```json
{
  "success": false,
  "error": "All LLM providers failed"
}
```
**Solution:** Check API keys for Groq/Cerebras/NVIDIA

## Performance

### Benchmarks
- **Search:** < 2 seconds
- **Send Email:** < 1 second
- **Summarize (5 emails):** < 10 seconds
- **Batch Download (10 attachments):** < 5 seconds

### Optimization
- Concurrent email fetching (max 5 parallel)
- Service instance caching (per-user)
- Memory-based search result caching
- Recursive summarization for large content

## Logging

Logs written to: `logs/gmail_agent.log`

```python
import logging
logger = logging.getLogger("gmail_agent")

# Levels used:
# INFO  - Normal operations
# DEBUG - Detailed tool execution
# WARNING - Recoverable errors
# ERROR - Failures requiring attention
```

## Testing

### Unit Tests
```bash
pytest agents/gmail_agent/tests/
```

### Integration Tests
```bash
pytest agents/gmail_agent/tests/test_integration.py
```

### Manual Testing
```bash
# Search
curl -X POST http://localhost:8003/search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","query":"is:unread","max_results":5}'

# Send
curl -X POST http://localhost:8003/send \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","to":"test@example.com","subject":"Test","body":"Hello"}'
```

## Troubleshooting

### Agent Won't Start
- Check Composio API key in `.env`
- Verify port 8003 not in use
- Check logs for import errors

### User Connection Errors
- Verify user connected via `/api/integrations/status/{user_id}`
- Check OAuth tokens not expired
- Refresh connection: `/api/integrations/refresh/{user_id}/gmail`

### LLM Features Not Working
- Verify at least one LLM API key configured
- Check provider status (Groq/Cerebras/NVIDIA)
- Review LLM logs for rate limits

### Attachment Downloads Fail
- Check storage directory permissions: `storage/gmail_agent/attachments/`
- Verify disk space available
- Check attachment IDs are valid

## Future Enhancements

### Phase 7: Advanced Features
- Email templates (reusable)
- Scheduled sending
- ML-based categorization
- Bulk operations UI
- Usage analytics
- Calendar integration
- Task integration

### Phase 8: Performance
- Redis caching
- Background job queue
- Connection pooling
- Lazy loading for large result sets

## Support

**Issues:** Report to development team  
**Documentation:** See `IMPLEMENTATION_PLAN.md` for architecture details  
**API Reference:** Auto-generated at `/docs` (FastAPI Swagger UI)

## License

Internal use only - Orbimesh project

---

**Version:** 1.0.0  
**Last Updated:** February 7, 2026  
**Status:** Production Ready ✅
