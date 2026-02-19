---
id: gmail_agent
name: Gmail Agent
port: 8003
version: 1.0.0
---

# Gmail Agent

Advanced Gmail automation agent with native Composio SDK integration and per-user authentication.

## Capabilities

- **Email Search**: Natural language email search with filters (from, to, subject, date ranges, labels)
- **Email Reading**: Read full email content including threads and attachments metadata
- **Email Sending**: Compose and send emails with HTML support, CC/BCC, and attachments
- **Email Management**: Archive, delete, star/unstar, mark as read/unread
- **Draft Management**: Create, read, update, and send drafts
- **Label Operations**: Add, remove, and create Gmail labels
- **Attachment Handling**: List attachments, get attachment IDs for downloading
- **Thread Navigation**: Navigate email threads, get thread context
- **Bulk Operations**: Process multiple emails at once (batch archive, label, etc.)
- **Smart Filtering**: Filter by unread, starred, important, promotions, social tabs

## When to Use

Use this agent when the user:
- Mentions Gmail, email, inbox, or messages specifically
- Wants to search through their emails with complex criteria
- Needs to send professional emails with formatting
- Wants to organize emails with labels
- Asks to clean up their inbox (archive/delete in bulk)
- Needs to work with email drafts
- Mentions managing email threads or conversations
- Wants to handle email attachments

## NOT For

- Calendar events or scheduling → use Calendar Agent (future)
- Spreadsheet processing → use Spreadsheet Agent  
- Document analysis → use Document Agent
- Web research → use Browser Agent
- Accounting/invoicing → use Zoho Books Agent

## Example Prompts

- "Search my emails from boss@company.com in the last week"
- "Show me all unread emails with attachments"
- "Draft an email to team@company.com about the project update"
- "Archive all promotional emails from last month"
- "Create a label called 'Important Clients' and apply it to emails from X"
- "Mark all emails in the 'Newsletter' folder as read"
- "Get the attachment from John's email about the proposal"
- "Reply to the latest thread from Sarah"

## Authentication Requirements

- **Requires**: Active Gmail connection via Composio OAuth
- **Multi-user**: Each user must authenticate their own Gmail account
- **Connection Setup**: User must visit `/connections` page and link Gmail
- **Entity ID**: Uses `user_id` as the Composio entity identifier

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/execute` | POST | Main execution endpoint (accepts UAP format) |
| `/search` | POST | Search emails with filters |
| `/read/{message_id}` | GET | Read a specific email |
| `/send` | POST | Send a new email |
| `/reply/{message_id}` | POST | Reply to an email |
| `/draft` | POST | Create or manage drafts |
| `/labels` | GET/POST | Manage Gmail labels |
| `/archive/{message_id}` | POST | Archive an email |
| `/delete/{message_id}` | DELETE | Delete an email |
| `/star/{message_id}` | POST | Star/unstar an email |
| `/health` | GET | Health check |

## Input Format

### Execute Endpoint (Preferred)
```json
{
  "prompt": "Find emails from john@example.com about the project",
  "payload": {
    "user_id": "user_123"
  },
  "task_id": "task_abc"
}
```

### Direct Tool Calls
```json
{
  "user_id": "user_123",
  "action": "search",
  "params": {
    "query": "from:john@example.com subject:project",
    "max_results": 10
  }
}
```

## Output Format

Returns standard `AgentResponse`:
```json
{
  "status": "COMPLETE|ERROR|NEEDS_INPUT",
  "result": {
    "emails": [...],
    "summary": "Found 5 emails from john@example.com",
    "actions_taken": ["searched", "filtered"]
  },
  "metadata": {
    "execution_time": 1.2,
    "tool": "GMAIL_SEARCH_EMAILS"
  }
}
```

## Available Composio Tools (23 total)

- GMAIL_SEARCH_EMAILS
- GMAIL_GET_EMAIL
- GMAIL_SEND_EMAIL
- GMAIL_REPLY_TO_EMAIL
- GMAIL_CREATE_DRAFT
- GMAIL_UPDATE_DRAFT
- GMAIL_SEND_DRAFT
- GMAIL_DELETE_DRAFT
- GMAIL_LIST_THREADS
- GMAIL_GET_THREAD
- GMAIL_MODIFY_MESSAGE
- GMAIL_TRASH_MESSAGE
- GMAIL_UNTRASH_MESSAGE
- GMAIL_DELETE_MESSAGE
- GMAIL_CREATE_LABEL
- GMAIL_DELETE_LABEL
- GMAIL_ADD_LABEL
- GMAIL_REMOVE_LABEL
- GMAIL_GET_ATTACHMENT
- GMAIL_LIST_LABELS
- GMAIL_MODIFY_LABELS
- GMAIL_BATCH_MODIFY
- GMAIL_GET_PROFILE

## Technical Details

- **SDK**: Native Composio Python SDK (not MCP)
- **Auth**: Per-user OAuth via ComposioAuthManager
- **LLM**: Uses centralized InferenceService
- **Memory**: In-memory session cache (per user_id)
- **Rate Limits**: Respects Gmail API quotas
- **Error Handling**: Returns structured errors with retry suggestions

## Configuration

Required environment variables:
```bash
COMPOSIO_API_KEY=your_api_key
GMAIL_AGENT_PORT=8003
```

## Notes

- This agent uses the newer Composio SDK approach (better than mail_agent's MCP approach)
- Session state is per-user and in-memory (lost on restart)
- For production: Consider backing memory with Redis or CMS
- Attachments are returned as metadata (IDs) — actual download happens client-side
- HTML emails are supported for sending
- Thread context is maintained for follow-up questions

## Integration Status

- ✅ Composio SDK configured
- ✅ Per-user authentication working
- ⚠️ Not yet registered in orchestrator (needs DB entry + this SKILL.md)
- ⚠️ Response format needs UAP wrapping for orchestrator compatibility
- ⚠️ Memory is in-memory only (needs Redis/CMS backing)

## Future Enhancements

- Add calendar integration for meeting extraction
- Implement smart categorization (urgent/important)
- Add sentiment analysis for email triage
- Support multiple Gmail accounts per user
- Add email templates library
- Implement scheduled sends
