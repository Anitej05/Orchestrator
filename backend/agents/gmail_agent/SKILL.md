---
id: gmail_agent
name: Gmail Agent
port: 8003
version: 2.0.0
description: >
  Advanced Gmail automation with full Composio SDK (v3) integration.
  Covers all 60 Gmail tools: search, send, reply, forward, drafts, labels,
  filters, batch operations, settings, contacts, and universal tool execution.
model: cerebras/llama-3.3-70b
context_strategy: minimal
requires_auth: true
composio_app_slug: gmail
triggers:
  - email
  - gmail
  - inbox
  - send mail
  - draft
  - attachments
  - label
  - thread
  - compose
  - reply
  - forward
  - filter
  - batch
  - archive
  - vacation reply
capabilities:
  - search_emails
  - send_email
  - reply_email
  - forward_email
  - get_email
  - summarize_emails
  - draft_smart_reply
  - extract_action_items
  - manage_drafts
  - manage_labels
  - batch_operations
  - manage_filters
  - get_settings
  - execute_gmail_tool
not_for:
  - calendar events
  - spreadsheets
  - document processing
  - web browsing
  - accounting
  - non-Gmail integrations (use integrations_agent)
---

# Gmail Agent

Advanced Gmail automation agent with full Composio SDK v3 integration and per-user authentication.

## Capabilities (14 total)

- **Email Search** — NL queries with LLM optimization
- **Send & Reply** — HTML, CC/BCC, attachments
- **Forward** — Forward emails with optional message
- **Draft Management** — Create, get, update, send, delete drafts
- **Label Operations** — Create, delete, rename, list labels
- **Batch Operations** — Archive, delete, star, mark read/unread in bulk
- **Filter Management** — Create, list, delete Gmail filter rules
- **Settings** — Vacation auto-reply, forwarding, language, aliases
- **Universal Tool** — `execute_gmail_tool` for any of 60 Composio tools
- **AI Features** — Summarize emails, smart reply drafts, extract action items

## Composio Tools (60 total)

All 60 Composio Gmail tools are accessible:
- 35 have explicit wrappers in `tools.py`
- Remaining 25 are accessible via `execute_any_tool(slug, params)`

## Triggers (2)

- `GMAIL_NEW_GMAIL_MESSAGE` — New incoming email
- `GMAIL_EMAIL_SENT_TRIGGER` — Email sent

## Authentication

- **Requires**: Active Gmail connection via Composio OAuth
- **Multi-user**: Each user authenticates their own Gmail account
- **SDK**: Composio v3 (`composio.tools.execute()`)
