---
id: mail_agent
name: Mail Agent
port: 8040
version: 1.0.0
description: >
  Deprecated mail agent retained for compatibility with legacy Gmail-style workflows.
  Prefer gmail_agent for new email tasks.
model: cerebras/llama-3.3-70b
context_strategy: minimal
requires_auth: true
composio_app_slug: gmail
deprecated: true
prefer: gmail_agent
triggers: []
capabilities:
  - search_emails
  - summarize_threads
  - draft_reply
  - send_email
  - manage_emails
  - extract_action_items
not_for:
  - spreadsheets
  - PDF documents
  - web browsing
  - calendar events
---

# Mail Agent (Legacy)

Smart, stateful Gmail assistant with LLM-powered email understanding.
**Note:** Prefer `gmail_agent` for new tasks — this agent is retained for backward compatibility.

## Capabilities

- Search emails using natural language queries
- Summarize email threads and batches
- Draft context-aware replies based on thread history
- Send new emails with HTML support
- Manage emails: archive, delete, star, label
- Extract action items, tasks, and deadlines from emails
- Download and analyze email attachments

## Notes

- Requires Composio API key for Gmail integration
- Supports multi-turn dialogues for clarification
- Maintains session state for follow-up queries
