---
id: mail_agent
name: Mail Agent
port: 8040
version: 2.0.0
---

# Mail Agent

Smart, stateful Gmail assistant with LLM-powered email understanding.

## Capabilities

- Search emails using natural language queries
- Summarize email threads and batches
- Draft context-aware replies based on thread history
- Send new emails with HTML support
- Manage emails: archive, delete, star, label
- Extract action items, tasks, and deadlines from emails
- Download and analyze email attachments

## When to Use

Use this agent when the user:
- Mentions email, Gmail, inbox, or messages
- Wants to search, read, or send emails
- Asks to reply to or forward emails
- Needs to summarize email threads
- Wants to find action items in emails
- Mentions drafting or composing emails

## NOT For

- Spreadsheet files → use Spreadsheet Agent
- PDF/Word documents → use Document Agent
- Web browsing → use Browser Agent
- Calendar events → (future Calendar Agent)

## Example Prompts

- "Find emails from John about the budget proposal"
- "Summarize my unread emails from today"
- "Draft a reply thanking them for the meeting"
- "Mark all emails from newsletter@ as read"
- "What action items do I have from recent emails?"

## Notes

- Requires Composio API key for Gmail integration
- Supports multi-turn dialogues for clarification
- Maintains session state for follow-up queries
