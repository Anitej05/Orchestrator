# agents/mail_agent/config.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure logging with DEBUG level for verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agents.mail_agent.config")

# Configuration
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
MCP_URL = os.getenv("GMAIL_MCP_URL")  # e.g., https://mcp.composio.dev/gmail/sse?user_id=...
CONNECTION_ID = os.getenv("GMAIL_CONNECTION_ID")  # Gmail connection ID

if not COMPOSIO_API_KEY:
    logger.warning("COMPOSIO_API_KEY not set. Gmail agent will not function.")
if not MCP_URL:
    logger.warning("GMAIL_MCP_URL not set. Gmail agent will not function.")
if not CONNECTION_ID:
    logger.warning("GMAIL_CONNECTION_ID not set. Using URL without connection ID.")

# Agent Definition
AGENT_DEFINITION = {
    "id": "mail_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Mail Agent",
    "description": "Gmail integration via Composio. Read, send, search, and manage emails.",
    "capabilities": [
        "read emails",
        "send emails",
        "search emails",
        "fetch emails",
        "compose emails",
        "reply to emails",
        "manage email labels",
        "get email threads",
        "fetch unread emails",
        "send email with attachments",
        "search inbox",
        "check email",
        "email automation",
        "gmail integration",
        "email management",
        "fetch sent emails",
        "get sent email content",
        "view sent mail",
        "check sent folder",
        "get last sent email",
        "verify sent email"
    ],
    "price_per_call_usd": 0.005,
    "status": "active",
    "agent_type": "http_rest",
    "connection_config": {
        "base_url": "http://localhost:8040"
    },
    "endpoints": [
        {
            "endpoint": "/search",
            "http_method": "POST",
            "description": "Smart semantic search (state-aware). Stores results for follow-up.",
            "parameters": [
                {
                    "name": "query",
                    "param_type": "string",
                    "required": True,
                    "description": "Search query (natural language or specific)"
                },
                {
                    "name": "max_results",
                    "param_type": "integer",
                    "required": False,
                    "description": "Max results per keyword (default: 10)"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID"
                }
            ]
        },
        {
            "endpoint": "/summarize_emails",
            "http_method": "POST",
            "description": "Smart batch summarization. Can use context from previous search.",
            "parameters": [
                {
                    "name": "message_ids",
                    "param_type": "array",
                    "required": False,
                    "description": "List of message IDs. If empty, summarizes last search results."
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID"
                },
                {
                    "name": "use_history",
                    "param_type": "boolean",
                    "required": False,
                    "description": "Whether to look up recent search results if no IDs provided (default: True)"
                }
            ]
        },
        {
            "endpoint": "/draft_reply",
            "http_method": "POST",
            "description": "Smartly draft a reply based on the full thread context and user intent",
            "parameters": [
                {
                    "name": "message_id",
                    "param_type": "string",
                    "required": True,
                    "description": "ID of the email/thread to reply to"
                },
                {
                    "name": "intent",
                    "param_type": "string",
                    "required": True,
                    "description": "Your intent (e.g. 'accept with thanks', 'decline gently')"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID"
                }
            ]
        },
        {
            "endpoint": "/extract_action_items",
            "http_method": "POST",
            "description": "Extract tasks, deadlines, and meetings from emails",
            "parameters": [
                {
                    "name": "message_ids",
                    "param_type": "array",
                    "required": False,
                    "description": "List of message IDs. If empty, scans last search results."
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID"
                },
                {
                    "name": "use_history",
                    "param_type": "boolean",
                    "required": False,
                    "description": "Use recent search results if no IDs provided (default: True)"
                }
            ]
        },
        {
             "endpoint": "/get_message",
             "http_method": "POST",
             "description": "Get full details of a specific message. Includes AI ANALYSIS of attachments (will advise if you need to download).",
             "parameters": [
                 {
                    "name": "message_id",
                    "param_type": "string",
                    "required": True,
                    "description": "Gmail message ID"
                 }
             ]
        },
        {
            "endpoint": "/download_attachments",
            "http_method": "POST",
            "description": "Download all attachments from an email to the agent's file system",
            "parameters": [
                {
                    "name": "message_id",
                    "param_type": "string",
                    "required": True,
                    "description": "Gmail message ID"
                },
                {
                    "name": "thread_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Thread ID (optional)"
                }
            ]
        },
        {
            "endpoint": "/files",
            "http_method": "GET",
            "description": "List downloaded attachment files",
            "parameters": [
                 {
                    "name": "status",
                    "param_type": "string",
                    "required": False,
                    "description": "Filter by status: active, archived"
                 }
            ]
        },
        {
            "endpoint": "/send_email",
            "http_method": "POST",
            "description": "Send a new email. Supports HTML, attachments, and PREVIEW. To pause and review draft, set show_preview=true first.",
            "parameters": [
                {
                    "name": "to",
                    "param_type": "array",
                    "required": True,
                    "description": "Recipient address(es)"
                },
                {
                    "name": "subject",
                    "param_type": "string",
                    "required": True,
                    "description": "Email subject"
                },
                {
                    "name": "body",
                    "param_type": "string",
                    "required": True,
                    "description": "Email body (Markdown/HTML/Text)"
                },
                {
                    "name": "is_html",
                    "param_type": "boolean",
                    "required": False,
                    "description": "Format as HTML? (default: false)"
                },
                {
                    "name": "attachment_file_ids",
                    "param_type": "array",
                    "required": False,
                    "description": "File IDs to attach"
                },
                {
                    "name": "show_preview",
                    "param_type": "boolean",
                    "required": False,
                    "description": "SET TO TRUE to pause and show draft in canvas before sending. Default: False (sends immediately)"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID"
                }
            ]
        },
        {
            "endpoint": "/manage_emails",
            "http_method": "POST",
            "description": "Unified endpoint to Archive, Delete, Star, Label, or Mark Read/Unread.",
            "parameters": [
                {
                    "name": "message_ids",
                    "param_type": "array",
                    "required": False,
                    "description": "List of message IDs. If empty, uses last search results."
                },
                {
                    "name": "action",
                    "param_type": "string",
                    "required": True,
                    "description": "Action: mark_read, mark_unread, archive, delete, star, unstar, add_labels, remove_labels"
                },
                {
                    "name": "labels",
                    "param_type": "array",
                    "required": False,
                    "description": "Required for add/remove_labels"
                },
                {
                    "name": "use_history",
                    "param_type": "boolean",
                    "required": False,
                    "description": "Apply to last search results if IDs empty (default: True)"
                }
            ]
        },
        {
            "endpoint": "/files/{file_id}",
            "http_method": "GET",
            "description": "Get metadata for a specific file",
            "parameters": []
        }
    ]
}
