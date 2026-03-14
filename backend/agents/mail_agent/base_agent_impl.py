"""
Mail Agent v2.0 - Complete BaseAgent Implementation (Fixed)

Full Gmail integration with correct method names from actual client.py and llm.py.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from backend.agents.base import BaseAgent, AgentServices, AgentConfig
from backend.agents.base.types import ExecutionContext
from backend.agents.base.capability import capability, ParameterSchema

from .config import COMPOSIO_API_KEY, MCP_URL, logger
from .client import GmailClient
from .llm_helpers import MailLLMHelpers
from .memory import AgentMemory

logger = logging.getLogger("agents.mail_agent")


@dataclass
class MailAgentConfig(AgentConfig):
    """Configuration specific to Mail Agent."""

    max_results_per_search: int = 50
    attachment_ttl_hours: int = 72
    enable_attachment_analysis: bool = True
    enable_batch_operations: bool = True
    max_batch_size: int = 100


class SmartResolver:
    """Smart resolver for message IDs and email references."""

    def __init__(self, gmail_client: GmailClient, memory: AgentMemory):
        self.gmail = gmail_client
        self.memory = memory

    async def resolve_message_ids(
        self, identifiers: List[str], user_id: str = "me"
    ) -> List[str]:
        """Resolve various message identifiers to actual message IDs."""
        resolved = []

        for identifier in identifiers:
            # Check if it's already a message ID format
            if len(identifier) > 20 and not identifier.startswith("thread:"):
                resolved.append(identifier)
                continue

            # Try to resolve from memory
            if identifier in self.memory.resolved_ids:
                resolved.append(self.memory.resolved_ids[identifier])
                continue

            # Try to search for it
            try:
                search_results = await self.gmail.semantic_search(
                    identifier, 5, user_id
                )
                if search_results.get("success"):
                    messages = search_results.get("data", {}).get("messages", [])
                    if messages:
                        msg_id = messages[0].get("id")
                        if msg_id:
                            self.memory.resolved_ids[identifier] = msg_id
                            resolved.append(msg_id)
            except Exception as e:
                logger.warning(f"Failed to resolve identifier '{identifier}': {e}")

        return resolved

    def resolve_from_history(self, user_id: str = "me") -> Optional[List[str]]:
        """Get message IDs from recent search history."""
        history = self.memory.get_recent_search(user_id)
        if history:
            return history.get("message_ids", [])
        return None


class MailAgent(BaseAgent, MailLLMHelpers):
    """
    Complete Gmail integration agent with corrected method mappings.
    
    Inherits from MailLLMHelpers for all LLM methods.
    """

    def __init__(
        self,
        agent_id: str = "mail_agent",
        agent_name: str = "Mail Agent",
        services: Optional[AgentServices] = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services,
            config=config or MailAgentConfig(),
        )

        self.gmail_client: Optional[GmailClient] = None
        # No need for self.llm_client - LLM methods are inherited from MailLLMHelpers
        self.memory: Optional[AgentMemory] = None
        self.resolver: Optional[SmartResolver] = None

        # Metrics
        self._metrics = {
            "emails_sent": 0,
            "emails_read": 0,
            "searches": 0,
            "attachments_downloaded": 0,
        }

    async def _initialize_resources(self):
        """Initialize Gmail client, LLM client, and memory."""
        logger.info("Initializing Mail Agent resources...")

        if not COMPOSIO_API_KEY or not MCP_URL:
            raise RuntimeError(
                "Mail Agent requires COMPOSIO_API_KEY and GMAIL_MCP_URL environment variables"
            )

        self.gmail_client = GmailClient()
        self.memory = AgentMemory()
        self.resolver = SmartResolver(self.gmail_client, self.memory)

        logger.info("Mail Agent resources initialized successfully")

    async def _cleanup_resources(self):
        """Cleanup resources."""
        logger.info("Cleaning up Mail Agent resources...")
        if self.memory:
            self.memory.clear()

    async def _get_custom_metrics(self) -> Optional[Dict[str, Any]]:
        """Return Mail Agent specific metrics."""
        metrics = self._metrics.copy()
        if self.gmail_client and hasattr(self.gmail_client, "metrics"):
            metrics.update(self.gmail_client.metrics)
        return metrics

    # ========================================================================
    # CAPABILITIES - Search & Read
    # ========================================================================

    @capability(
        name="search_emails",
        description="Search emails using natural language or Gmail search operators",
        parameters=[
            ParameterSchema(
                name="query",
                type="string",
                description="Search query (natural language or Gmail operators)",
                required=True,
            ),
            ParameterSchema(
                name="max_results",
                type="integer",
                description="Maximum number of results",
                required=False,
                default=10,
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
        ],
    )
    async def search_emails(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Search emails with semantic understanding."""
        query = params.get("query", "")
        max_results = params.get("max_results", 10)
        user_id = params.get("user_id", "me")

        logger.info(f"Searching emails: '{query}' (max: {max_results})")

        try:
            # Optimize query if it's natural language
            if not any(
                op in query for op in ["from:", "to:", "subject:", "label:", "is:"]
            ):
                optimized_query = await self.generate_optimized_query(query)
                query = optimized_query

            result = await self.gmail_client.semantic_search(
                query, max_results, user_id
            )
            self._metrics["searches"] += 1

            if result.get("success"):
                messages = result.get("data", {}).get("messages", [])
                message_ids = [msg.get("id") for msg in messages if msg.get("id")]

                # Save to memory
                self.memory.save_search_results(user_id, message_ids)

                return {
                    "success": True,
                    "data": {
                        "messages": messages,
                        "total_found": len(messages),
                        "query": query,
                    },
                    "message": f"Found {len(messages)} emails",
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Search failed"),
                    "data": {"query": query},
                }
        except Exception as e:
            logger.error(f"Email search failed: {e}")
            return {"success": False, "error": f"Search failed: {str(e)}"}

    @capability(
        name="read_email",
        description="Get full details of a specific email including attachments",
        parameters=[
            ParameterSchema(
                name="message_id",
                type="string",
                description="Gmail message ID",
                required=True,
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
            ParameterSchema(
                name="analyze_attachments",
                type="boolean",
                description="Analyze attachment content",
                required=False,
                default=True,
            ),
        ],
    )
    async def read_email(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Read a specific email with full details."""
        message_id = params.get("message_id", "")
        user_id = params.get("user_id", "me")
        analyze_attachments = params.get("analyze_attachments", True)

        logger.info(f"Reading email: {message_id}")

        try:
            # Use call_tool with GET_EMAIL action (correct method name)
            result = await self.gmail_client.call_tool("GET_EMAIL", {"id": message_id})
            self._metrics["emails_read"] += 1

            if result.get("success"):
                message_data = result.get("data", {})

                # Analyze attachments if requested
                if analyze_attachments and message_data.get("attachments"):
                    for attachment in message_data["attachments"]:
                        if attachment.get("size", 0) < 5 * 1024 * 1024:
                            attachment["ai_analysis"] = (
                                "Attachment available for download"
                            )

                return {
                    "success": True,
                    "data": message_data,
                    "message": f"Retrieved email: {message_data.get('subject', 'No subject')}",
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to read email"),
                }
        except Exception as e:
            logger.error(f"Read email failed: {e}")
            return {"success": False, "error": f"Failed to read email: {str(e)}"}

    # ========================================================================
    # CAPABILITIES - Send & Draft
    # ========================================================================

    @capability(
        name="send_email",
        description="Send an email with optional HTML formatting and attachments",
        parameters=[
            ParameterSchema(
                name="to",
                type="array",
                description="List of recipient email addresses",
                required=True,
            ),
            ParameterSchema(
                name="subject",
                type="string",
                description="Email subject line",
                required=True,
            ),
            ParameterSchema(
                name="body",
                type="string",
                description="Email body content (text or HTML)",
                required=True,
            ),
            ParameterSchema(
                name="cc",
                type="array",
                description="List of CC recipients",
                required=False,
                default=[],
            ),
            ParameterSchema(
                name="is_html",
                type="boolean",
                description="Whether body is HTML formatted",
                required=False,
                default=False,
            ),
            ParameterSchema(
                name="attachment_file_ids",
                type="array",
                description="File IDs to attach to the email",
                required=False,
                default=[],
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
        ],
    )
    async def send_email(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Send an email with attachments."""
        to = params.get("to", [])
        subject = params.get("subject", "")
        body = params.get("body", "")
        cc = params.get("cc", [])
        is_html = params.get("is_html", False)
        attachment_ids = params.get("attachment_file_ids", [])
        user_id = params.get("user_id", "me")

        logger.info(f"Sending email to: {to}, subject: '{subject}'")

        try:
            result = await self.gmail_client.send_email_with_attachments(
                to=to,
                subject=subject,
                body=body,
                cc=cc,
                is_html=is_html,
                attachment_file_ids=attachment_ids,
                user_id=user_id,
            )

            if result.get("success"):
                self._metrics["emails_sent"] += 1
                return {
                    "success": True,
                    "data": result.get("data"),
                    "message": f"Email sent successfully to {', '.join(to)}",
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to send email"),
                }
        except Exception as e:
            logger.error(f"Send email failed: {e}")
            return {"success": False, "error": f"Failed to send email: {str(e)}"}

    @capability(
        name="draft_reply",
        description="Draft a context-aware reply to an email thread",
        parameters=[
            ParameterSchema(
                name="message_id",
                type="string",
                description="ID of the email/message to reply to",
                required=True,
            ),
            ParameterSchema(
                name="intent",
                type="string",
                description="Your intent for the reply",
                required=True,
            ),
            ParameterSchema(
                name="tone",
                type="string",
                description="Tone of the reply",
                required=False,
                default="professional",
                enum=["professional", "casual", "formal", "friendly"],
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
        ],
    )
    async def draft_reply(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Draft a reply to an email."""
        message_id = params.get("message_id", "")
        intent = params.get("intent", "")
        tone = params.get("tone", "professional")
        user_id = params.get("user_id", "me")

        logger.info(f"Drafting reply to {message_id} with intent: '{intent}'")

        try:
            # Get the original message
            result = await self.gmail_client.call_tool("GET_EMAIL", {"id": message_id})
            if not result.get("success"):
                return {
                    "success": False,
                    "error": "Could not retrieve original message",
                }

            original = result.get("data", {})

            # Get thread context using batch_fetch_emails (correct method name)
            thread_messages = [original]
            if original.get("thread_id"):
                # Use semantic_search or batch fetch for thread
                thread_result = await self.gmail_client.semantic_search(
                    f"thread:{original['thread_id']}", 20, user_id
                )
                if thread_result.get("success"):
                    thread_messages = thread_result.get("data", {}).get(
                        "messages", [original]
                    )

            # Build thread content
            thread_content = "\n\n---\n\n".join(
                [
                    f"From: {m.get('from', '')}\nSubject: {m.get('subject', '')}\n\n{m.get('body', '')}"
                    for m in thread_messages
                ]
            )

            # Generate reply using draft_email_reply (correct method name)
            sender_name = original.get("from", "").split("<")[0].strip()
            draft_result = await self.draft_email_reply(
                thread_content, intent, sender_name
            )

            return {
                "success": True,
                "data": {
                    "draft": draft_result.get("reply_body", ""),
                    "subject": draft_result.get(
                        "subject", f"Re: {original.get('subject', '')}"
                    ),
                    "original_subject": original.get("subject"),
                    "original_from": original.get("from"),
                    "thread_id": original.get("thread_id"),
                    "message_id": message_id,
                },
                "message": "Reply drafted successfully",
            }
        except Exception as e:
            logger.error(f"Draft reply failed: {e}")
            return {"success": False, "error": f"Failed to draft reply: {str(e)}"}

    # ========================================================================
    # CAPABILITIES - Batch Operations
    # ========================================================================

    @capability(
        name="summarize_emails",
        description="Summarize multiple emails by their message IDs",
        parameters=[
            ParameterSchema(
                name="message_ids",
                type="array",
                description="List of message IDs to summarize",
                required=False,
                default=[],
            ),
            ParameterSchema(
                name="use_history",
                type="boolean",
                description="Use recent search results if message_ids is empty",
                required=False,
                default=True,
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
        ],
    )
    async def summarize_emails(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Summarize a batch of emails."""
        message_ids = params.get("message_ids", [])
        use_history = params.get("use_history", True)
        user_id = params.get("user_id", "me")

        # Resolve from history if needed
        if not message_ids and use_history:
            history = self.memory.get_recent_search(user_id)
            if history:
                message_ids = history.get("message_ids", [])

        if not message_ids:
            return {"success": False, "error": "No message IDs provided"}

        logger.info(f"Summarizing {len(message_ids)} emails")

        try:
            # Fetch all messages using batch_fetch_emails (correct method name)
            messages = await self.gmail_client.batch_fetch_emails(message_ids[:20])

            if not messages:
                return {"success": False, "error": "Could not retrieve any messages"}

            # Extract text content
            email_texts = []
            for msg in messages:
                content = f"Subject: {msg.get('subject', '')}\nFrom: {msg.get('from', '')}\n\n{msg.get('body', '')}"
                email_texts.append(content)

            # Generate summary using summarize_text_batch (correct method name)
            summary = await self.summarize_text_batch(email_texts)

            return {
                "success": True,
                "data": {
                    "summary": summary,
                    "messages_summarized": len(messages),
                    "message_ids": [m.get("id") for m in messages],
                },
                "message": f"Summarized {len(messages)} emails",
            }
        except Exception as e:
            logger.error(f"Summarize emails failed: {e}")
            return {"success": False, "error": f"Failed to summarize: {str(e)}"}

    @capability(
        name="extract_action_items",
        description="Extract tasks, deadlines, and action items from emails",
        parameters=[
            ParameterSchema(
                name="message_ids",
                type="array",
                description="List of message IDs to analyze",
                required=False,
                default=[],
            ),
            ParameterSchema(
                name="use_history",
                type="boolean",
                description="Use recent search results if message_ids is empty",
                required=False,
                default=True,
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
        ],
    )
    async def extract_action_items(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Extract action items from emails."""
        message_ids = params.get("message_ids", [])
        use_history = params.get("use_history", True)
        user_id = params.get("user_id", "me")

        if not message_ids and use_history:
            history = self.memory.get_recent_search(user_id)
            if history:
                message_ids = history.get("message_ids", [])

        if not message_ids:
            return {"success": False, "error": "No message IDs provided"}

        logger.info(f"Extracting action items from {len(message_ids)} emails")

        try:
            # Fetch all messages using batch_fetch_emails (correct method name)
            messages = await self.gmail_client.batch_fetch_emails(message_ids[:20])

            if not messages:
                return {"success": False, "error": "Could not retrieve any messages"}

            # Extract text content
            email_texts = []
            for msg in messages:
                content = f"Subject: {msg.get('subject', '')}\nFrom: {msg.get('from', '')}\n\n{msg.get('body', '')}"
                email_texts.append(content)

            # Extract action items using extract_actions (correct method name)
            actions = await self.extract_actions(email_texts)

            return {
                "success": True,
                "data": {
                    "action_items": actions,
                    "total_found": len(actions),
                    "emails_analyzed": len(messages),
                },
                "message": f"Found {len(actions)} action items",
            }
        except Exception as e:
            logger.error(f"Extract action items failed: {e}")
            return {"success": False, "error": f"Failed to extract: {str(e)}"}

    @capability(
        name="manage_emails",
        description="Archive, delete, star, label, or mark emails as read/unread",
        parameters=[
            ParameterSchema(
                name="action",
                type="string",
                description="Action to perform",
                required=True,
                enum=[
                    "mark_read",
                    "mark_unread",
                    "archive",
                    "delete",
                    "star",
                    "unstar",
                    "add_labels",
                    "remove_labels",
                ],
            ),
            ParameterSchema(
                name="message_ids",
                type="array",
                description="List of message IDs",
                required=False,
                default=[],
            ),
            ParameterSchema(
                name="labels",
                type="array",
                description="Label names (for add_labels/remove_labels)",
                required=False,
                default=[],
            ),
            ParameterSchema(
                name="use_history",
                type="boolean",
                description="Use recent search results if message_ids is empty",
                required=False,
                default=True,
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
        ],
    )
    async def manage_emails(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Manage emails (batch operations)."""
        action = params.get("action", "")
        message_ids = params.get("message_ids", [])
        labels = params.get("labels", [])
        use_history = params.get("use_history", True)
        user_id = params.get("user_id", "me")

        if not message_ids and use_history:
            history = self.memory.get_recent_search(user_id)
            if history:
                message_ids = history.get("message_ids", [])

        if not message_ids:
            return {"success": False, "error": "No message IDs provided"}

        # Limit batch size
        if len(message_ids) > self.config.max_batch_size:
            message_ids = message_ids[: self.config.max_batch_size]

        logger.info(f"Managing {len(message_ids)} emails with action: {action}")

        try:
            results = []
            for msg_id in message_ids:
                # Map action to tool name
                tool_mapping = {
                    "mark_read": "MARK_AS_READ",
                    "mark_unread": "MARK_AS_UNREAD",
                    "archive": "ARCHIVE_EMAIL",
                    "delete": "DELETE_EMAIL",
                    "star": "STAR_EMAIL",
                    "unstar": "UNSTAR_EMAIL",
                    "add_labels": "ADD_LABELS",
                    "remove_labels": "REMOVE_LABELS",
                }

                tool_name = tool_mapping.get(action)
                if not tool_name:
                    return {"success": False, "error": f"Unknown action: {action}"}

                # Build parameters
                tool_params = {"id": msg_id}
                if action in ["add_labels", "remove_labels"]:
                    tool_params["labels"] = labels

                result = await self.gmail_client.call_tool(tool_name, tool_params)
                results.append(
                    {"message_id": msg_id, "success": result.get("success", False)}
                )

            successful = sum(1 for r in results if r["success"])

            return {
                "success": successful == len(results),
                "data": {
                    "action": action,
                    "total_processed": len(message_ids),
                    "successful": successful,
                    "results": results,
                },
                "message": f"{action} applied to {successful}/{len(message_ids)} emails",
            }
        except Exception as e:
            logger.error(f"Manage emails failed: {e}")
            return {"success": False, "error": f"Failed to manage emails: {str(e)}"}

    # ========================================================================
    # CAPABILITIES - Attachments
    # ========================================================================

    @capability(
        name="download_attachments",
        description="Download attachments from an email",
        parameters=[
            ParameterSchema(
                name="message_id",
                type="string",
                description="Gmail message ID",
                required=True,
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
        ],
    )
    async def download_attachments(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Download attachments from an email."""
        message_id = params.get("message_id", "")
        user_id = params.get("user_id", "me")

        logger.info(f"Downloading attachments from email: {message_id}")

        try:
            # Use download_email_attachments (correct method name)
            result = await self.gmail_client.download_email_attachments(
                message_id, user_id
            )

            if result.get("success"):
                self._metrics["attachments_downloaded"] += len(result.get("files", []))
                files = result.get("files", [])
                return {
                    "success": True,
                    "data": {
                        "files": files,
                        "total_downloaded": len(files),
                        "message_id": message_id,
                    },
                    "message": f"Downloaded {len(files)} attachments",
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Download failed"),
                }
        except Exception as e:
            logger.error(f"Download attachments failed: {e}")
            return {"success": False, "error": f"Failed to download: {str(e)}"}

    # ========================================================================
    # CAPABILITIES - Smart Resolution
    # ========================================================================

    @capability(
        name="resolve_message_ids",
        description="Resolve message references to actual message IDs using smart resolver",
        parameters=[
            ParameterSchema(
                name="identifiers",
                type="array",
                description="List of identifiers to resolve (can be partial subjects, sender names, etc.)",
                required=True,
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="Gmail user ID",
                required=False,
                default="me",
            ),
        ],
    )
    async def resolve_message_ids(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Smart resolution of message identifiers."""
        identifiers = params.get("identifiers", [])
        user_id = params.get("user_id", "me")

        if not identifiers:
            return {"success": False, "error": "No identifiers provided"}

        try:
            resolved = await self.resolver.resolve_message_ids(identifiers, user_id)

            return {
                "success": True,
                "data": {
                    "identifiers": identifiers,
                    "resolved_ids": resolved,
                    "resolution_rate": len(resolved) / len(identifiers)
                    if identifiers
                    else 0,
                },
                "message": f"Resolved {len(resolved)}/{len(identifiers)} identifiers",
            }
        except Exception as e:
            logger.error(f"Resolve message IDs failed: {e}")
            return {"success": False, "error": str(e)}
