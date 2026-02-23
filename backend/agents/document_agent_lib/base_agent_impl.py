"""
Document Agent v2.0 - Complete BaseAgent Implementation

Full document analysis and management with:
- Document analysis with RAG
- Document creation (DOCX, PDF)
- Document editing with version control
- Data extraction
- Canvas display
- Undo/redo support
"""

import logging
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

from backend.agents.base import BaseAgent, AgentServices, AgentConfig
from backend.agents.base.types import AgentRequest, AgentResponse, ExecutionContext
from backend.agents.base.capability import capability, ParameterSchema

from .agent_schemas import EditAction
from .editors import DocumentEditor
from .state import DocumentSessionManager, DocumentVersionManager
from .llm import DocumentLLMClient
from .utils import (
    extract_document_content,
    create_docx,
    create_pdf,
    analyze_document_structure,
)

logger = logging.getLogger("agents.document_agent")

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DEFAULT_STORAGE_DIR = WORKSPACE_ROOT / "storage" / "document_agent"


@dataclass
class DocumentAgentConfig(AgentConfig):
    """Configuration for Document Agent."""

    max_file_size_mb: int = 50
    max_cache_size: int = 100
    enable_rag: bool = True
    enable_versioning: bool = True
    max_versions_per_doc: int = 20
    risky_action_threshold: float = 0.6


class DocumentAgent(BaseAgent):
    """
    Complete document analysis and management agent.

    Features:
    - Document analysis with RAG-based question answering
    - Document creation (DOCX, PDF)
    - Document editing with natural language
    - Version management and undo/redo
    - Data extraction from documents
    - Canvas display for frontend
    """

    def __init__(
        self,
        agent_id: str = "document_agent",
        agent_name: str = "Document Agent",
        services: Optional[AgentServices] = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services,
            config=config or DocumentAgentConfig(),
        )

        # Components
        self.session_manager = DocumentSessionManager()
        self.version_manager = DocumentVersionManager()
        self.llm_client = DocumentLLMClient()

        # Risk assessment
        self._risky_action_types = {
            "delete_content",
            "replace_content",
            "convert_format",
        }
        self._risk_keywords = [
            "delete",
            "remove",
            "overwrite",
            "purge",
            "wipe",
            "truncate",
        ]
        self._max_safe_actions = 25

    async def _initialize_resources(self):
        """Initialize document management components."""
        logger.info("Initializing Document Agent resources...")
        DEFAULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Document Agent resources initialized")

    async def _cleanup_resources(self):
        """Cleanup resources."""
        logger.info("Cleaning up Document Agent resources...")

    async def _get_custom_metrics(self) -> Optional[Dict[str, Any]]:
        """Return document agent metrics."""
        return {
            "sessions": len(self.session_manager._sessions)
            if hasattr(self.session_manager, "_sessions")
            else 0,
            "storage_dir": str(DEFAULT_STORAGE_DIR),
        }

    def _classify_edit_intent(self, instruction: str) -> Dict[str, Any]:
        """Assess risk of edit instruction."""
        text = (instruction or "").lower()
        risk_hits = [kw for kw in self._risk_keywords if kw in text]

        if any(kw in text for kw in ["delete", "remove", "purge", "wipe", "truncate"]):
            intent = "destructive"
        elif any(kw in text for kw in ["replace", "rewrite", "overwrite"]):
            intent = "overwrite"
        else:
            intent = "edit"

        base = 0.25 if intent == "edit" else 0.35
        per = 0.05 if intent == "edit" else 0.15
        score = min(1.0, base + per * len(risk_hits))

        return {
            "intent": intent,
            "risk_score": round(score, 2),
            "risk_signals": risk_hits,
        }

    def _validate_edit_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate edit plan against allowed actions."""
        actions = plan.get("actions") or []
        issues: List[str] = []
        normalized: List[Dict[str, Any]] = []

        if not isinstance(actions, list):
            return {
                "valid": False,
                "issues": ["Plan actions must be a list"],
                "actions": [],
            }

        if len(actions) > self._max_safe_actions:
            issues.append(
                f"Plan proposes {len(actions)} actions (> {self._max_safe_actions})"
            )

        allowed = {
            "add_paragraph",
            "add_heading",
            "format_text",
            "replace_text",
            "add_table",
            "add_content",
            "replace_content",
            "delete_content",
            "add_image",
            "modify_style",
            "convert_format",
        }

        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                issues.append(f"Action {idx + 1} must be an object")
                continue
            a_type = str(action.get("type", "")).lower().strip()
            if not a_type:
                issues.append(f"Action {idx + 1} missing type")
                continue
            if a_type not in allowed:
                issues.append(f"Unsupported action type: {a_type}")
                continue
            normalized.append(
                {"type": a_type, **{k: v for k, v in action.items() if k != "type"}}
            )

        return {"valid": len(issues) == 0, "issues": issues, "actions": normalized}

    def _execute_edit_action(
        self, editor: DocumentEditor, action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single edit action."""
        action_type = action.get("type", "").lower()

        try:
            if action_type == "add_paragraph":
                result = editor.add_paragraph(
                    action.get("text", ""), action.get("style", "Normal")
                )
            elif action_type == "add_heading":
                result = editor.add_heading(
                    action.get("text", ""), action.get("level", 1)
                )
            elif action_type == "format_text":
                result = editor.format_text(
                    action.get("text", ""), **action.get("options", {})
                )
            elif action_type == "replace_text":
                result = editor.replace_text(
                    action.get("old_text", ""), action.get("new_text", "")
                )
            elif action_type == "add_table":
                result = editor.add_table(action.get("rows", 2), action.get("cols", 2))
            else:
                result = f"Unknown action type: {action_type}"

            return {"type": action_type, "result": result, "success": "✓" in result}

        except Exception as e:
            return {"type": action_type, "result": f"Error: {str(e)}", "success": False}

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    @capability(
        name="analyze_document",
        description="Analyze a document and answer questions using RAG",
        parameters=[
            ParameterSchema(
                name="file_path",
                type="string",
                description="Path to the document file",
                required=True,
            ),
            ParameterSchema(
                name="query",
                type="string",
                description="Question or query about the document",
                required=True,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def analyze_document(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Analyze document with RAG-based question answering."""
        file_path = params.get("file_path")
        query = params.get("query")
        thread_id = params.get("thread_id", "default")

        if not file_path or not query:
            return {"success": False, "error": "file_path and query are required"}

        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                alt_path = WORKSPACE_ROOT / file_path
                if alt_path.exists():
                    path_obj = alt_path
                else:
                    return {"success": False, "error": f"File not found: {file_path}"}

            # Extract content
            content, _ = extract_document_content(str(path_obj))

            # Use LLM to answer
            answer = await self.llm_client.analyze_document_with_query(content, query)

            return {
                "success": True,
                "data": {"answer": answer, "file_path": str(path_obj), "query": query},
                "message": "Document analyzed successfully",
            }

        except Exception as e:
            logger.error(f"Analyze document failed: {e}")
            return {"success": False, "error": f"Analysis failed: {str(e)}"}

    @capability(
        name="create_document",
        description="Create a new document (DOCX or PDF)",
        parameters=[
            ParameterSchema(
                name="content",
                type="string",
                description="Document content text",
                required=True,
            ),
            ParameterSchema(
                name="filename",
                type="string",
                description="Output filename (should end with .docx or .pdf)",
                required=True,
            ),
            ParameterSchema(
                name="output_dir",
                type="string",
                description="Output directory",
                required=False,
                default=str(DEFAULT_STORAGE_DIR),
            ),
            ParameterSchema(
                name="title",
                type="string",
                description="Document title",
                required=False,
            ),
        ],
    )
    async def create_document(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Create a new document."""
        content = params.get("content", "")
        filename = params.get("filename", "document.docx")
        output_dir = params.get("output_dir", str(DEFAULT_STORAGE_DIR))
        title = params.get("title", "")

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            file_path = output_path / filename

            ext = Path(filename).suffix.lower()

            if ext == ".docx":
                # Prepend title to content if provided
                if title:
                    content = f"{title}\n\n{content}"
                create_docx(content, str(file_path))
            elif ext == ".pdf":
                # Prepend title to content if provided
                if title:
                    content = f"{title}\n\n{content}"
                create_pdf(content, str(file_path))
            else:
                with open(file_path, "w") as f:
                    if title:
                        f.write(f"{title}\n\n")
                    f.write(content)

            # Create initial version
            self.version_manager.save_version(str(file_path), "Initial creation")

            return {
                "success": True,
                "data": {
                    "file_path": str(file_path),
                    "filename": filename,
                    "file_type": ext,
                },
                "message": f"Created {filename}",
            }

        except Exception as e:
            logger.error(f"Create document failed: {e}")
            return {"success": False, "error": f"Creation failed: {str(e)}"}

    @capability(
        name="edit_document",
        description="Edit a document using natural language instructions",
        parameters=[
            ParameterSchema(
                name="file_path",
                type="string",
                description="Path to the document to edit",
                required=True,
            ),
            ParameterSchema(
                name="instruction",
                type="string",
                description="Edit instruction (e.g., 'Add a conclusion paragraph')",
                required=True,
            ),
            ParameterSchema(
                name="auto_approve",
                type="boolean",
                description="Auto-approve edits without confirmation",
                required=False,
                default=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def edit_document(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Edit document using natural language."""
        file_path = params.get("file_path")
        instruction = params.get("instruction")
        auto_approve = params.get("auto_approve", False)
        thread_id = params.get("thread_id", "default")

        if not file_path or not instruction:
            return {"success": False, "error": "file_path and instruction are required"}

        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            # Risk assessment
            risk = self._classify_edit_intent(instruction)

            # Get session
            session = self.session_manager.get_or_create_session(
                file_path, path_obj.name, thread_id
            )

            # Extract content and structure
            content, _ = extract_document_content(file_path)
            structure = analyze_document_structure(file_path)

            # Plan edits using LLM
            plan = await self.llm_client.interpret_edit_instruction(
                instruction, content, structure
            )

            if not plan.get("success"):
                return {
                    "success": False,
                    "error": f"Failed to plan edits: {plan.get('error', 'Unknown error')}",
                }

            # Validate plan
            validation = self._validate_edit_plan(plan)

            # Check risk
            if (
                risk.get("risk_score", 0) >= 0.6
                or risk.get("intent") in {"destructive", "overwrite"}
            ) and not auto_approve:
                return {
                    "success": False,
                    "error": "Approval required",
                    "data": {
                        "question": f"Approve edit plan with {len(validation.get('actions', []))} actions?",
                        "question_type": "confirmation",
                        "risk_assessment": risk,
                        "pending_plan": validation,
                    },
                }

            # Execute edits
            editor = DocumentEditor(file_path)
            results = []

            for action in validation.get("actions", []):
                result = self._execute_edit_action(editor, action)
                results.append(result)

            # Save and create version
            editor.save()
            self.version_manager.save_version(file_path, f"Edit: {instruction[:50]}")

            return {
                "success": True,
                "data": {
                    "file_path": file_path,
                    "actions_executed": len(results),
                    "results": results,
                },
                "message": f"Applied {len(results)} edits",
            }

        except Exception as e:
            logger.error(f"Edit document failed: {e}")
            return {"success": False, "error": f"Edit failed: {str(e)}"}

    @capability(
        name="get_versions",
        description="Get version history of a document",
        parameters=[
            ParameterSchema(
                name="file_path",
                type="string",
                description="Path to the document",
                required=True,
            )
        ],
    )
    async def get_versions(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Get document version history."""
        file_path = params.get("file_path")

        if not file_path:
            return {"success": False, "error": "file_path is required"}

        try:
            versions = self.version_manager.get_versions(file_path)

            return {
                "success": True,
                "data": {
                    "file_path": file_path,
                    "versions": versions,
                    "version_count": len(versions),
                },
                "message": f"Found {len(versions)} versions",
            }

        except Exception as e:
            logger.error(f"Get versions failed: {e}")
            return {"success": False, "error": f"Failed to get versions: {str(e)}"}

    @capability(
        name="undo_redo",
        description="Undo or redo document edits",
        parameters=[
            ParameterSchema(
                name="file_path",
                type="string",
                description="Path to the document",
                required=True,
            ),
            ParameterSchema(
                name="action",
                type="string",
                description="Action to perform: 'undo' or 'redo'",
                required=True,
                enum=["undo", "redo"],
            ),
        ],
    )
    async def undo_redo(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Undo or redo document changes."""
        file_path = params.get("file_path")
        action = params.get("action", "undo")

        if not file_path:
            return {"success": False, "error": "file_path is required"}

        try:
            versions = self.version_manager.get_versions(file_path)

            if not versions:
                return {"success": False, "error": "No versions available"}

            # Get current version index
            doc_key = self.version_manager._get_document_key(file_path)
            current_idx = self.version_manager.index.get(doc_key, {}).get(
                "current_version", -1
            )

            if action == "undo":
                if current_idx > 0:
                    target_version = versions[current_idx - 1]["version_id"]
                    success = self.version_manager.restore_version(
                        file_path, target_version
                    )
                    return {
                        "success": success,
                        "data": {"action": "undo", "restored_to": target_version},
                        "message": "Undo successful" if success else "Undo failed",
                    }
                else:
                    return {"success": False, "error": "Nothing to undo"}
            else:  # redo
                if current_idx < len(versions) - 1:
                    target_version = versions[current_idx + 1]["version_id"]
                    success = self.version_manager.restore_version(
                        file_path, target_version
                    )
                    return {
                        "success": success,
                        "data": {"action": "redo", "restored_to": target_version},
                        "message": "Redo successful" if success else "Redo failed",
                    }
                else:
                    return {"success": False, "error": "Nothing to redo"}

        except Exception as e:
            logger.error(f"Undo/redo failed: {e}")
            return {"success": False, "error": f"Failed: {str(e)}"}

    @capability(
        name="extract_data",
        description="Extract structured data from a document",
        parameters=[
            ParameterSchema(
                name="file_path",
                type="string",
                description="Path to the document",
                required=True,
            ),
            ParameterSchema(
                name="extraction_type",
                type="string",
                description="Type of data to extract",
                required=True,
                enum=["entities", "tables", "key_value", "summary", "all"],
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def extract_data(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Extract structured data from document."""
        file_path = params.get("file_path")
        extraction_type = params.get("extraction_type")
        thread_id = params.get("thread_id", "default")

        if not file_path or not extraction_type:
            return {
                "success": False,
                "error": "file_path and extraction_type are required",
            }

        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            # Extract content
            content, metadata = extract_document_content(file_path)

            # Extract based on type
            if extraction_type == "tables":
                extracted = metadata.get("tables", [])
            else:
                # Use extract_structured_data for all other types
                extraction_params = {
                    "summary": {"extract_summary": True},
                    "entities": {"extract_entities": True},
                    "key_value": {"extract_key_values": True},
                    "all": {
                        "extract_summary": True,
                        "extract_entities": True,
                        "extract_key_values": True,
                    },
                }
                params = extraction_params.get(
                    extraction_type, extraction_params["all"]
                )
                extracted = await self.llm_client.extract_structured_data(
                    content, **params
                )

            return {
                "success": True,
                "data": {
                    "extracted_data": extracted,
                    "extraction_type": extraction_type,
                    "file_path": file_path,
                },
                "message": "Data extracted successfully",
            }

        except Exception as e:
            logger.error(f"Extract data failed: {e}")
            return {"success": False, "error": f"Extraction failed: {str(e)}"}

    @capability(
        name="display_document",
        description="Display a document with canvas rendering",
        parameters=[
            ParameterSchema(
                name="file_path",
                type="string",
                description="Path to the document",
                required=True,
            )
        ],
    )
    async def display_document(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Display document with canvas rendering."""
        file_path = params.get("file_path")

        if not file_path:
            return {"success": False, "error": "file_path is required"}

        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            # Extract content for display
            content, metadata = extract_document_content(file_path)

            # Build canvas display
            ext = path_obj.suffix.lower()

            canvas_display = {
                "canvas_type": "document",
                "canvas_title": path_obj.name,
                "canvas_data": {
                    "content_preview": content[:2000] if content else "",
                    "file_type": ext,
                    "file_path": str(path_obj),
                    "metadata": metadata,
                },
            }

            return {
                "success": True,
                "data": {
                    "file_path": str(path_obj),
                    "file_type": ext,
                    "content_length": len(content) if content else 0,
                },
                "canvas_display": canvas_display,
                "message": f"Displaying {path_obj.name}",
            }

        except Exception as e:
            logger.error(f"Display document failed: {e}")
            return {"success": False, "error": f"Display failed: {str(e)}"}

    @capability(
        name="compare_documents",
        description="Compare two documents and show differences",
        parameters=[
            ParameterSchema(
                name="file_path_1",
                type="string",
                description="Path to first document",
                required=True,
            ),
            ParameterSchema(
                name="file_path_2",
                type="string",
                description="Path to second document",
                required=True,
            ),
        ],
    )
    async def compare_documents(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Compare two documents."""
        file_path_1 = params.get("file_path_1")
        file_path_2 = params.get("file_path_2")

        if not file_path_1 or not file_path_2:
            return {"success": False, "error": "Both file paths are required"}

        try:
            # Extract content from both documents
            content_1, _ = extract_document_content(file_path_1)
            content_2, _ = extract_document_content(file_path_2)

            # Use LLM to compare by analyzing both documents
            combined_content = (
                f"DOCUMENT 1:\n{content_1[:5000]}\n\nDOCUMENT 2:\n{content_2[:5000]}"
            )
            comparison_query = "Compare these two documents and highlight key differences, similarities, and unique content in each."
            comparison = await self.llm_client.analyze_document_with_query(
                combined_content, comparison_query
            )

            return {
                "success": True,
                "data": {
                    "file_1": file_path_1,
                    "file_2": file_path_2,
                    "comparison": comparison,
                },
                "message": "Documents compared successfully",
            }

        except Exception as e:
            logger.error(f"Compare documents failed: {e}")
            return {"success": False, "error": f"Comparison failed: {str(e)}"}
