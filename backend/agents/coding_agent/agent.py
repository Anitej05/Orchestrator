"""
Coding Agent - BaseAgent Implementation

9 capabilities powered by OpenCode. Canvas decisions are made by LLM
via inference_service — no hardcoded canvas routing.
"""

import os
import time
from typing import Dict, Any, Optional

from backend.base_agent.agent import BaseAgent, AgentConfig
from backend.base_agent.types import (
    AgentRequest,
    AgentResponse,
    ExecutionContext,
    CapabilityResult,
    ParameterSchema,
)
from backend.base_agent.services import AgentServices
from backend.base_agent.capability import capability

from .opencode_client import OpenCodeClient, CodeTaskResult
from .config import (
    AGENT_ID,
    AGENT_VERSION,
    OPENCODE_SERVER_PORT,
    OPENCODE_SERVER_HOST,
    OPENCODE_SERVER_PASSWORD,
    OPENCODE_SERVER_USERNAME,
    OPENCODE_PROJECT_DIR,
    MAX_DIFF_FILES,
    MAX_DIFF_LINES_PER_FILE,
    logger,
)

# Canvas imports
try:
    from backend.services.canvas_service import CanvasService
    from services.canvas_templates import get_template_ids
    CANVAS_AVAILABLE = True
except ImportError:
    CanvasService = None
    get_template_ids = None
    CANVAS_AVAILABLE = False

# System context injected into OpenCode prompts
CANVAS_SYSTEM_CONTEXT = """
You are running inside the Orbimesh orchestrator as the Coding Agent.
Your output will be displayed in a Canvas panel on the frontend.

CANVAS CAPABILITIES:
- Code diffs: file changes shown in multi-file diff viewer with apply/reject
- Markdown: analysis and explanations rendered as rich markdown
- HTML/iframe: you can generate complete HTML pages displayed in a canvas iframe
  - Use for: previews, dashboards, interactive demos, styled documentation
  - HTML must be self-contained (inline CSS/JS, no external dependencies)
- Charts: data shown as bar/line/pie charts
- JSON: structured data in collapsible tree viewer
- Spreadsheets: tabular data in interactive grid

When generating HTML previews, create complete self-contained HTML with inline styles.
When generating docs, use rich markdown with headers, code blocks, and tables.
"""


class CodingAgent(BaseAgent):
    """
    Full-featured coding agent with 9 capabilities.
    All canvas decisions are made by LLM — no hardcoded routing.
    """

    def __init__(
        self,
        agent_id: str = AGENT_ID,
        agent_name: str = "Coding Agent",
        services: Optional[AgentServices] = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services,
            config=config or AgentConfig(
                max_retries=2,
                use_llm_recovery=False,
                request_timeout=300.0,
            ),
        )
        self._opencode: Optional[OpenCodeClient] = None
        self._metrics = {
            "tasks_completed": 0, "tasks_failed": 0,
            "files_modified_total": 0, "reviews_completed": 0,
            "tests_run": 0, "debugs": 0, "explanations": 0,
            "docs_generated": 0, "git_ops": 0, "searches": 0,
            "previews": 0,
        }

    # ================================================================
    # LIFECYCLE
    # ================================================================

    async def _initialize_resources(self):
        project_dir = OPENCODE_PROJECT_DIR or os.getcwd()
        self._opencode = OpenCodeClient(
            host=OPENCODE_SERVER_HOST, port=OPENCODE_SERVER_PORT,
            password=OPENCODE_SERVER_PASSWORD, username=OPENCODE_SERVER_USERNAME,
            project_dir=project_dir,
        )
        started = await self._opencode.start_server()
        if not started:
            raise RuntimeError(
                "Failed to start OpenCode server. "
                "Ensure npm is installed — CLI auto-installs on first use."
            )
        logger.info(f"CodingAgent v{AGENT_VERSION} initialized (LLM-driven canvas)")

    async def _cleanup_resources(self):
        if self._opencode:
            await self._opencode.stop_server()
            self._opencode = None

    def _get_custom_metrics(self) -> Dict[str, Any]:
        return {**self._metrics, "opencode_running": self._opencode.is_running if self._opencode else False}

    # ================================================================
    # CANVAS — LLM-driven via shared CanvasService.decide_canvas_llm()
    # ================================================================

    async def _decide_canvas(
        self, result: CodeTaskResult, capability_name: str
    ) -> Dict[str, Any]:
        """
        Delegate canvas decision to the shared CanvasService LLM.
        Returns a dict ready for AgentResponse.canvas_display.
        """
        # Prepare file change dicts
        file_change_dicts = []
        for change in result.file_changes[:MAX_DIFF_FILES]:
            diff_text = change.diff
            lines = diff_text.split("\n")
            if len(lines) > MAX_DIFF_LINES_PER_FILE:
                diff_text = "\n".join(lines[:MAX_DIFF_LINES_PER_FILE])
                diff_text += f"\n... ({len(lines) - MAX_DIFF_LINES_PER_FILE} more lines)"
            file_change_dicts.append({
                "file": change.file_path, "diff": diff_text,
                "language": change.language, "status": change.status,
            })

        if CANVAS_AVAILABLE:
            try:
                display = await CanvasService.decide_canvas_llm(
                    output=result.output,
                    agent_name="coding_agent",
                    capability_name=capability_name,
                    file_changes=file_change_dicts,
                    files_modified=result.files_modified,
                )
                if display:
                    return display.model_dump() if hasattr(display, "model_dump") else display.dict()
            except Exception as e:
                logger.warning(f"LLM canvas decision failed: {e}")

        # Fallback
        return {"canvas_type": "markdown", "canvas_title": capability_name, "canvas_content": result.output or ""}

    def build_dynamic_canvas(
        self, template_id: str, data: Dict[str, Any],
        title: Optional[str] = None, requires_confirmation: bool = False,
        confirmation_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Public API: build any canvas from a registered template."""
        if not CANVAS_AVAILABLE:
            return None
        try:
            display = CanvasService.build_from_template(
                template_id=template_id, data=data, title=title,
                requires_confirmation=requires_confirmation,
                confirmation_message=confirmation_message,
            )
            if display:
                return display.model_dump() if hasattr(display, "model_dump") else display.dict()
        except Exception as e:
            logger.warning(f"Dynamic canvas '{template_id}' failed: {e}")
        return None

    # ================================================================
    # CAPABILITIES (9 total) — all use _decide_canvas()
    # ================================================================

    @capability(
        name="code_task",
        description="Write, edit, refactor, or debug code. Modifies files. Returns diffs for approval.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="Coding instruction", required=True),
            ParameterSchema(name="project_dir", type="string", description="Project directory", required=False),
        ],
    )
    async def code_task(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = self._extract_prompt(params)
        if not prompt:
            return CapabilityResult.fail("No coding prompt provided")
        try:
            result = await self._opencode.send_prompt(CANVAS_SYSTEM_CONTEXT + prompt)
            if not result.success:
                self._metrics["tasks_failed"] += 1
                return CapabilityResult.fail(result.error or "Coding task failed")
            self._metrics["tasks_completed"] += 1
            self._metrics["files_modified_total"] += len(result.files_modified)
            canvas = await self._decide_canvas(result, "code_task")
            return CapabilityResult.ok(
                data={"files_modified": result.files_modified, "file_count": len(result.file_changes),
                      "summary": result.summary, "output": result.output},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            self._metrics["tasks_failed"] += 1
            return CapabilityResult.fail(str(e))

    @capability(
        name="review_code",
        description="Read-only code analysis. Does NOT modify files.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="Review question", required=True),
            ParameterSchema(name="file_path", type="string", description="Specific file to review", required=False),
        ],
    )
    async def review_code(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = self._extract_prompt(params)
        if not prompt:
            return CapabilityResult.fail("No review prompt provided")
        file_path = params.get("file_path")
        review_prompt = (
            "IMPORTANT: READ-ONLY review. Do NOT modify files.\n\n"
            f"{prompt}" + (f"\n\nFocus on: {file_path}" if file_path else "")
        )
        try:
            result = await self._opencode.send_prompt(CANVAS_SYSTEM_CONTEXT + review_prompt)
            self._metrics["reviews_completed"] += 1
            canvas = await self._decide_canvas(result, "review_code")
            return CapabilityResult.ok(
                data={"analysis": result.output, "summary": result.summary, "file_path": file_path},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            return CapabilityResult.fail(str(e))

    @capability(
        name="run_tests",
        description="Run the project test suite. Reports pass/fail with terminal output.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="Test instruction", required=False),
            ParameterSchema(name="test_command", type="string", description="e.g. 'pytest tests/'", required=False),
        ],
    )
    async def run_tests(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = params.get("prompt") or params.get("test_command") or "Run all project tests"
        try:
            result = await self._opencode.send_prompt(f"Run tests and report results.\n\n{prompt}")
            self._metrics["tests_run"] += 1
            canvas = await self._decide_canvas(result, "run_tests")
            return CapabilityResult.ok(
                data={"summary": result.summary, "terminal_log": result.terminal_log, "output": result.output},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            return CapabilityResult.fail(str(e))

    @capability(
        name="debug",
        description="Debug an error: analyze tracebacks, identify root cause, suggest or apply fixes.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="Error description or traceback", required=True),
            ParameterSchema(name="traceback", type="string", description="Full traceback text", required=False),
            ParameterSchema(name="auto_fix", type="boolean", description="Apply fix automatically", required=False),
        ],
    )
    async def debug(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = self._extract_prompt(params)
        traceback_text = params.get("traceback", "")
        auto_fix = params.get("auto_fix", False)
        mode = "Fix the bug" if auto_fix else "Analyze but do NOT modify files"
        debug_prompt = (
            f"{CANVAS_SYSTEM_CONTEXT}\n{mode}.\n\nError: {prompt}\n"
            + (f"\nTraceback:\n```\n{traceback_text}\n```" if traceback_text else "")
        )
        try:
            result = await self._opencode.send_prompt(debug_prompt)
            self._metrics["debugs"] += 1
            canvas = await self._decide_canvas(result, "debug")
            return CapabilityResult.ok(
                data={"analysis": result.output, "summary": result.summary,
                      "files_modified": result.files_modified, "auto_fix": auto_fix},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            return CapabilityResult.fail(str(e))

    @capability(
        name="explain_code",
        description="Explain a file, function, class, or concept. Rich markdown output.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="What to explain", required=True),
            ParameterSchema(name="file_path", type="string", description="File to explain", required=False),
        ],
    )
    async def explain_code(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = self._extract_prompt(params)
        file_path = params.get("file_path")
        explain_prompt = (
            f"{CANVAS_SYSTEM_CONTEXT}\nExplain in detail with markdown, code blocks, examples. "
            "Do NOT modify files.\n\n"
            f"{prompt}" + (f"\n\nFile: {file_path}" if file_path else "")
        )
        try:
            result = await self._opencode.send_prompt(explain_prompt)
            self._metrics["explanations"] += 1
            canvas = await self._decide_canvas(result, "explain_code")
            return CapabilityResult.ok(
                data={"explanation": result.output, "summary": result.summary},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            return CapabilityResult.fail(str(e))

    @capability(
        name="generate_docs",
        description="Generate documentation: README, API docs, docstrings. Can render as HTML preview.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="Documentation instruction", required=True),
            ParameterSchema(name="format", type="string", description="'markdown' or 'html'", required=False),
            ParameterSchema(name="file_path", type="string", description="File to document", required=False),
        ],
    )
    async def generate_docs(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = self._extract_prompt(params)
        fmt = params.get("format", "markdown")
        file_path = params.get("file_path")
        if fmt == "html":
            doc_prompt = (
                f"{CANVAS_SYSTEM_CONTEXT}\nGenerate COMPLETE self-contained HTML documentation "
                "with inline CSS. Visually polished, dark theme. Output ONLY HTML. Do NOT modify files.\n\n"
                f"{prompt}" + (f"\n\nFor: {file_path}" if file_path else "")
            )
        else:
            doc_prompt = (
                f"{CANVAS_SYSTEM_CONTEXT}\nGenerate comprehensive markdown docs with headers, "
                "code blocks, tables. Do NOT modify files.\n\n"
                f"{prompt}" + (f"\n\nFor: {file_path}" if file_path else "")
            )
        try:
            result = await self._opencode.send_prompt(doc_prompt)
            self._metrics["docs_generated"] += 1
            canvas = await self._decide_canvas(result, "generate_docs")
            return CapabilityResult.ok(
                data={"documentation": result.output, "format": fmt, "summary": result.summary},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            return CapabilityResult.fail(str(e))

    @capability(
        name="git_operations",
        description="Git info: status, diff, log, branches. Read-only version control.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="Git operation", required=True),
        ],
    )
    async def git_operations(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = self._extract_prompt(params)
        git_prompt = (
            f"{CANVAS_SYSTEM_CONTEXT}\nExecute the git operation. "
            "Do NOT create commits unless explicitly asked.\n\n{prompt}"
        )
        try:
            result = await self._opencode.send_prompt(git_prompt)
            self._metrics["git_ops"] += 1
            canvas = await self._decide_canvas(result, "git_operations")
            return CapabilityResult.ok(
                data={"output": result.output, "summary": result.summary},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            return CapabilityResult.fail(str(e))

    @capability(
        name="search_codebase",
        description="Search for patterns, functions, classes, or files across the project.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="Search query", required=True),
            ParameterSchema(name="pattern", type="string", description="Regex or glob pattern", required=False),
        ],
    )
    async def search_codebase(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = self._extract_prompt(params)
        pattern = params.get("pattern")
        search_prompt = (
            f"{CANVAS_SYSTEM_CONTEXT}\nSearch the codebase. Show file paths, line numbers, "
            "matching code. Do NOT modify files.\n\n"
            f"Search: {prompt}" + (f"\nPattern: {pattern}" if pattern else "")
        )
        try:
            result = await self._opencode.send_prompt(search_prompt)
            self._metrics["searches"] += 1
            canvas = await self._decide_canvas(result, "search_codebase")
            return CapabilityResult.ok(
                data={"results": result.output, "summary": result.summary},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            return CapabilityResult.fail(str(e))

    @capability(
        name="generate_preview",
        description="Generate HTML/React/CSS and display as live iframe in canvas.",
        parameters=[
            ParameterSchema(name="prompt", type="string", description="What to preview", required=True),
            ParameterSchema(name="framework", type="string", description="'html', 'react', 'vue'", required=False),
        ],
    )
    async def generate_preview(self, params: Dict[str, Any], context: ExecutionContext) -> CapabilityResult:
        prompt = self._extract_prompt(params)
        preview_prompt = (
            f"{CANVAS_SYSTEM_CONTEXT}\n"
            "Generate a COMPLETE self-contained HTML page for iframe display.\n"
            "- All CSS inline or in <style> tags\n"
            "- All JS inline\n"
            "- Modern design, dark mode, smooth gradients\n"
            "- Output ONLY raw HTML — no markdown wrapping\n"
            "- Do NOT modify project files\n\n"
            f"Create: {prompt}"
        )
        try:
            result = await self._opencode.send_prompt(preview_prompt)
            self._metrics["previews"] += 1
            canvas = await self._decide_canvas(result, "generate_preview")
            return CapabilityResult.ok(
                data={"html": result.output, "summary": result.summary},
                metadata={"canvas_display": canvas},
            )
        except Exception as e:
            return CapabilityResult.fail(str(e))

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def _extract_prompt(params: Dict[str, Any]) -> str:
        for key in ("prompt", "query", "instruction", "text", "task"):
            val = params.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    # ================================================================
    # EXECUTE OVERRIDE
    # ================================================================

    async def execute(self, request: AgentRequest) -> AgentResponse:
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        action = request.action or ""

        try:
            context = ExecutionContext(
                thread_id=request.thread_id or "default",
                user_id=request.user_id or "anonymous",
                task_id=request.task_id,
            )

            action_map = {
                "review_code": self.review_code, "review": self.review_code,
                "run_tests": self.run_tests, "test": self.run_tests,
                "debug": self.debug, "fix": self.debug,
                "explain_code": self.explain_code, "explain": self.explain_code,
                "generate_docs": self.generate_docs, "docs": self.generate_docs,
                "git_operations": self.git_operations, "git": self.git_operations,
                "search_codebase": self.search_codebase, "search": self.search_codebase,
                "generate_preview": self.generate_preview, "preview": self.generate_preview,
            }

            handler = action_map.get(action)
            if handler:
                cap_result = await handler(request.payload, context)
            else:
                params = dict(request.payload)
                if request.prompt and "prompt" not in params:
                    params["prompt"] = request.prompt
                cap_result = await self.code_task(params, context)

            elapsed_ms = (time.time() - start_time) * 1000

            if cap_result.success:
                summary = cap_result.data.get("summary", "Done") if isinstance(cap_result.data, dict) else "Done"
                return AgentResponse.success(
                    result=cap_result.data, summary=summary,
                    canvas_display=cap_result.metadata.get("canvas_display"),
                    execution_time_ms=elapsed_ms,
                    capabilities_used=[action or "code_task"],
                )
            else:
                return AgentResponse.error(
                    message=cap_result.error or "Task failed",
                    execution_time_ms=elapsed_ms,
                    capabilities_used=[action or "code_task"],
                )
        except Exception as e:
            return AgentResponse.error(
                message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
