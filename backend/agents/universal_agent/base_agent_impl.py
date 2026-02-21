"""
Universal Agent - General Purpose Task Executor

A flexible agent capable of handling any arbitrary task through
LLM reasoning, code execution, tool usage, and full file system access.
"""

import logging
import json
import os
import glob
import shutil
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from backend.agents.base import BaseAgent, AgentRequest, AgentResponse, capability
from backend.agents.utils.agent_file_manager import AgentFileManager, FileType, FileStatus
from services.canvas_service import CanvasService

logger = logging.getLogger(__name__)


class UniversalAgent(BaseAgent):
    """
    Universal Agent - handles any arbitrary task not covered by specialized agents.

    Capabilities:
    - General task execution with planning
    - Code generation and execution
    - Analysis and research
    - Creative writing
    - Problem solving
    """

    def __init__(
        self,
        agent_id="universal_agent",
        agent_name="Universal Agent",
        services=None,
        config=None,
    ):
        super().__init__(
            agent_id=agent_id, agent_name=agent_name, services=services, config=config
        )
        self.description = "General-purpose agent for arbitrary tasks"

    async def _initialize_resources(self):
        """Initialize agent-specific resources."""
        logger.info("Initializing Universal Agent resources")
        # Initialize file manager for file operations
        self.file_manager = AgentFileManager(
            agent_id="universal_agent",
            storage_dir="storage/universal_agent",
        )
        logger.info("Universal Agent file manager initialized")

    @capability(
        name="execute_task",
        description="Execute any arbitrary task through planning and execution",
    )
    async def execute_task(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Main entry point for task execution."""
        prompt = params.get("prompt") or params.get("query") or params.get("task") or str(params)
        request = AgentRequest(prompt=prompt, payload=params)
        return await self._execute_general_task(request)

    @capability(
        name="analyze",
        description="Analyze data, text, or situations and provide insights",
    )
    async def analyze(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Analyze content and provide insights."""
        prompt = params.get("prompt") or params.get("query") or params.get("content") or str(params)
        request = AgentRequest(prompt=prompt, payload=params)
        return await self._analyze_content(request)

    @capability(
        name="generate_code",
        description="Generate and optionally execute code to solve problems",
    )
    async def generate_code(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Generate code for the given task."""
        prompt = params.get("prompt") or params.get("requirement") or params.get("task") or str(params)
        request = AgentRequest(prompt=prompt, payload=params)
        return await self._generate_code(request)

    @capability(
        name="research",
        description="Research topics and compile comprehensive information",
    )
    async def research(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Research a topic and compile information."""
        prompt = params.get("prompt") or params.get("query") or params.get("topic") or str(params)
        request = AgentRequest(prompt=prompt, payload=params)
        return await self._research_topic(request)

    @capability(
        name="creative_write",
        description="Create creative content like stories, poems, scripts",
    )
    async def creative_write(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Generate creative writing."""
        prompt = params.get("prompt") or params.get("topic") or params.get("description") or str(params)
        request = AgentRequest(prompt=prompt, payload=params)
        return await self._creative_writing(request)

    @capability(
        name="solve_problem",
        description="Break down and solve complex problems systematically",
    )
    async def solve_problem(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Solve a complex problem step by step."""
        prompt = params.get("prompt") or params.get("problem") or params.get("question") or str(params)
        request = AgentRequest(prompt=prompt, payload=params)
        return await self._solve_problem(request)

    # =========================================================================
    # FILE SYSTEM CAPABILITIES
    # =========================================================================

    @capability(
        name="read_file",
        description="Read any file from the local filesystem by path",
    )
    async def read_file(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Read a file from the filesystem."""
        file_path = params.get("prompt") or params.get("file_path") or params.get("path")
        if not file_path:
            return {"success": False, "error": "No file path provided"}
        return await self._read_file(file_path)

    @capability(
        name="write_file",
        description="Write or create files at any path on the filesystem",
    )
    async def write_file(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Write content to a file."""
        file_path = params.get("file_path") or params.get("path")
        content = params.get("content") or params.get("data") or params.get("prompt", "")
        if not file_path:
            return {"success": False, "error": "No file path provided"}
        return await self._write_file(file_path, content)

    @capability(
        name="list_directory",
        description="List contents of any directory with optional filtering",
    )
    async def list_directory(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """List directory contents."""
        dir_path = params.get("prompt") or params.get("path") or params.get("directory") or "."
        pattern = params.get("pattern") or params.get("filter")
        return await self._list_directory(dir_path, pattern)

    @capability(
        name="search_files",
        description="Search for files by name pattern across directories",
    )
    async def search_files(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Search for files matching a pattern."""
        pattern = params.get("prompt") or params.get("pattern") or params.get("query") or "*"
        directory = params.get("directory") or params.get("path") or "."
        return await self._search_files(pattern, directory)

    @capability(
        name="manage_file",
        description="Delete, move, copy, or get info about files",
    )
    async def manage_file(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Manage files: delete, move, copy, info."""
        operation = params.get("operation") or params.get("action") or "info"
        source = params.get("prompt") or params.get("source") or params.get("path")
        destination = params.get("destination") or params.get("target")
        if not source:
            return {"success": False, "error": "No source path provided"}
        return await self._manage_file(operation, source, destination)

    # =========================================================================
    # FILE IMPLEMENTATION METHODS
    # =========================================================================

    async def _read_file(self, file_path: str) -> Dict[str, Any]:
        """Read file content from filesystem."""
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            if not path.is_file():
                return {"success": False, "error": f"Not a file: {file_path}"}

            size = path.stat().st_size
            # Safety limit: 10MB
            if size > 10 * 1024 * 1024:
                return {
                    "success": False,
                    "error": f"File too large ({size / (1024*1024):.1f}MB). Max 10MB.",
                }

            # Try text first, fall back to binary
            try:
                content = path.read_text(encoding="utf-8")
                is_binary = False
            except UnicodeDecodeError:
                content = base64.b64encode(path.read_bytes()).decode("ascii")
                is_binary = True

            result = {
                "success": True,
                "result": content if not is_binary else f"[Binary file, {size} bytes, base64 encoded]",
                "data": {
                    "content": content,
                    "file_path": str(path),
                    "file_name": path.name,
                    "size_bytes": size,
                    "is_binary": is_binary,
                    "extension": path.suffix,
                },
            }

            # Auto-detect canvas type for text files
            if not is_binary:
                ext = path.suffix.lower()
                try:
                    if ext == '.csv':
                        # Parse CSV into spreadsheet canvas
                        import csv
                        import io
                        reader = csv.reader(io.StringIO(content))
                        rows_list = list(reader)
                        if rows_list:
                            canvas = CanvasService.build_from_template(
                                "spreadsheet_viewer",
                                {"headers": rows_list[0], "rows": rows_list[1:], "filename": path.name},
                                title=path.name,
                            )
                            if canvas:
                                result["canvas_display"] = canvas.model_dump()
                    elif ext in ('.py', '.js', '.ts', '.java', '.c', '.cpp', '.go', '.rs', '.sh', '.rb', '.php'):
                        lang_map = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.java': 'java',
                                    '.c': 'c', '.cpp': 'cpp', '.go': 'go', '.rs': 'rust', '.sh': 'bash', '.rb': 'ruby', '.php': 'php'}
                        canvas = CanvasService.build_from_template(
                            "code_viewer",
                            {"code": content, "language": lang_map.get(ext, 'text'), "filename": path.name},
                            title=path.name,
                        )
                        if canvas:
                            result["canvas_display"] = canvas.model_dump()
                    elif ext in ('.md', '.txt', '.rst') and len(content) > 200:
                        canvas = CanvasService.build_from_template(
                            "document_viewer",
                            {"content": content, "title": path.name, "file_path": str(path)},
                            title=path.name,
                        )
                        if canvas:
                            result["canvas_display"] = canvas.model_dump()
                    elif ext == '.json':
                        parsed = json.loads(content)
                        canvas = CanvasService.build_from_template(
                            "json_tree",
                            {"data": parsed, "title": path.name},
                            title=path.name,
                        )
                        if canvas:
                            result["canvas_display"] = canvas.model_dump()
                except Exception as e:
                    logger.debug(f"Canvas auto-detect skipped for {path.name}: {e}")

            return result
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {file_path}"}
        except Exception as e:
            return {"success": False, "error": f"Error reading file: {str(e)}"}

    async def _write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            path = Path(file_path).resolve()
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(content, encoding="utf-8")
            size = path.stat().st_size

            # Register with file manager for tracking
            try:
                await self.file_manager.register_file(
                    content=content.encode("utf-8"),
                    filename=path.name,
                    thread_id="default",
                    tags=["universal_agent", "written"],
                )
            except Exception as reg_err:
                logger.warning(f"Failed to register file in manager: {reg_err}")

            return {
                "success": True,
                "result": f"File written successfully: {path} ({size} bytes)",
                "data": {
                    "file_path": str(path),
                    "file_name": path.name,
                    "size_bytes": size,
                },
            }
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {file_path}"}
        except Exception as e:
            return {"success": False, "error": f"Error writing file: {str(e)}"}

    async def _list_directory(self, dir_path: str, pattern: Optional[str] = None) -> Dict[str, Any]:
        """List directory contents."""
        try:
            path = Path(dir_path).resolve()
            if not path.exists():
                return {"success": False, "error": f"Directory not found: {dir_path}"}
            if not path.is_dir():
                return {"success": False, "error": f"Not a directory: {dir_path}"}

            entries = []
            items = list(path.glob(pattern)) if pattern else list(path.iterdir())
            for item in sorted(items)[:200]:  # Cap at 200 entries
                try:
                    stat = item.stat()
                    entries.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size_bytes": stat.st_size if item.is_file() else None,
                        "extension": item.suffix if item.is_file() else None,
                    })
                except (PermissionError, OSError):
                    entries.append({"name": item.name, "type": "unknown", "error": "access denied"})

            # Build readable summary
            dirs = [e for e in entries if e["type"] == "directory"]
            files = [e for e in entries if e["type"] == "file"]
            summary = f"Directory: {path}\n{len(dirs)} directories, {len(files)} files"
            if dirs:
                summary += "\n\nDirectories:\n" + "\n".join(f"  📁 {d['name']}" for d in dirs[:50])
            if files:
                summary += "\n\nFiles:\n" + "\n".join(
                    f"  📄 {f['name']} ({f.get('size_bytes', 0)} bytes)" for f in files[:50]
                )

            return {
                "success": True,
                "result": summary,
                "data": {
                    "directory": str(path),
                    "total_entries": len(entries),
                    "entries": entries,
                },
            }
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {dir_path}"}
        except Exception as e:
            return {"success": False, "error": f"Error listing directory: {str(e)}"}

    async def _search_files(self, pattern: str, directory: str = ".") -> Dict[str, Any]:
        """Search for files matching a pattern."""
        try:
            base_path = Path(directory).resolve()
            if not base_path.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}

            matches = []
            for match in base_path.rglob(pattern):
                if len(matches) >= 100:  # Cap results
                    break
                try:
                    stat = match.stat()
                    matches.append({
                        "path": str(match),
                        "name": match.name,
                        "type": "directory" if match.is_dir() else "file",
                        "size_bytes": stat.st_size if match.is_file() else None,
                    })
                except (PermissionError, OSError):
                    pass

            summary = f"Found {len(matches)} matches for '{pattern}' in {base_path}:\n"
            summary += "\n".join(f"  {m['path']}" for m in matches[:50])

            return {
                "success": True,
                "result": summary,
                "data": {"pattern": pattern, "directory": str(base_path), "matches": matches},
            }
        except Exception as e:
            return {"success": False, "error": f"Error searching files: {str(e)}"}

    async def _manage_file(
        self, operation: str, source: str, destination: Optional[str] = None
    ) -> Dict[str, Any]:
        """Manage files: info, delete, move, copy."""
        try:
            src_path = Path(source).resolve()
            if not src_path.exists() and operation != "info":
                return {"success": False, "error": f"Source not found: {source}"}

            if operation == "info":
                if not src_path.exists():
                    return {"success": False, "error": f"File not found: {source}"}
                stat = src_path.stat()
                import mimetypes
                mime, _ = mimetypes.guess_type(str(src_path))
                info = {
                    "path": str(src_path),
                    "name": src_path.name,
                    "type": "directory" if src_path.is_dir() else "file",
                    "size_bytes": stat.st_size,
                    "extension": src_path.suffix,
                    "mime_type": mime or "unknown",
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat() if hasattr(stat, 'st_mtime') else None,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat() if hasattr(stat, 'st_ctime') else None,
                }
                return {
                    "success": True,
                    "result": f"File info: {src_path.name} ({stat.st_size} bytes, {mime})",
                    "data": info,
                }

            elif operation == "delete":
                if src_path.is_dir():
                    shutil.rmtree(str(src_path))
                else:
                    src_path.unlink()
                return {
                    "success": True,
                    "result": f"Deleted: {src_path}",
                }

            elif operation == "move":
                if not destination:
                    return {"success": False, "error": "Destination required for move"}
                dst_path = Path(destination).resolve()
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
                return {
                    "success": True,
                    "result": f"Moved: {src_path} → {dst_path}",
                }

            elif operation == "copy":
                if not destination:
                    return {"success": False, "error": "Destination required for copy"}
                dst_path = Path(destination).resolve()
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if src_path.is_dir():
                    shutil.copytree(str(src_path), str(dst_path))
                else:
                    shutil.copy2(str(src_path), str(dst_path))
                return {
                    "success": True,
                    "result": f"Copied: {src_path} → {dst_path}",
                }

            else:
                return {"success": False, "error": f"Unknown operation: {operation}. Use: info, delete, move, copy"}

        except PermissionError:
            return {"success": False, "error": f"Permission denied: {source}"}
        except Exception as e:
            return {"success": False, "error": f"Error in {operation}: {str(e)}"}

    # =========================================================================
    # ORIGINAL TASK CAPABILITIES
    # =========================================================================

    async def _execute_general_task(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute a general task with planning and execution."""
        prompt = request.prompt
        logger.info(f"Universal Agent executing task: {prompt[:100]}...")

        # Step 1: Plan the task
        plan = await self._plan_task(prompt, request.payload)
        logger.info(f"Task plan: {plan}")

        # Step 2: Execute each step
        results = []
        for step in plan.get("steps", []):
            step_result = await self._execute_universal_step(step, request)
            results.append(step_result)

            # If step fails, attempt recovery
            if not step_result.get("success", False):
                recovery = await self._attempt_recovery(step, step_result, request)
                if recovery:
                    results.append(recovery)

        # Step 3: Synthesize final response
        final_response = await self._synthesize_response(prompt, results)

        result = {
            "result": final_response,
            "plan": plan,
            "steps_executed": len(results),
            "success": True,
        }

        # Enrich with canvas if applicable
        result = await self._resolve_canvas_dynamic(result, prompt, "execute_task")
        return result

    async def _plan_task(self, prompt: str, payload: Dict) -> Dict:
        """Create a plan for executing the task."""
        planning_prompt = f"""You are a task planning assistant. Break down the following task into clear, executable steps.

Task: {prompt}

Context: {json.dumps(payload, default=str)}

Create a step-by-step plan. Each step should be:
1. Clear and specific
2. Executable on its own
3. Ordered logically

IMPORTANT: You have full filesystem access. For any task involving files:
- Use action_type "file_operation" for reading, writing, listing, or managing files
- You CAN read any file path on the local machine
- You CAN write/create files at any path

Respond in JSON format:
{{
    "steps": [
        {{
            "step_number": 1,
            "description": "What to do in this step",
            "action_type": "reasoning|code|research|writing|file_operation",
            "estimated_difficulty": "easy|medium|hard"
        }}
    ],
    "estimated_total_time": "brief|moderate|extended",
    "requires_code": true/false,
    "requires_research": true/false
}}"""

        try:
            # Use the agent's LLM service for planning
            from langchain_core.messages import HumanMessage, SystemMessage
            response = await self.services.inference.generate(
                messages=[HumanMessage(content=planning_prompt)],
                temperature=0.3,
            )
            # Try to parse JSON from response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback to simple single-step plan
            return {
                "steps": [
                    {
                        "step_number": 1,
                        "description": prompt,
                        "action_type": "reasoning",
                        "estimated_difficulty": "medium",
                    }
                ],
                "estimated_total_time": "brief",
                "requires_code": False,
                "requires_research": False,
            }

    async def _execute_universal_step(self, step: Dict, request: AgentRequest) -> Dict:
        """Execute a single step from the plan."""
        step_num = step.get("step_number", 1)
        description = step.get("description", "")
        action_type = step.get("action_type", "reasoning")

        logger.info(f"Executing step {step_num}: {description}")

        try:
            if action_type == "code":
                return await self._execute_code_step(description, request)
            elif action_type == "research":
                return await self._execute_research_step(description, request)
            elif action_type == "writing":
                return await self._execute_writing_step(description, request)
            elif action_type == "file_operation":
                return await self._execute_file_step(description, step, request)
            else:  # reasoning
                return await self._execute_reasoning_step(description, request)
        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            return {
                "step_number": step_num,
                "success": False,
                "error": str(e),
                "result": None,
            }

    async def _execute_reasoning_step(
        self, description: str, request: AgentRequest
    ) -> Dict:
        """Execute a reasoning/analysis step."""
        from langchain_core.messages import HumanMessage
        response = await self.services.inference.generate(
            messages=[HumanMessage(content=description)],
            temperature=0.7,
        )

        return {
            "step_number": 1,
            "success": True,
            "result": response,
            "type": "reasoning",
        }

    async def _execute_code_step(self, description: str, request: AgentRequest) -> Dict:
        """Execute a code generation step."""
        from langchain_core.messages import HumanMessage
        # Generate code
        code_prompt = f"""Write Python code to accomplish the following:

{description}

Requirements:
- Write clean, well-commented code
- Include error handling
- Save any output to a 'result' variable
- Don't include example usage or test code

Provide the code in a code block."""

        response = await self.services.inference.generate(
            messages=[HumanMessage(content=code_prompt)], 
            temperature=0.3
        )

        # Extract code from response
        code = self._extract_code(response)

        # Execute the code in sandbox
        try:
            from backend.services.code_sandbox_service import code_sandbox

            # Use thread_id as session_id for persistent state
            session_id = request.thread_id or "default"
            execution_result = code_sandbox.execute_code(
                code=code, session_id=session_id
            )

            # Get meaningful output — filter out sandbox system warnings
            sandbox_result = execution_result.get("result")
            sandbox_stdout = execution_result.get("stdout", "")
            has_real_output = sandbox_stdout and "[SYSTEM WARNING]" not in sandbox_stdout

            return await self._resolve_canvas_dynamic({
                "step_number": 1,
                "success": True,
                "code": code,
                "result": sandbox_result or (sandbox_stdout if has_real_output else None) or code,
                "output": sandbox_stdout,
                "error": execution_result.get("error"),
                "type": "code",
            }, description, "generate_code")
        except Exception as e:
            # Sandbox failed, but we still have the generated code
            return await self._resolve_canvas_dynamic({
                "step_number": 1,
                "success": True,
                "code": code,
                "result": code,
                "error": f"Code sandbox error (code still generated): {str(e)}",
                "type": "code",
            }, description, "generate_code")

    async def _execute_research_step(
        self, description: str, request: AgentRequest
    ) -> Dict:
        """Execute a research step using web search."""
        # Use web search tool if available
        try:
            from backend.services.tool_registry_service import tool_registry

            tool_registry.initialize()
            search_tool = tool_registry.get_tool("web_search_and_summarize")

            if search_tool:
                search_result = await search_tool.ainvoke({"query": description})
                return {
                    "step_number": 1,
                    "success": True,
                    "result": search_result,
                    "type": "research",
                }
        except Exception as e:
            logger.warning(f"Web search failed, falling back to LLM: {e}")

        # Fallback to LLM knowledge
        from langchain_core.messages import HumanMessage
        response = await self.services.inference.generate(
            messages=[HumanMessage(content=f"Research and provide comprehensive information about: {description}")],
            temperature=0.5,
        )

        return {
            "step_number": 1,
            "success": True,
            "result": response,
            "type": "research",
            "source": "llm_fallback",
        }

    async def _execute_writing_step(
        self, description: str, request: AgentRequest
    ) -> Dict:
        """Execute a creative writing step."""
        from langchain_core.messages import HumanMessage
        response = await self.services.inference.generate(
            messages=[HumanMessage(content=description)],
            temperature=0.8,  # Higher creativity
            max_tokens=2000,
        )

        return {
            "step_number": 1,
            "success": True,
            "result": response,
            "type": "writing",
        }

    async def _execute_file_step(
        self, description: str, step: Dict, request: AgentRequest
    ) -> Dict:
        """Execute a file operation step based on the description."""
        # Extract file operation details from description or step params
        params = step.get("parameters", {})
        desc_lower = description.lower()

        try:
            # Determine operation from description keywords
            if any(kw in desc_lower for kw in ["read", "open", "load", "view", "cat", "show contents"]):
                file_path = params.get("file_path") or params.get("path") or self._extract_path_from_text(description)
                if file_path:
                    result = await self._read_file(file_path)
                else:
                    return {"step_number": step.get("step_number", 1), "success": False, "error": "Could not determine file path from description"}

            elif any(kw in desc_lower for kw in ["write", "create", "save", "output"]):
                file_path = params.get("file_path") or params.get("path") or self._extract_path_from_text(description)
                content = params.get("content", "")
                if file_path:
                    result = await self._write_file(file_path, content)
                else:
                    return {"step_number": step.get("step_number", 1), "success": False, "error": "Could not determine file path from description"}

            elif any(kw in desc_lower for kw in ["list", "ls", "dir", "browse", "contents of"]):
                dir_path = params.get("path") or params.get("directory") or self._extract_path_from_text(description) or "."
                pattern = params.get("pattern")
                result = await self._list_directory(dir_path, pattern)

            elif any(kw in desc_lower for kw in ["search", "find", "locate", "glob"]):
                pattern = params.get("pattern", "*")
                directory = params.get("directory") or self._extract_path_from_text(description) or "."
                result = await self._search_files(pattern, directory)

            elif any(kw in desc_lower for kw in ["delete", "remove", "move", "copy", "rename"]):
                source = params.get("source") or params.get("path") or self._extract_path_from_text(description)
                destination = params.get("destination")
                op = "delete" if "delete" in desc_lower or "remove" in desc_lower else \
                     "move" if "move" in desc_lower or "rename" in desc_lower else "copy"
                if source:
                    result = await self._manage_file(op, source, destination)
                else:
                    return {"step_number": step.get("step_number", 1), "success": False, "error": "Could not determine file path from description"}

            else:
                # Default: use LLM reasoning about the file task
                return await self._execute_reasoning_step(description, request)

            # Wrap result in step format
            return {
                "step_number": step.get("step_number", 1),
                "success": result.get("success", False),
                "result": result.get("result"),
                "data": result.get("data"),
                "error": result.get("error"),
                "type": "file_operation",
            }

        except Exception as e:
            return {
                "step_number": step.get("step_number", 1),
                "success": False,
                "error": f"File operation failed: {str(e)}",
                "type": "file_operation",
            }

    def _extract_path_from_text(self, text: str) -> Optional[str]:
        """Try to extract a file/directory path from text."""
        import re
        # Match common path patterns (Windows and Unix)
        patterns = [
            r'[A-Za-z]:\\[\w\\\.\-\s]+',  # Windows absolute: C:\path\to\file
            r'/[\w/\.\-]+',                 # Unix absolute: /path/to/file
            r'[\w\.\-]+(?:[/\\][\w\.\-]+)+', # Relative: path/to/file
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return match.group(0).strip()
        return None

    async def _attempt_recovery(
        self, step: Dict, step_result: Dict, request: AgentRequest
    ) -> Optional[Dict]:
        """Attempt to recover from a failed step."""
        from langchain_core.messages import HumanMessage
        logger.info(f"Attempting recovery for step {step.get('step_number')}")

        recovery_prompt = f"""The following step failed:
Step: {step.get("description")}
Error: {step_result.get("error")}

Suggest an alternative approach or workaround.
Be concise and practical."""

        try:
            recovery_suggestion = await self.services.inference.generate(
                messages=[HumanMessage(content=recovery_prompt)], 
                temperature=0.4
            )

            # Create recovery step
            recovery_step = {
                "step_number": step.get("step_number"),
                "description": recovery_suggestion,
                "action_type": "reasoning",
                "is_recovery": True,
            }

            return await self._execute_universal_step(recovery_step, request)
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return None

    async def _synthesize_response(
        self, original_prompt: str, results: List[Dict]
    ) -> str:
        """Synthesize final response from all step results."""
        from langchain_core.messages import HumanMessage
        if len(results) == 1:
            return results[0].get("result", "Task completed")

        synthesis_prompt = f"""Synthesize the following step results into a coherent final response.

Original Task: {original_prompt}

Step Results:
{json.dumps(results, default=str, indent=2)}

Provide a comprehensive but concise response that addresses the original task.
Highlight key findings, code outputs, or insights where relevant."""

        try:
            response = await self.services.inference.generate(
                messages=[HumanMessage(content=synthesis_prompt)], 
                temperature=0.5
            )
            return response
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback to concatenating results
            return "\n\n".join(
                [r.get("result", "") for r in results if r.get("result")]
            )

    async def _analyze_content(self, request: AgentRequest) -> Dict[str, Any]:
        """Analyze content and provide insights."""
        from langchain_core.messages import HumanMessage
        prompt = request.prompt

        analysis_prompt = f"""Analyze the following content and provide insights:

{prompt}

Provide:
1. Key points or findings
2. Patterns or trends
3. Insights or implications
4. Recommendations (if applicable)"""

        response = await self.services.inference.generate(
            messages=[HumanMessage(content=analysis_prompt)], 
            temperature=0.4
        )

        return await self._resolve_canvas_dynamic(
            {"result": response, "type": "analysis", "success": True},
            prompt, "analyze",
        )

    async def _generate_code(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate code for the task."""
        return await self._execute_code_step(request.prompt, request)

    async def _research_topic(self, request: AgentRequest) -> Dict[str, Any]:
        """Research a topic comprehensively."""
        prompt = request.prompt

        # Multi-step research
        research_plan = [
            {
                "step_number": 1,
                "description": f"Search for current information about: {prompt}",
                "action_type": "research",
            },
            {
                "step_number": 2,
                "description": f"Analyze and synthesize findings about: {prompt}",
                "action_type": "reasoning",
            },
        ]

        results = []
        for step in research_plan:
            result = await self._execute_universal_step(step, request)
            results.append(result)

        final_response = await self._synthesize_response(prompt, results)

        return {
            "result": final_response,
            "findings": results,
            "type": "research",
            "success": True,
        }

    async def _creative_writing(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate creative writing."""
        return await self._resolve_canvas_dynamic(
            await self._execute_writing_step(request.prompt, request),
            request.prompt, "creative_write",
        )

    async def _solve_problem(self, request: AgentRequest) -> Dict[str, Any]:
        """Solve a complex problem."""
        from langchain_core.messages import HumanMessage
        prompt = request.prompt

        problem_solving_prompt = f"""Solve the following problem step by step:

{prompt}

Approach:
1. Understand the problem clearly
2. Break it down into components
3. Solve each component
4. Combine solutions
5. Verify the answer

Show your work and reasoning clearly."""

        response = await self.services.inference.generate(
            messages=[HumanMessage(content=problem_solving_prompt)], 
            temperature=0.5
        )

        return await self._resolve_canvas_dynamic(
            {"result": response, "type": "problem_solving", "success": True},
            prompt, "solve_problem",
        )

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        import re

        # Try to extract code block
        code_block_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        code_block_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # If no code block, return the whole response
        return response.strip()

    # -----------------------------------------------------------------------
    # Canvas — LLM-driven via shared CanvasService.decide_canvas_llm()
    # -----------------------------------------------------------------------
    async def _resolve_canvas_dynamic(self, result: Dict[str, Any], prompt: str = "", capability: str = "") -> Dict[str, Any]:
        """
        LLM-driven canvas selection. Replaces hardcoded file-type detection.
        Falls back to simple template matching if LLM unavailable.
        """
        # Already has canvas
        if result.get("canvas_display"):
            return result

        text = result.get("result", "")
        if not isinstance(text, str) or len(text) < 20:
            return result

        try:
            display = await CanvasService.decide_canvas_llm(
                output=text[:3000],
                agent_name="universal_agent",
                capability_name=capability or result.get("type", "execute_task"),
            )
            if display:
                result["canvas_display"] = display.model_dump() if hasattr(display, "model_dump") else display.dict()
                return result
        except Exception as e:
            logger.debug(f"LLM canvas decision failed (non-fatal): {e}")

        # Fallback: old template-based detection
        result_type = result.get("type", "")
        try:
            if result_type == "code" or "```" in text:
                code = self._extract_code(text)
                if code and len(code) > 30:
                    import re
                    lang_match = re.search(r'```(\w+)', text)
                    lang = lang_match.group(1) if lang_match else 'python'
                    canvas = CanvasService.build_from_template(
                        "code_viewer", {"code": code, "language": lang}, title=f"Generated Code ({lang})",
                    )
                    if canvas:
                        result["canvas_display"] = canvas.model_dump()
                        return result

            if result_type in ("analysis", "creative_writing", "problem_solving", "research") and len(text) > 500:
                canvas = CanvasService.build_from_template(
                    "document_viewer",
                    {"content": text, "title": prompt[:80] if prompt else "Result", "status": "created"},
                    title=result_type.replace('_', ' ').title(),
                )
                if canvas:
                    result["canvas_display"] = canvas.model_dump()
        except Exception as e:
            logger.warning(f"Canvas enrichment failed (non-fatal): {e}")

        return result
