"""
OpenCode Client

Async Python client for OpenCode's headless server REST API.
Handles auto-installation, server lifecycle, session management,
coding prompts, and file change collection.

End users get the npm CLI (`opencode-ai`) which exposes clean REST
endpoints via `opencode serve`. The CLI is auto-installed on first use.

Docs: https://opencode.ai/docs/sdk
"""

import asyncio
import json
import subprocess
import shutil
import os
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import httpx

from .config import (
    OPENCODE_SERVER_PORT,
    OPENCODE_SERVER_HOST,
    OPENCODE_SERVER_PASSWORD,
    OPENCODE_SERVER_USERNAME,
    OPENCODE_STARTUP_TIMEOUT,
    OPENCODE_REQUEST_TIMEOUT,
    HEALTH_CHECK_INTERVAL,
    logger,
)

# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class OpenCodeSession:
    """Represents an OpenCode session."""
    id: str
    title: Optional[str] = None


@dataclass
class FileChange:
    """A single file change from OpenCode."""
    file_path: str
    diff: str
    language: str = "text"
    status: str = "modified"  # modified, created, deleted


@dataclass
class CodeTaskResult:
    """Result of a coding task execution."""
    success: bool
    output: str = ""
    terminal_log: str = ""
    file_changes: List[FileChange] = field(default_factory=list)
    summary: str = ""
    files_modified: List[str] = field(default_factory=list)
    error: Optional[str] = None
    session_id: Optional[str] = None


# ============================================================================
# OPENCODE CLIENT
# ============================================================================


class OpenCodeClient:
    """
    Async client for OpenCode's headless server.

    - Auto-installs the CLI via npm if not found
    - Manages `opencode serve` subprocess lifecycle
    - Creates sessions and sends prompts via REST API
    - Collects file diffs via git after prompt completion
    """

    def __init__(
        self,
        host: str = OPENCODE_SERVER_HOST,
        port: int = OPENCODE_SERVER_PORT,
        password: str = OPENCODE_SERVER_PASSWORD,
        username: str = OPENCODE_SERVER_USERNAME,
        project_dir: Optional[str] = None,
        default_model: str = "ollama/minimax-m2.5:cloud",
    ):
        self.host = host
        self.port = port
        self.password = password
        self.username = username
        self.project_dir = project_dir or os.getcwd()
        self.base_url = f"http://{host}:{port}"
        self.default_model = default_model

        self._process: Optional[subprocess.Popen] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._opencode_bin: Optional[str] = None

    # ------------------------------------------------------------------
    # AUTO-INSTALL
    # ------------------------------------------------------------------

    @staticmethod
    def _find_opencode() -> Optional[str]:
        """Find the opencode binary in PATH."""
        return shutil.which("opencode")

    @staticmethod
    async def _auto_install() -> bool:
        """
        Auto-install the OpenCode CLI via npm on first use.
        Returns True if installation succeeded or was already installed.
        """
        logger.info("OpenCode CLI not found. Auto-installing via npm...")

        # Check npm is available
        npm_bin = shutil.which("npm")
        if not npm_bin:
            logger.error(
                "npm not found. Cannot auto-install OpenCode. "
                "Please install Node.js from https://nodejs.org"
            )
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                npm_bin, "install", "-g", "opencode-ai@latest",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=120
            )

            if proc.returncode == 0:
                logger.info("OpenCode CLI installed successfully via npm")
                return True
            else:
                err_text = stderr.decode("utf-8", errors="replace")
                logger.error(f"npm install failed (rc={proc.returncode}): {err_text[:300]}")
                return False

        except asyncio.TimeoutError:
            logger.error("npm install timed out after 120s")
            return False
        except Exception as e:
            logger.error(f"Auto-install failed: {e}")
            return False

    async def _ensure_installed(self) -> bool:
        """Ensure OpenCode CLI is installed, auto-installing if needed."""
        self._opencode_bin = self._find_opencode()
        if self._opencode_bin:
            return True

        # Auto-install
        installed = await self._auto_install()
        if not installed:
            return False

        # Re-check after install
        self._opencode_bin = self._find_opencode()
        if not self._opencode_bin:
            logger.error("OpenCode CLI still not found after install. Check your PATH.")
            return False

        return True

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def start_server(self) -> bool:
        """
        Start the OpenCode headless server.
        Auto-installs the CLI if not found.
        Returns True if server is healthy.
        """
        # Ensure CLI is installed
        if not await self._ensure_installed():
            return False

        # Check if already running
        if self._process and self._process.poll() is None:
            if await self.health_check():
                logger.info("OpenCode server already running")
                return True

        cmd = [self._opencode_bin, "serve", "--port", str(self.port)]

        env = dict(os.environ)
        if self.password:
            env["OPENCODE_SERVER_PASSWORD"] = self.password

        # OpenCode expects CEREBRAS_API_KEY (singular).
        # Our .env stores CEREBRAS_API_KEYS (plural, comma-separated list).
        # Extract the first key so OpenCode can authenticate with Cerebras.
        if not env.get("CEREBRAS_API_KEY") and env.get("CEREBRAS_API_KEYS"):
            first_key = env["CEREBRAS_API_KEYS"].split(",")[0].strip()
            if first_key:
                env["CEREBRAS_API_KEY"] = first_key
                logger.info("Extracted CEREBRAS_API_KEY from CEREBRAS_API_KEYS for OpenCode subprocess")

        logger.info(f"Starting OpenCode server: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=self.project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=(sys.platform == "win32"),
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == "win32"
                    else 0
                ),
            )
            logger.info(f"OpenCode server started with PID {self._process.pid}")
        except Exception as e:
            logger.error(f"Failed to start OpenCode server: {e}")
            return False

        # Wait for health
        healthy = await self._wait_for_health()
        if not healthy:
            logger.error("OpenCode server failed health check")
            await self.stop_server()
            return False

        # Create HTTP client for REST API
        self._client = self._create_http_client()
        logger.info(f"OpenCode server ready at {self.base_url}")
        return True

    async def stop_server(self):
        """Stop the OpenCode server subprocess."""
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._process:
            if self._process.poll() is None:
                try:
                    if sys.platform == "win32":
                        self._process.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
                    else:
                        self._process.terminate()
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait()
                except Exception as e:
                    logger.error(f"Error stopping server: {e}")
            self._process = None

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create HTTP client with optional basic auth."""
        auth = None
        if self.password:
            auth = httpx.BasicAuth(self.username, self.password)
        return httpx.AsyncClient(
            base_url=self.base_url,
            auth=auth,
            timeout=httpx.Timeout(OPENCODE_REQUEST_TIMEOUT, connect=10.0),
        )

    async def _wait_for_health(self) -> bool:
        """Poll until the server responds."""
        import time
        start = time.time()
        logger.info(f"Waiting for OpenCode on port {self.port} (timeout: {OPENCODE_STARTUP_TIMEOUT}s)...")

        async with httpx.AsyncClient() as client:
            while time.time() - start < OPENCODE_STARTUP_TIMEOUT:
                if self._process and self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                    logger.error(f"OpenCode server exited. stderr: {stderr[:500]}")
                    return False
                try:
                    resp = await client.get(f"{self.base_url}/", timeout=3.0)
                    if resp.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        return False

    async def health_check(self) -> bool:
        """Quick health check."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/", timeout=5.0)
                return resp.status_code == 200
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # SESSION MANAGEMENT (REST API)
    # ------------------------------------------------------------------

    async def create_session(self, title: str = "Orbimesh Coding Task") -> OpenCodeSession:
        """Create a new coding session via REST."""
        resp = await self._request("POST", "/session", json={"title": title})
        return OpenCodeSession(
            id=resp.get("id", resp.get("ID", "")),
            title=resp.get("title", title),
        )

    # ------------------------------------------------------------------
    # PROMPTING (REST API or CLI fallback)
    # ------------------------------------------------------------------

    async def send_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
    ) -> CodeTaskResult:
        """
        Send a coding prompt. Tries REST API first, falls back to CLI.

        Args:
            prompt: Natural language coding instruction
            model: Model in "provider/model" format

        Returns:
            CodeTaskResult with output, file changes, and terminal log
        """
        model = model or self.default_model
        provider_id, model_id = self._parse_model(model)

        # Try REST API first (works with npm CLI's `opencode serve`)
        try:
            result = await self._send_prompt_rest(prompt, provider_id, model_id)
            if result.success:
                return result
            # If REST returned an error, fall through to CLI
            logger.warning(f"REST prompt failed: {result.error}. Trying CLI fallback...")
        except Exception as e:
            logger.warning(f"REST prompt exception: {e}. Trying CLI fallback...")

        # Fallback: CLI approach (works with both app and CLI)
        return await self._send_prompt_cli(prompt, model)

    async def _send_prompt_rest(
        self, prompt: str, provider_id: str, model_id: str
    ) -> CodeTaskResult:
        """Send prompt via REST API (POST /session/{id}/message)."""
        # Create a session
        session = await self.create_session(title=f"Task: {prompt[:40]}")

        body = {
            "parts": [{"type": "text", "text": prompt}],
            "model": {"providerID": provider_id, "modelID": model_id},
        }

        logger.info(f"REST prompt to session {session.id}: {prompt[:80]}...")

        resp = await self._request(
            "POST",
            f"/session/{session.id}/message",
            json=body,
            timeout=OPENCODE_REQUEST_TIMEOUT,
        )

        # Check if response is HTML (SPA routing — means app version, not CLI)
        if isinstance(resp, str) or (isinstance(resp, dict) and "<!doctype" in str(resp).lower()):
            return CodeTaskResult(
                success=False,
                error="REST API returned HTML (desktop app detected). Using CLI fallback.",
                session_id=session.id,
            )

        # Extract output from response
        output = self._extract_text_from_response(resp)
        terminal_log = self._extract_terminal_log(resp)
        summary = self._extract_summary(output)

        # Collect file changes
        file_changes, files_modified = await self._collect_file_changes()

        return CodeTaskResult(
            success=True,
            output=output,
            terminal_log=terminal_log or self._build_terminal_log(prompt, output, files_modified),
            file_changes=file_changes,
            summary=summary,
            files_modified=files_modified,
            session_id=session.id,
        )

    async def _send_prompt_cli(self, prompt: str, model: str) -> CodeTaskResult:
        """Send prompt via CLI (opencode run -m <model> <prompt>).

        Note: --attach was removed in opencode-ai v1.2.x.
        This fallback runs opencode directly without attaching to a server.
        """
        if not self._opencode_bin:
            self._opencode_bin = self._find_opencode()
            if not self._opencode_bin:
                return CodeTaskResult(success=False, error="OpenCode CLI not found")

        cmd = [self._opencode_bin, "run", "-m", model, prompt]
        logger.info(f"CLI prompt: {prompt[:80]}...")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(os.environ),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=OPENCODE_REQUEST_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                return CodeTaskResult(success=False, error=f"Timed out after {OPENCODE_REQUEST_TIMEOUT}s")

            output = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                return CodeTaskResult(success=False, output=output, error=err or f"Exit code {proc.returncode}")

            clean_output = self._strip_cli_header(output)
            file_changes, files_modified = await self._collect_file_changes()
            summary = self._extract_summary(clean_output)
            terminal_log = self._build_terminal_log(prompt, clean_output, files_modified)

            return CodeTaskResult(
                success=True,
                output=clean_output,
                terminal_log=terminal_log,
                file_changes=file_changes,
                summary=summary,
                files_modified=files_modified,
            )
        except Exception as e:
            return CodeTaskResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # FILE CHANGE COLLECTION (via git)
    # ------------------------------------------------------------------

    async def _collect_file_changes(self) -> tuple[List[FileChange], List[str]]:
        """Collect file changes via git diff after a prompt completes."""
        file_changes: List[FileChange] = []
        files_modified: List[str] = []

        try:
            # Modified/staged files
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "--name-status",
                cwd=self.project_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            names = stdout.decode("utf-8", errors="replace").strip()

            if not names:
                # Check untracked files
                proc2 = await asyncio.create_subprocess_exec(
                    "git", "ls-files", "--others", "--exclude-standard",
                    cwd=self.project_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout2, _ = await proc2.communicate()
                untracked = stdout2.decode("utf-8", errors="replace").strip()
                if not untracked:
                    return [], []
                for line in untracked.split("\n"):
                    fpath = line.strip()
                    if fpath:
                        files_modified.append(fpath)
                        file_changes.append(FileChange(
                            file_path=fpath, diff=f"+++ new file: {fpath}",
                            language=self._guess_language(fpath), status="created",
                        ))
                return file_changes, files_modified

            for line in names.split("\n"):
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                status_code, fpath = parts[0], parts[1]
                files_modified.append(fpath)
                status = "created" if status_code.startswith("A") else "deleted" if status_code.startswith("D") else "modified"

                # Get unified diff
                proc3 = await asyncio.create_subprocess_exec(
                    "git", "diff", "--", fpath,
                    cwd=self.project_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout3, _ = await proc3.communicate()
                diff = stdout3.decode("utf-8", errors="replace").strip()

                file_changes.append(FileChange(
                    file_path=fpath, diff=diff,
                    language=self._guess_language(fpath), status=status,
                ))
        except Exception as e:
            logger.warning(f"Error collecting file changes: {e}")

        return file_changes, files_modified

    async def revert_changes(self) -> bool:
        """Revert all uncommitted changes (user rejected via canvas)."""
        try:
            for cmd in [["git", "checkout", "."], ["git", "clean", "-fd"]]:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=self.project_dir,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            logger.info("Reverted all uncommitted changes")
            return True
        except Exception as e:
            logger.error(f"Error reverting: {e}")
            return False

    # ------------------------------------------------------------------
    # HTTP HELPERS
    # ------------------------------------------------------------------

    async def _request(
        self, method: str, path: str, json: Optional[Dict] = None,
        params: Optional[Dict] = None, timeout: Optional[float] = None,
    ) -> Any:
        """Make an HTTP request to the OpenCode server."""
        if not self._client:
            self._client = self._create_http_client()

        kwargs: Dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if params is not None:
            kwargs["params"] = params
        if timeout is not None:
            kwargs["timeout"] = timeout

        resp = await self._client.request(method, path, **kwargs)
        resp.raise_for_status()

        if not resp.content:
            return {}

        # Check for HTML response (SPA fallback)
        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type:
            return {"_html": True}

        return resp.json()

    # ------------------------------------------------------------------
    # PARSING HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model(model: str) -> tuple[str, str]:
        """Parse 'provider/model' into (provider_id, model_id)."""
        if "/" in model:
            parts = model.split("/", 1)
            return parts[0], parts[1]
        return "cerebras", model

    @staticmethod
    def _extract_text_from_response(resp: Dict[str, Any]) -> str:
        """Extract readable text from an OpenCode API response."""
        parts = resp.get("parts", [])
        text_parts = []
        for part in parts:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "\n".join(text_parts).strip() if text_parts else str(resp)

    @staticmethod
    def _extract_terminal_log(resp: Dict[str, Any]) -> str:
        """Extract tool calls and outputs as a terminal log."""
        lines: List[str] = []
        for part in resp.get("parts", []):
            ptype = part.get("type", "")
            if ptype == "text":
                text = part.get("text", "").strip()
                if text:
                    lines.append(text)
            elif ptype in ("tool-invocation", "tool_call"):
                tool = part.get("toolName", part.get("name", "tool"))
                args = part.get("args", {})
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: pass
                if tool in ("edit", "write", "patch"):
                    lines.append(f"✏️  Editing {args.get('file_path', args.get('filePath', ''))}")
                elif tool in ("bash", "shell"):
                    lines.append(f"🖥️  Running: {args.get('command', '')}")
                elif tool in ("read", "view"):
                    lines.append(f"🔍 Reading {args.get('file_path', '')}")
                else:
                    lines.append(f"🔧 {tool}")
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _strip_cli_header(output: str) -> str:
        """Strip '> build · model-name' header from CLI output."""
        lines = output.split("\n")
        if lines and lines[0].strip().startswith(">"):
            return "\n".join(lines[1:]).strip()
        return output.strip()

    @staticmethod
    def _build_terminal_log(prompt: str, output: str, files: List[str]) -> str:
        """Build a terminal-log-style display."""
        lines = [f"🔧 Task: {prompt[:100]}"]
        if files:
            lines.append(f"📁 Files changed: {len(files)}")
            for f in files[:10]:
                lines.append(f"   • {f}")
            if len(files) > 10:
                lines.append(f"   ... and {len(files) - 10} more")
        if output:
            lines.append("\n💬 AI Response:")
            for line in output.split("\n")[:30]:
                lines.append(f"   {line}")
        lines.append("\n✅ Task completed.")
        return "\n".join(lines)

    @staticmethod
    def _extract_summary(output: str) -> str:
        """First paragraph as summary (max 200 chars)."""
        if not output:
            return "Coding task completed."
        first = output.split("\n\n")[0].strip()
        return first[:197] + "..." if len(first) > 200 else (first or "Coding task completed.")

    @staticmethod
    def _guess_language(path: str) -> str:
        """Guess language from file extension."""
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".tsx": "tsx", ".jsx": "jsx", ".java": "java", ".go": "go",
            ".rs": "rust", ".rb": "ruby", ".php": "php", ".c": "c",
            ".cpp": "cpp", ".cs": "csharp", ".sh": "bash",
            ".yaml": "yaml", ".yml": "yaml", ".json": "json",
            ".md": "markdown", ".html": "html", ".css": "css", ".sql": "sql",
        }
        _, ext = os.path.splitext(path.lower())
        return ext_map.get(ext, "text")
