"""
Browser Agent v2.0 - Complete BaseAgent Implementation (Fixed)

Stateful browser automation with corrected method mappings.
Uses actual Browser class methods from browser.py.
"""

import logging
import asyncio
import base64
from typing import Dict, Any, Optional
from dataclasses import dataclass

from backend.agents.base import BaseAgent, AgentServices, AgentConfig
from backend.agents.base.types import ExecutionContext
from backend.agents.base.capability import capability, ParameterSchema

from .browser import Browser
from .dom import DOMExtractor
from .llm import LLMClient
from .state import AgentMemory

logger = logging.getLogger("agents.browser_agent")


@dataclass
class BrowserAgentConfig(AgentConfig):
    """Configuration for Browser Agent."""

    headless: bool = False
    max_steps: int = 50
    screenshot_quality: int = 70
    enable_vision: bool = True
    download_timeout: int = 60


class BrowserAgent(BaseAgent):
    """
    Complete browser automation agent with corrected method mappings.
    """

    def __init__(
        self,
        agent_id: str = "browser_agent",
        agent_name: str = "Browser Agent",
        services: Optional[AgentServices] = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services,
            config=config or BrowserAgentConfig(),
        )

        # Components
        self.browser: Optional[Browser] = None
        self.dom: Optional[DOMExtractor] = None
        self.llm: Optional[LLMClient] = None
        self.memory: Optional[AgentMemory] = None

        # Metrics
        self._execution_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "screenshots_taken": 0,
            "downloads_completed": 0,
        }

    async def _initialize_resources(self):
        """Initialize browser and components."""
        logger.info("Initializing Browser Agent resources...")

        self.browser = Browser()
        self.dom = DOMExtractor()
        self.llm = LLMClient()
        self.memory = AgentMemory(task="Initialize Browser Agent")

        logger.info("Browser Agent resources initialized")

    async def _cleanup_resources(self):
        """Cleanup browser and resources."""
        logger.info("Cleaning up Browser Agent resources...")
        if self.browser:
            await self.browser.close()

    async def _get_custom_metrics(self) -> Optional[Dict[str, Any]]:
        """Return browser metrics."""
        return self._execution_metrics.copy()

    def _get_page(self):
        """Helper to get active page."""
        if not self.browser:
            return None
        return self.browser.get_active_page()

    # ========================================================================
    # CAPABILITIES - Navigation & Interaction
    # ========================================================================

    @capability(
        name="navigate",
        description="Navigate to a URL and load the page",
        parameters=[
            ParameterSchema(
                name="url",
                type="string",
                description="URL to navigate to",
                required=True,
            ),
            ParameterSchema(
                name="wait_for_load",
                type="boolean",
                description="Wait for page load",
                required=False,
                default=True,
            ),
            ParameterSchema(
                name="timeout",
                type="integer",
                description="Navigation timeout",
                required=False,
                default=30,
            ),
        ],
    )
    async def navigate(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Navigate to a URL."""
        url = params.get("url", "")
        timeout = params.get("timeout", 30)

        if not url:
            return {"success": False, "error": "URL is required"}

        try:
            page = self._get_page()
            if not page:
                await self.browser.launch(headless=self.config.headless)

            success = await self.browser.navigate(url, timeout=timeout * 1000)

            if success:
                page_url = await self.browser.get_url()
                title = await self.browser.get_title()
                return {
                    "success": True,
                    "data": {"url": page_url, "title": title},
                    "message": f"Navigated to {page_url}",
                }
            else:
                return {"success": False, "error": "Navigation failed"}
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return {"success": False, "error": f"Navigation failed: {str(e)}"}

    @capability(
        name="click",
        description="Click an element on the page",
        parameters=[
            ParameterSchema(
                name="selector",
                type="string",
                description="CSS selector",
                required=True,
            ),
            ParameterSchema(
                name="wait_for_navigation",
                type="boolean",
                description="Wait for navigation",
                required=False,
                default=False,
            ),
        ],
    )
    async def click(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Click an element."""
        selector = params.get("selector")

        if not selector:
            return {"success": False, "error": "Selector is required"}

        try:
            page = self._get_page()
            if not page:
                return {"success": False, "error": "Browser not initialized"}

            await page.click(selector)
            await asyncio.sleep(0.5)

            return {
                "success": True,
                "data": {"url": await self.browser.get_url()},
                "message": f"Clicked element: {selector}",
            }
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return {"success": False, "error": f"Click failed: {str(e)}"}

    @capability(
        name="type_text",
        description="Type text into an input field",
        parameters=[
            ParameterSchema(
                name="selector",
                type="string",
                description="CSS selector",
                required=True,
            ),
            ParameterSchema(
                name="text", type="string", description="Text to type", required=True
            ),
            ParameterSchema(
                name="press_enter",
                type="boolean",
                description="Press Enter after typing",
                required=False,
                default=False,
            ),
        ],
    )
    async def type_text(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Type text into an input field."""
        selector = params.get("selector")
        text = params.get("text", "")
        press_enter = params.get("press_enter", False)

        if not selector:
            return {"success": False, "error": "Selector is required"}

        try:
            page = self._get_page()
            if not page:
                return {"success": False, "error": "Browser not initialized"}

            await page.fill(selector, "")
            await page.fill(selector, text)

            if press_enter:
                await page.press(selector, "Enter")
                await asyncio.sleep(1)

            return {"success": True, "message": f"Typed text into {selector}"}
        except Exception as e:
            logger.error(f"Type text failed: {e}")
            return {"success": False, "error": f"Type failed: {str(e)}"}

    @capability(
        name="wait_for_element",
        description="Wait for an element to appear",
        parameters=[
            ParameterSchema(
                name="selector",
                type="string",
                description="CSS selector",
                required=True,
            ),
            ParameterSchema(
                name="timeout",
                type="integer",
                description="Timeout in seconds",
                required=False,
                default=10,
            ),
        ],
    )
    async def wait_for_element(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Wait for an element."""
        selector = params.get("selector")
        timeout = params.get("timeout", 10)

        if not selector:
            return {"success": False, "error": "Selector is required"}

        try:
            success = await self.browser.wait_for_element(
                selector, timeout=timeout * 1000
            )

            return {
                "success": success,
                "message": f"Element {selector} is visible"
                if success
                else f"Timeout waiting for {selector}",
            }
        except Exception as e:
            return {"success": False, "error": f"Wait failed: {str(e)}"}

    @capability(
        name="execute_javascript",
        description="Execute JavaScript code on the page. IMPORTANT: Do NOT use top-level 'return' statements; just write the expression directly (e.g. 'document.querySelector(\"h1\").textContent' instead of 'return document.querySelector(\"h1\").textContent').",
        parameters=[
            ParameterSchema(
                name="script",
                type="string",
                description="JavaScript expression or code. Do NOT use top-level return statements — just write the expression.",
                required=True,
            )
        ],
    )
    async def execute_javascript(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute JavaScript."""
        script = params.get("script", "")

        if not script:
            return {"success": False, "error": "Script is required"}

        # Sanitize LLM-generated scripts
        script = script.strip()

        # Strip markdown code fences that LLMs sometimes wrap code in
        if script.startswith("```"):
            lines = script.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            script = "\n".join(lines).strip()

        # Wrap in IIFE if script contains a top-level 'return' statement.
        # Playwright's page.evaluate() uses eval() where bare 'return' is
        # a SyntaxError. LLMs consistently generate 'return expr;' style code.
        stripped = script.lstrip()
        if stripped.startswith("return ") or stripped.startswith("return;") or stripped.startswith("return\n"):
            script = f"(() => {{ {script} }})()"

        try:
            page = self._get_page()
            if not page:
                return {"success": False, "error": "Browser not initialized"}

            result = await page.evaluate(script)

            return {
                "success": True,
                "data": {"result": result},
                "message": "JavaScript executed",
            }
        except Exception as e:
            logger.error(f"JavaScript execution failed: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # CAPABILITIES - Data Extraction
    # ========================================================================

    @capability(
        name="extract_data",
        description="Extract data from the page",
        parameters=[
            ParameterSchema(
                name="extract_type",
                type="string",
                description="Type: text, links, images",
                required=False,
                default="text",
            )
        ],
    )
    async def extract_data(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Extract data from page."""
        extract_type = params.get("extract_type", "text")

        try:
            page = self._get_page()
            if not page:
                return {"success": False, "error": "Browser not initialized"}

            page_content = await self.dom.get_page_content(page)

            extracted = {
                "url": page_content.get("url"),
                "title": page_content.get("title"),
            }

            if extract_type == "text":
                extracted["text"] = await self.dom.extract_text(page)
            elif extract_type == "links":
                elements = page_content.get("elements", [])
                extracted["links"] = [el for el in elements if el.get("tag") == "a"][
                    :50
                ]

            return {
                "success": True,
                "data": extracted,
                "message": f"Extracted {extract_type} from page",
            }
        except Exception as e:
            logger.error(f"Extract failed: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # CAPABILITIES - Screenshots
    # ========================================================================

    @capability(
        name="screenshot",
        description="Take a screenshot",
        parameters=[
            ParameterSchema(
                name="full_page",
                type="boolean",
                description="Full page screenshot",
                required=False,
                default=False,
            )
        ],
    )
    async def screenshot(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Take a screenshot."""
        params.get("full_page", False)

        try:
            screenshot_bytes = await self.browser.screenshot()

            if screenshot_bytes:
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                self._execution_metrics["screenshots_taken"] += 1

                return {
                    "success": True,
                    "data": {"screenshot_base64": screenshot_b64, "format": "png"},
                    "message": "Screenshot captured",
                }
            else:
                return {"success": False, "error": "Screenshot failed"}
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # CAPABILITIES - Page Info & Navigation
    # ========================================================================

    @capability(
        name="get_page_info", description="Get current page information", parameters=[]
    )
    async def get_page_info(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Get page information."""
        try:
            url = await self.browser.get_url()
            title = await self.browser.get_title()

            return {
                "success": True,
                "data": {"url": url, "title": title},
                "message": f"Current page: {title}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @capability(
        name="scroll",
        description="Scroll the page",
        parameters=[
            ParameterSchema(
                name="direction",
                type="string",
                description="Direction: up, down, top, bottom",
                required=True,
            )
        ],
    )
    async def scroll(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Scroll page."""
        direction = params.get("direction", "down")

        try:
            page = self._get_page()
            if not page:
                return {"success": False, "error": "Browser not initialized"}

            if direction == "down":
                await page.evaluate("window.scrollBy(0, 500)")
            elif direction == "up":
                await page.evaluate("window.scrollBy(0, -500)")
            elif direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            return {"success": True, "message": f"Scrolled {direction}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @capability(name="go_back", description="Navigate back", parameters=[])
    async def go_back(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Go back."""
        try:
            page = self._get_page()
            if page:
                await page.go_back()
                await asyncio.sleep(1)
                return {"success": True, "data": {"url": await self.browser.get_url()}}
            return {"success": False, "error": "No page"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @capability(name="go_forward", description="Navigate forward", parameters=[])
    async def go_forward(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Go forward."""
        try:
            page = self._get_page()
            if page:
                await page.go_forward()
                await asyncio.sleep(1)
                return {"success": True, "data": {"url": await self.browser.get_url()}}
            return {"success": False, "error": "No page"}
        except Exception as e:
            return {"success": False, "error": str(e)}
