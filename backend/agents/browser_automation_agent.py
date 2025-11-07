# agents/custom_browser_agent.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import asyncio
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI
import logging
from pathlib import Path
import uuid
from datetime import datetime
import json
import base64
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not CEREBRAS_API_KEY and not GROQ_API_KEY and not NVIDIA_API_KEY:
    raise RuntimeError("At least one of CEREBRAS_API_KEY, GROQ_API_KEY, or NVIDIA_API_KEY must be set")

# Vision model configuration (optional)
VISION_ENABLED = bool(OLLAMA_API_KEY)
if VISION_ENABLED:
    logger.info("🎨 Vision capabilities enabled with Ollama Qwen3-VL")
else:
    logger.info("📝 Running in text-only mode (set OLLAMA_API_KEY for vision)")

# --- Agent Definition ---
AGENT_DEFINITION = {
    "id": "custom_browser_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Custom Browser Automation Agent",
    "description": "A powerful custom browser automation agent with full control over web interactions",
    "capabilities": [
        "web browsing", "data extraction", "form filling", "screenshot capture",
        "web scraping", "page navigation", "element interaction"
    ],
    "price_per_call_usd": 0.01,
    "status": "active",
    "endpoints": [{
        "endpoint": "http://localhost:8070/browse",
        "http_method": "POST",
        "description": "Execute browser automation task",
        "parameters": [{
            "name": "task",
            "param_type": "string",
            "required": True,
            "description": "Task description"
        }]
    }]
}

app = FastAPI(title="Custom Browser Automation Agent")

# Storage
STORAGE_DIR = Path("storage/browser_screenshots")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS_DIR = Path("storage/browser_downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = Path("storage/browser_uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Live streaming storage - for canvas updates
live_screenshots: Dict[str, Dict[str, Any]] = {}
live_screenshots_lock = asyncio.Lock()

# Request/Response Models
class BrowseRequest(BaseModel):
    task: str
    extract_data: Optional[bool] = False
    max_steps: Optional[int] = 10

class BrowseResponse(BaseModel):
    success: bool
    task_summary: str
    actions_taken: List[Dict[str, Any]]
    extracted_data: Optional[Dict[str, Any]] = None
    screenshot_files: Optional[List[str]] = None
    error: Optional[str] = None

# Track active browser instances to prevent duplicates
active_browsers = {}
active_browsers_lock = asyncio.Lock()

# LLM Client with fallback chain
class LLMManager:
    """Manages LLM providers with intelligent fallback"""
    
    def __init__(self):
        self.providers = []
        self.current_provider_idx = 0
        self.failure_counts = {}
        
        # Build provider chain: Cerebras → Groq → NVIDIA
        if CEREBRAS_API_KEY:
            self.providers.append({
                "name": "cerebras",
                "client": OpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1"),
                "model": "llama-3.3-70b",
                "max_tokens": 1000
            })
        
        if GROQ_API_KEY:
            self.providers.append({
                "name": "groq",
                "client": OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 1000
            })
        
        if NVIDIA_API_KEY:
            self.providers.append({
                "name": "nvidia",
                "client": OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1"),
                "model": "meta/llama-3.1-70b-instruct",
                "max_tokens": 1000
            })
        
        if not self.providers:
            raise RuntimeError("No LLM providers available")
        
        logger.info(f"🔧 Initialized LLM chain: {' → '.join([p['name'] for p in self.providers])}")
    
    def get_completion(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 500):
        """Get completion with automatic fallback"""
        last_error = None
        
        # Try each provider in order
        for idx, provider in enumerate(self.providers):
            try:
                logger.info(f"🤖 Using {provider['name'].upper()} for completion")
                
                response = provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=min(max_tokens, provider["max_tokens"])
                )
                
                # Success - reset failure count
                self.failure_counts[provider["name"]] = 0
                return response.choices[0].message.content.strip(), provider["name"]
                
            except Exception as e:
                error_msg = str(e).lower()
                self.failure_counts[provider["name"]] = self.failure_counts.get(provider["name"], 0) + 1
                
                # Check if it's a rate limit or temporary error
                if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                    logger.warning(f"⚠️  {provider['name'].upper()} rate limited, trying next provider...")
                elif "timeout" in error_msg:
                    logger.warning(f"⚠️  {provider['name'].upper()} timeout, trying next provider...")
                else:
                    logger.warning(f"⚠️  {provider['name'].upper()} error: {str(e)[:100]}")
                
                last_error = e
                
                # If this is the last provider, raise the error
                if idx == len(self.providers) - 1:
                    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

class BrowserAgent:
    """Custom SOTA browser automation agent with multi-provider LLM fallback and task planning"""
    
    def __init__(self, task: str, max_steps: int = 10, headless: bool = False, enable_streaming: bool = True, thread_id: Optional[str] = None, backend_url: Optional[str] = None):
        self.task = task
        self.max_steps = max_steps
        self.headless = headless
        self.enable_streaming = enable_streaming
        self.thread_id = thread_id  # For pushing updates to backend
        self.backend_url = backend_url or "http://localhost:8000"
        self.actions_taken = []
        self.screenshots = []
        self.downloads = []
        self.uploaded_files = []
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.llm_manager = LLMManager()
        self.task_id = str(uuid.uuid4())[:8]
        self.streaming_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Task planning and tracking
        self.task_plan: List[Dict[str, Any]] = []
        self.completed_subtasks: List[str] = []
        self.failed_subtasks: List[Dict[str, str]] = []
        self.plan_created = False
        self.consecutive_same_actions = 0
        self.last_action_type = None
        
        # Track page state for dynamic updates
        self.last_page_url = None
        self.last_page_hash = None
        
        # Vision modality tracking
        self.next_step_needs_vision = False  # Vision model decides this for next step
        
        # Performance metrics
        self.start_time = None
        self.metrics = {
            "total_time": 0,
            "navigation_time": 0,
            "action_time": 0,
            "llm_calls": 0,
            "page_loads": 0,
            "screenshots_taken": 0
        }
        
    async def __aenter__(self):
        """Initialize browser - ensures only one instance per task"""
        # Track this browser instance
        async with active_browsers_lock:
            active_browsers[self.task_id] = True
            logger.info(f"🌐 Initializing browser for task {self.task_id} (Active: {len(active_browsers)})")
        
        self.playwright = await async_playwright().start()
        
        # Launch browser with stealth configuration
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials'
            ],
            slow_mo=500 if not self.headless else 0
        )
        
        # Create context with stealth settings
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale='en-US',
            timezone_id='America/New_York',
            permissions=['geolocation'],
            geolocation={'latitude': 40.7128, 'longitude': -74.0060},  # NYC
            color_scheme='light',
            accept_downloads=True
        )
        
        # Set default timeout
        self.context.set_default_timeout(15000)
        
        # Create page and setup handlers
        self.page = await self.context.new_page()
        
        # Handle downloads
        self.page.on("download", lambda download: asyncio.create_task(self._handle_download(download)))
        
        # Handle popups/dialogs automatically
        self.page.on("dialog", lambda dialog: asyncio.create_task(dialog.accept()))
        
        # Handle console errors for debugging
        self.page.on("console", lambda msg: logger.debug(f"Console: {msg.text}") if msg.type == "error" else None)
        
        # Handle page errors (suppress common React/Pinterest errors)
        def handle_page_error(err):
            err_str = str(err)
            # Suppress common harmless errors
            if any(x in err_str.lower() for x in ['react', 'chunk', 'hydration', 'minified']):
                logger.debug(f"Page error (suppressed): {err_str[:100]}")
            else:
                logger.warning(f"Page error: {err_str[:200]}")
        
        self.page.on("pageerror", handle_page_error)
        
        # Log failed requests for debugging
        self.page.on("requestfailed", lambda req: logger.debug(f"Request failed: {req.url[:100]}"))
        
        # Inject stealth scripts to avoid detection
        await self.page.add_init_script("""
            // Overwrite the `navigator.webdriver` property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Overwrite the `plugins` property
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Overwrite the `languages` property
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
        """)
        
        logger.info(f"🌐 Browser launched ({'visible' if not self.headless else 'headless'} mode) - 1280x800")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup browser - ensures no orphaned browser instances"""
        try:
            if self.context:
                await self.context.close()
                logger.info("🔒 Browser context closed")
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
        
        try:
            if self.browser:
                await self.browser.close()
                logger.info("🔒 Browser closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        
        try:
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
                logger.info("🔒 Playwright stopped")
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
        
        # Remove from active browsers
        async with active_browsers_lock:
            if self.task_id in active_browsers:
                del active_browsers[self.task_id]
            logger.info(f"🔒 Browser cleanup complete (Active: {len(active_browsers)})")
    
    async def _handle_download(self, download):
        """Handle file downloads automatically"""
        try:
            # Generate unique filename
            filename = f"{self.task_id}_{download.suggested_filename}"
            filepath = DOWNLOADS_DIR / filename
            
            # Save the download
            await download.save_as(str(filepath))
            self.downloads.append(str(filepath))
            
            logger.info(f"📥 Downloaded file: {filename} ({download.suggested_filename})")
        except Exception as e:
            logger.warning(f"Failed to save download: {e}")
    
    async def capture_screenshot(self, name: str = "screenshot") -> str:
        """Capture and save screenshot with robust error handling"""
        filename = f"{self.task_id}_{name}_{len(self.screenshots)}.png"
        filepath = STORAGE_DIR / filename
        
        # Try multiple strategies to capture screenshot
        strategies = [
            # Strategy 1: Standard with animations disabled
            {"full_page": False, "timeout": 15000, "animations": "disabled"},
            # Strategy 2: Quick capture without waiting
            {"full_page": False, "timeout": 5000},
            # Strategy 3: Minimal capture
            {"timeout": 3000}
        ]
        
        for i, kwargs in enumerate(strategies):
            try:
                await self.page.screenshot(path=str(filepath), **kwargs)
                self.screenshots.append(str(filepath))
                logger.info(f"📸 Screenshot saved: {filename}")
                return str(filepath)
            except Exception as e:
                if i == len(strategies) - 1:  # Last attempt
                    logger.warning(f"⚠️  Screenshot failed after {len(strategies)} attempts, continuing without it")
                    return ""
                # Try next strategy
                continue
        
        return ""
    
    async def push_screenshot_to_backend(self, screenshot_base64: str):
        """Push screenshot AND task plan update directly to backend"""
        if not self.thread_id:
            return
        
        try:
            import httpx
            
            # Get current action from last action taken
            current_action = ""
            if self.actions_taken:
                last_action = self.actions_taken[-1]
                action_type = last_action.get('action', '')
                reasoning = last_action.get('reasoning', '')
                current_action = f"{action_type}: {reasoning[:80]}" if reasoning else action_type
            
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.backend_url}/api/canvas/update",
                    json={
                        "thread_id": self.thread_id,
                        "screenshot_data": screenshot_base64,
                        "url": self.page.url if self.page else "",
                        "step": len(self.actions_taken),
                        "task": self.task,
                        "task_plan": self.task_plan,  # Send task plan for plan view
                        "current_action": current_action  # Send current action
                    },
                    timeout=2.0
                )
        except Exception as e:
            logger.debug(f"Failed to push update to backend: {e}")
    
    async def stream_screenshots(self):
        """Background task to capture screenshots every second and push to backend"""
        screenshot_count = 0
        while self.is_running:
            try:
                if self.page and self.page.url != "about:blank":
                    try:
                        # Capture screenshot to bytes (in-memory, not saved to disk)
                        # Try with multiple timeouts
                        screenshot_bytes = None
                        for timeout_val in [10000, 5000, 3000]:
                            try:
                                screenshot_bytes = await self.page.screenshot(
                                    full_page=False, 
                                    timeout=timeout_val,
                                    animations="disabled" if timeout_val > 5000 else None
                                )
                                break
                            except:
                                if timeout_val == 3000:
                                    raise
                                continue
                        
                        if not screenshot_bytes:
                            continue  # Skip this streaming update
                        
                        # Convert to base64 for transmission
                        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                        
                        # Update live screenshots dictionary (in-memory only)
                        async with live_screenshots_lock:
                            live_screenshots[self.task_id] = {
                                "screenshot_data": screenshot_base64,
                                "url": self.page.url,
                                "timestamp": datetime.now().isoformat(),
                                "step": len(self.actions_taken),
                                "task": self.task
                            }
                        
                        # Push to backend if thread_id is provided
                        if self.thread_id:
                            await self.push_screenshot_to_backend(screenshot_base64)
                        
                        screenshot_count += 1
                        if screenshot_count % 10 == 0:  # Log every 10 screenshots
                            logger.info(f"📹 Live streaming: {screenshot_count} screenshots")
                    except Exception as screenshot_error:
                        logger.debug(f"Screenshot capture failed: {screenshot_error}")
                
                await asyncio.sleep(1)  # Capture every second
                
            except Exception as e:
                logger.debug(f"Screenshot streaming error: {e}")
                await asyncio.sleep(1)
    
    async def create_task_plan(self) -> List[Dict[str, Any]]:
        """Create an initial plan by breaking down the task into subtasks"""
        
        planning_prompt = f"""Analyze this browser automation task and break it down into clear, actionable subtasks.

TASK: {self.task}

Break this down into a list of subtasks. Each subtask should be:
1. A SPECIFIC action that can be done with navigate/click/type/extract actions
2. Independent and verifiable
3. In logical order

Respond with ONLY a JSON array of subtasks:
[
    {{"subtask": "Navigate to google.com", "status": "pending"}},
    {{"subtask": "Search for artificial intelligence", "status": "pending"}},
    {{"subtask": "Extract search results", "status": "pending"}}
]

Rules:
- DO NOT include generic subtasks like "Launch browser", "Wait for page", "Parse data"
- DO NOT include "Click search button" - most sites auto-submit on Enter (Wikipedia, Google, etc.)
- Each subtask should be HIGH-LEVEL (e.g., "Get LeetCode contests" not "Navigate to LeetCode", "Click contests", "Extract data")
- Combine multiple steps into one subtask (e.g., "Extract contests from LeetCode" includes navigate + extract)
- If task mentions multiple sites, create ONE subtask per site
- Keep subtasks specific and actionable
- MAXIMUM 3-4 subtasks total (be concise!)
- Focus on WHAT to achieve, not HOW to do it
- Example: For "Get contests from 3 sites" → 3 subtasks: "Get LeetCode contests", "Get CodeChef contests", "Get Codeforces contests"
"""

        try:
            content, provider = self.llm_manager.get_completion(
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.2,
                max_tokens=500
            )
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                logger.info(f"📋 Created task plan with {len(plan)} subtasks")
                for i, subtask in enumerate(plan, 1):
                    logger.info(f"   {i}. {subtask.get('subtask', 'Unknown')}")
                return plan
            else:
                logger.warning("Could not parse task plan, using single-task mode")
                return [{"subtask": self.task, "status": "pending"}]
                
        except Exception as e:
            logger.error(f"Error creating task plan: {e}")
            return [{"subtask": self.task, "status": "pending"}]
    
    async def get_page_content(self) -> Dict[str, Any]:
        """Extract comprehensive page content with full DOM structure and live updates"""
        
        # Ensure page is ready
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except:
            logger.debug("Page still loading, continuing anyway")
        
        # Check for iframes and log them
        try:
            frames = self.page.frames
            if len(frames) > 1:
                logger.info(f"🖼️  Detected {len(frames)} frames (including main)")
                # Note: For now we focus on main frame, but we're aware of iframes
        except:
            pass
        
        # Get complete page state including all interactive elements with precise selectors
        page_data = await self.page.evaluate("""
            () => {
                // Helper to check if element is truly visible and interactive
                function isVisible(el) {
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' && 
                           style.visibility !== 'hidden' && 
                           style.opacity !== '0' &&
                           rect.width > 0 && 
                           rect.height > 0 &&
                           rect.top < window.innerHeight &&
                           rect.left < window.innerWidth;
                }
                
                // Helper to generate reliable CSS selector
                function getSelector(el) {
                    // Priority: ID > Name > Aria-label > Data attributes > Placeholder > Type+Name combo
                    if (el.id && !el.id.match(/^[\\d]/) && document.querySelectorAll(`#${el.id}`).length === 1) {
                        return `#${el.id}`;
                    }
                    
                    if (el.name && document.querySelectorAll(`[name="${el.name}"]`).length <= 3) {
                        return `[name="${el.name}"]`;
                    }
                    
                    const ariaLabel = el.getAttribute('aria-label');
                    if (ariaLabel && document.querySelectorAll(`[aria-label="${ariaLabel}"]`).length <= 3) {
                        return `[aria-label="${ariaLabel}"]`;
                    }
                    
                    // Try data attributes
                    const dataTestId = el.getAttribute('data-test-id') || el.getAttribute('data-testid');
                    if (dataTestId) {
                        return `[data-test-id="${dataTestId}"], [data-testid="${dataTestId}"]`;
                    }
                    
                    // For inputs with placeholder
                    if (el.placeholder && el.tagName.toLowerCase() === 'input') {
                        return `input[placeholder="${el.placeholder}"]`;
                    }
                    
                    // Combination selector
                    let path = el.tagName.toLowerCase();
                    if (el.type) path += `[type="${el.type}"]`;
                    if (el.name) path += `[name="${el.name}"]`;
                    
                    // Only add classes if they're stable (not random hashes)
                    if (el.className && typeof el.className === 'string') {
                        const classes = el.className.trim().split(/\\s+/)
                            .filter(c => c && !c.match(/^[\\d]/) && !c.match(/[a-f0-9]{6,}/i) && c.length < 20)
                            .slice(0, 2);
                        if (classes.length > 0) path += '.' + classes.join('.');
                    }
                    
                    return path;
                }
                
                // Extract ALL interactive elements with complete details
                const interactiveElements = [];
                const selectors = [
                    'input:not([type="hidden"])', 'button', 'a[href]', 'select', 'textarea',
                    '[role="button"]', '[role="link"]', '[role="textbox"]', '[role="searchbox"]',
                    '[onclick]', '[contenteditable="true"]', 'label'
                ];
                
                const allElements = document.querySelectorAll(selectors.join(','));
                
                allElements.forEach((el, idx) => {
                    if (isVisible(el)) {
                        const rect = el.getBoundingClientRect();
                        const elementData = {
                            index: idx,
                            tag: el.tagName.toLowerCase(),
                            type: el.type || el.getAttribute('type') || '',
                            text: (el.innerText || el.textContent || '').trim().substring(0, 200),
                            value: el.value || '',
                            placeholder: el.placeholder || '',
                            name: el.name || '',
                            id: el.id || '',
                            ariaLabel: el.getAttribute('aria-label') || '',
                            role: el.getAttribute('role') || '',
                            href: el.href || '',
                            selector: getSelector(el),
                            classes: el.className || '',
                            position: {
                                x: Math.round(rect.x),
                                y: Math.round(rect.y),
                                width: Math.round(rect.width),
                                height: Math.round(rect.height),
                                inViewport: rect.top >= 0 && rect.top <= window.innerHeight
                            },
                            isInput: ['input', 'textarea'].includes(el.tagName.toLowerCase()),
                            isButton: el.tagName.toLowerCase() === 'button' || el.type === 'button' || el.type === 'submit',
                            isLink: el.tagName.toLowerCase() === 'a',
                            isDisabled: el.disabled || el.getAttribute('aria-disabled') === 'true'
                        };
                        
                        // Add description for better understanding
                        if (elementData.isInput) {
                            elementData.description = `${elementData.type || 'text'} input` + 
                                (elementData.placeholder ? ` (${elementData.placeholder})` : '') +
                                (elementData.name ? ` [${elementData.name}]` : '');
                        } else if (elementData.isButton) {
                            elementData.description = `Button: ${elementData.text || elementData.ariaLabel || 'unnamed'}`;
                        } else if (elementData.isLink) {
                            elementData.description = `Link: ${elementData.text || 'unnamed'} -> ${elementData.href}`;
                        }
                        
                        interactiveElements.push(elementData);
                    }
                });
                
                // Get comprehensive page text
                const bodyText = document.body.innerText || '';
                
                // Get page structure
                const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
                    .filter(isVisible)
                    .map(h => `${h.tagName}: ${h.innerText.trim().substring(0, 100)}`);
                
                // Get forms info
                const forms = Array.from(document.querySelectorAll('form'))
                    .filter(isVisible)
                    .map(f => ({
                        action: f.action,
                        method: f.method,
                        inputs: Array.from(f.querySelectorAll('input, textarea, select')).length
                    }));
                
                return {
                    url: window.location.href,
                    title: document.title,
                    bodyText: bodyText.substring(0, 3000),
                    interactiveElements: interactiveElements,
                    headings: headings,
                    forms: forms,
                    elementCount: {
                        inputs: interactiveElements.filter(e => e.isInput && !e.isDisabled).length,
                        buttons: interactiveElements.filter(e => e.isButton && !e.isDisabled).length,
                        links: interactiveElements.filter(e => e.isLink).length,
                        total: interactiveElements.length
                    },
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight,
                        scrollY: window.scrollY
                    }
                };
            }
        """)
        
        return page_data
    
    def plan_next_action(self, page_content: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        """Use LLM with fallback to plan next action"""
        
        # Build task plan status
        plan_status = ""
        if self.task_plan:
            plan_status = "\n\nTASK PLAN:\n"
            for i, subtask in enumerate(self.task_plan, 1):
                status_icon = "✅" if subtask['status'] == 'completed' else "⏳"
                plan_status += f"{status_icon} {i}. {subtask['subtask']} [{subtask['status']}]\n"
            
            pending_count = sum(1 for t in self.task_plan if t['status'] == 'pending')
            plan_status += f"\nPending subtasks: {pending_count}/{len(self.task_plan)}"
        
        # Build context-aware prompt with comprehensive page data
        elements_summary = page_content.get('elementCount', {})
        interactive_elements = page_content.get('interactiveElements', [])
        
        # Prioritize elements: inputs first, then buttons, then links
        # Also prioritize elements in viewport
        inputs = sorted(
            [e for e in interactive_elements if e.get('isInput') and not e.get('isDisabled')],
            key=lambda x: (not x.get('position', {}).get('inViewport', False), x.get('index', 999))
        )[:10]
        
        buttons = sorted(
            [e for e in interactive_elements if e.get('isButton') and not e.get('isDisabled')],
            key=lambda x: (not x.get('position', {}).get('inViewport', False), x.get('index', 999))
        )[:10]
        
        links = sorted(
            [e for e in interactive_elements if e.get('isLink')],
            key=lambda x: (not x.get('position', {}).get('inViewport', False), x.get('index', 999))
        )[:15]
        
        prompt = f"""You are an expert browser automation agent. Analyze the COMPLETE page state and decide the next action.

OVERALL TASK: {self.task}
{plan_status}

CURRENT PAGE STATE:
URL: {page_content.get('url')}
Title: {page_content.get('title')}

PAGE STRUCTURE:
{chr(10).join(page_content.get('headings', [])[:5])}

PAGE TEXT (first 1500 chars):
{page_content.get('bodyText', '')[:1500]}

AVAILABLE INTERACTIVE ELEMENTS ({elements_summary.get('total', 0)} total):

INPUT FIELDS ({elements_summary.get('inputs', 0)} available):
{json.dumps([{'selector': e['selector'], 'type': e['type'], 'placeholder': e['placeholder'], 'name': e['name'], 'ariaLabel': e['ariaLabel']} for e in inputs], indent=2) if inputs else 'None'}

BUTTONS ({elements_summary.get('buttons', 0)} available):
{json.dumps([{'selector': e['selector'], 'text': e['text'], 'ariaLabel': e['ariaLabel']} for e in buttons], indent=2) if buttons else 'None'}

LINKS ({elements_summary.get('links', 0)} available, showing first 15):
{json.dumps([{'text': e['text'][:50], 'href': e['href'][:100]} for e in links], indent=2) if links else 'None'}

FORMS ON PAGE:
{json.dumps(page_content.get('forms', []), indent=2) if page_content.get('forms') else 'None'}

ALL ACTIONS TAKEN SO FAR:
{json.dumps(self.actions_taken[-5:], indent=2) if self.actions_taken else 'None'}

FAILED SUBTASKS:
{json.dumps(self.failed_subtasks, indent=2) if self.failed_subtasks else 'None'}

PROGRESS: Step {step_num}/{self.max_steps}
WARNING: If you have tried the same action 3+ times without progress, SKIP to next subtask or mark it as failed!

Decide the next action. Respond with ONLY a valid JSON object (no markdown, no explanation):
{{
    "action": "navigate|click|type|scroll|extract|upload|download|skip|done",
    "reasoning": "brief explanation",
    "params": {{
        "url": "full URL for navigate",
        "text": "exact text to click (if no selector)",
        "input_text": "text to type",
        "selector": "USE THE EXACT SELECTOR from the elements above (e.g., #searchInput, [name='q'])",
        "file_path": "path to file for upload action"
    }},
    "completed_subtask": "name of subtask just completed (if any)",
    "failed_subtask": "name of subtask to skip/fail (if blocked/stuck)",
    "new_subtask": "NEW subtask to add if you discover additional work needed (optional)",
    "is_complete": true/false,
    "result_summary": "summary of ALL findings if ALL subtasks complete"
}}

CRITICAL SELECTOR USAGE:
- ALWAYS use the EXACT "selector" field from the INPUT FIELDS, BUTTONS, or LINKS list above
- For typing: Copy the exact selector from INPUT FIELDS (e.g., {{"selector": "[name='searchBoxInput']", "input_text": "query"}})
- For clicking buttons: Copy the exact selector from BUTTONS (e.g., {{"selector": "button.search-btn"}})
- For clicking links: Use the "text" field from LINKS (e.g., {{"text": "Explore"}}) - DO NOT use href as selector
- For upload: Use {{"action": "upload", "params": {{"file_path": "filename.pdf", "selector": "input[type='file']"}}}}
- For download: Use {{"action": "download", "params": {{"selector": "download-button-selector"}}}}
- NEVER make up selectors - ONLY use selectors provided in the lists above
- If you need to click a link, use the "text" field, NOT the href
- If a selector is not in the list, use text-based clicking with the visible text

AVAILABLE ACTIONS (YOU MUST USE EXACTLY ONE OF THESE):
- "navigate": Go to a URL (provide full URL in params.url)
- "click": Click an element (provide text in params.text or selector)
- "type": Type text into input field (provide text in params.input_text and selector)
- "scroll": Scroll down the page
- "extract": Extract data from current page (marks subtask as completed)
- "analyze_images": Use vision to analyze images on the page and make decisions (use when task requires looking at images)
- "save_images": Save images from the page (vision will identify and save images with descriptions)
- "upload": Upload a file (provide file_path and selector for file input)
- "download": Download a file (provide selector or text for download button)
- "skip": Skip current subtask if blocked/stuck
- "done": Mark entire task as complete (ONLY when ALL subtasks are done)

INVALID ACTIONS (DO NOT USE): "parse", "report", "wait", or any other action not listed above!

RULES:
1. If URL is "about:blank", use navigate action first
2. Check TASK PLAN - work on pending subtasks in order
3. PREFER navigate over click - if you know the URL, navigate directly
4. When you complete a subtask action (navigate somewhere, type something, extract data), set completed_subtask
5. After completing a subtask, move to the NEXT pending subtask
6. If stuck or blocked, use skip action and set failed_subtask
7. Use done action ONLY when ALL subtasks are completed or failed
8. DO NOT repeat the same action more than twice - try something different!
9. You CAN add new subtasks mid-way if you discover additional work needed (use "new_subtask" field)

WHEN TO MARK SUBTASK COMPLETE:
- "Navigate to X" - complete after navigate action succeeds
- "Search for X" - complete after type action succeeds
- "Extract X" - complete after extract action succeeds
- "Click X" - complete after click action succeeds

CRITICAL: 
- After completing a subtask, immediately move to the NEXT pending subtask
- If no pending subtasks remain, use done action with is_complete=true
- Check RECENT ACTIONS - if repeating same action, you are stuck! Try different approach or skip
- Do not overthink - if you did the action, mark the subtask complete and move on!"""

        try:
            # Use LLM manager with automatic fallback
            content, provider = self.llm_manager.get_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                action_plan = json.loads(json_match.group())
                logger.info(f"✅ Action planned by {provider.upper()}")
                return action_plan
            else:
                logger.warning(f"No JSON found in response: {content[:200]}")
                return {"action": "done", "is_complete": True, "result_summary": "Could not parse action"}
                
        except Exception as e:
            logger.error(f"❌ All LLM providers failed: {e}")
            return {"action": "done", "is_complete": True, "result_summary": f"LLM Error: {e}"}
    
    async def should_use_vision(self, page_content: Dict[str, Any]) -> bool:
        """Decide if vision is needed for the next step - ONLY when absolutely necessary"""
        if not VISION_ENABLED:
            return False
        
        # If previous vision call decided modality for this step, use that
        if hasattr(self, 'next_step_needs_vision') and self.next_step_needs_vision:
            logger.info("🎨 Vision needed (decided by previous vision call)")
            self.next_step_needs_vision = False  # Reset for next time
            return True
        
        # Check current subtask to see if it requires vision
        current_subtask = None
        if self.task_plan:
            for subtask in self.task_plan:
                if subtask['status'] == 'pending':
                    current_subtask = subtask['subtask'].lower()
                    break
        
        # Vision ONLY needed if current subtask explicitly requires image analysis
        if current_subtask:
            vision_subtask_keywords = [
                'look at', 'analyze', 'describe', 'pick', 'choose', 'select',
                'evaluate', 'compare', 'find the best', 'which is better'
            ]
            
            needs_vision_for_subtask = any(kw in current_subtask for kw in vision_subtask_keywords)
            
            if needs_vision_for_subtask and any(word in current_subtask for word in ['image', 'photo', 'picture', 'visual']):
                logger.info(f"🎨 Vision needed (current subtask requires image analysis: {current_subtask})")
                return True
        
        # Otherwise, use heuristics to detect if vision is needed
        url = page_content.get('url', '')
        title = page_content.get('title', '').lower()
        body_text = page_content.get('bodyText', '').lower()
        
        # Skip vision for about:blank or empty pages
        if url == 'about:blank' or not body_text:
            return False
        
        # Vision ONLY needed for:
        # 1. Cloudflare/CAPTCHA challenges (blocking navigation)
        # 2. When NO interactive elements detected (page broken/unusual)
        
        element_count = page_content.get('elementCount', {}).get('total', 0)
        
        vision_indicators = [
            'cloudflare' in title or ('cloudflare' in body_text and 'challenge' in body_text),
            'captcha' in body_text or 'recaptcha' in body_text,
            'verify you are human' in body_text,
            'security check' in body_text and 'cloudflare' in body_text,
            element_count == 0,  # NO elements detected at all
        ]
        
        needs_vision = any(vision_indicators)
        
        if needs_vision:
            logger.info("🎨 Vision needed (detected challenge or no elements)")
        
        return needs_vision
    
    async def plan_action_with_vision(self, screenshot_base64: str, page_content: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        """Single vision API call with fallback chain: Ollama → NVIDIA → Text-only"""
        
        # Build task plan status
        plan_status = ""
        if self.task_plan:
            plan_status = "\n\nTASK PLAN:\n"
            for i, subtask in enumerate(self.task_plan, 1):
                status_icon = "✅" if subtask['status'] == 'completed' else "⏳"
                plan_status += f"{status_icon} {i}. {subtask['subtask']} [{subtask['status']}]\n"
        
        # Try Ollama Qwen3-VL first
        if OLLAMA_API_KEY:
            try:
                logger.info("🎨 Trying vision: Ollama Qwen3-VL")
                result = await self._call_vision_model_ollama(screenshot_base64, page_content, step_num, plan_status)
                if result:
                    logger.info("✅ Vision success: Ollama Qwen3-VL")
                    return result
            except Exception as e:
                logger.warning(f"⚠️  Ollama vision failed: {e}")
        
        # Fallback to NVIDIA Mistral vision
        if NVIDIA_API_KEY:
            try:
                logger.info("🎨 Trying vision fallback: NVIDIA Mistral")
                result = await self._call_vision_model_nvidia(screenshot_base64, page_content, step_num, plan_status)
                if result:
                    logger.info("✅ Vision success: NVIDIA Mistral")
                    return result
            except Exception as e:
                logger.warning(f"⚠️  NVIDIA vision failed: {e}")
        
        # Final fallback: text-only (no vision)
        logger.warning("⚠️  All vision models failed, falling back to text-only")
        return None
    
    async def _call_vision_model_ollama(self, screenshot_base64: str, page_content: Dict[str, Any], step_num: int, plan_status: str) -> Dict[str, Any]:
        """Call Ollama Qwen3-VL vision model"""
        try:
            # Check if task requires image analysis
            task_needs_image_analysis = any(kw in self.task.lower() for kw in ['look at', 'analyze', 'describe', 'pick', 'choose', 'image', 'photo', 'picture'])
            
            image_analysis_instruction = ""
            if task_needs_image_analysis:
                image_analysis_instruction = """
CRITICAL: This task requires IMAGE ANALYSIS!
- Look carefully at ALL images visible in the screenshot
- Use action "analyze_images" 
- In "reasoning", provide DETAILED descriptions of what you see:
  * Describe each image separately
  * Include: colors, objects, style, composition, text visible
  * Explain what makes each image interesting or notable
  * Be specific and detailed (at least 2-3 sentences per image)
"""
            
            prompt = f"""{image_analysis_instruction}

Analyze this webpage screenshot and provide a COMPLETE response with:
1. Bounding boxes of UI elements
2. Next action with coordinates
3. Whether NEXT step needs vision or text-only
4. Plan updates if needed

TASK: {self.task}
{plan_status}

CURRENT PAGE:
URL: {page_content.get('url')}
Title: {page_content.get('title')}
STEP: {step_num}/{self.max_steps}

Respond with ONLY a valid JSON object:
{{
    "bounding_boxes": [
        {{
            "type": "button|input|checkbox|link|dropdown",
            "bbox": {{"x": 100, "y": 200, "width": 80, "height": 40}},
            "text": "visible text",
            "purpose": "what it does"
        }}
    ],
    "action": "click|type|drag|hover|scroll_to|double_click|right_click|extract|navigate|done",
    "reasoning": "what you see and why this action",
    "params": {{
        "x": 100,
        "y": 200,
        "x2": 300,
        "y2": 400,
        "text": "text to type",
        "url": "url to navigate"
    }},
    "completed_subtask": "name of subtask if completed",
    "new_subtask": "new subtask to add if discovered additional work",
    "is_complete": false,
    "next_step_needs_vision": true|false,
    "next_step_reasoning": "why vision is/isn't needed for next step"
}}

MODALITY DECISION (next_step_needs_vision):
- true: If next page will have CAPTCHA, challenge, complex visual layout, or few DOM elements
- false: If next page will be standard HTML with good selectors (most websites)

PLAN UPDATES:
- Use "new_subtask" if you discover additional work needed
- Use "completed_subtask" when you finish a subtask

ACTIONS:
- click: Click at (x, y) - buttons, checkboxes, links
- type: Type text - provide x, y to click first, then text
- drag: Drag from (x, y) to (x2, y2) - sliders, drag-drop
- hover: Hover at (x, y) - dropdowns, tooltips
- scroll_to: Scroll to (x, y)
- extract: Extract visible data
- navigate: Go to URL
- done: Task complete

EXAMPLES:
Cloudflare: {{"action": "click", "params": {{"x": 50, "y": 300}}, "next_step_needs_vision": false, "reasoning": "Click checkbox, next page will be normal"}}
Search: {{"action": "type", "params": {{"x": 100, "y": 200, "text": "query"}}, "next_step_needs_vision": false}}
Slider: {{"action": "drag", "params": {{"x": 50, "y": 300, "x2": 200, "y2": 300}}, "next_step_needs_vision": false}}"""

            # Call Ollama vision API
            client = OpenAI(
                api_key=OLLAMA_API_KEY,
                base_url="https://ollama.com/v1"
            )
            
            response = client.chat.completions.create(
                model="qwen3-vl:235b-cloud",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            }
                        }
                    ]
                }],
                temperature=0.1,
                max_tokens=2000  # Increased for bounding boxes + action
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Log bounding boxes detected
                bboxes = result.get('bounding_boxes', [])
                if bboxes:
                    logger.info(f"✅ Detected {len(bboxes)} UI elements with bounding boxes")
                
                # Log modality decision for next step
                next_needs_vision = result.get('next_step_needs_vision', False)
                next_reasoning = result.get('next_step_reasoning', '')
                if next_needs_vision:
                    logger.info(f"🎨 Next step will use VISION: {next_reasoning}")
                else:
                    logger.info(f"📝 Next step will use TEXT-ONLY: {next_reasoning}")
                
                # Store modality decision for next iteration
                self.next_step_needs_vision = next_needs_vision
                
                # Extract action plan with all fields
                action_plan = {
                    'action': result.get('action'),
                    'reasoning': result.get('reasoning'),
                    'params': result.get('params', {}),
                    'completed_subtask': result.get('completed_subtask'),
                    'new_subtask': result.get('new_subtask'),
                    'is_complete': result.get('is_complete', False)
                }
                
                # Add confidence and bbox to params for multi-strategy retry
                action_plan['params']['confidence'] = result.get('confidence', 0.7)
                
                # Find relevant bbox for the action
                bboxes = result.get('bounding_boxes', [])
                if bboxes and action_plan['action'] in ['click', 'type', 'hover']:
                    # Use first bbox (most relevant)
                    action_plan['params']['bbox'] = bboxes[0].get('bbox')
                
                logger.info(f"✅ Ollama vision action: {action_plan.get('action')} (confidence: {action_plan['params']['confidence']:.2f})")
                return action_plan
            else:
                logger.warning("Ollama vision didn't return valid JSON")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ollama vision failed: {e}")
            raise  # Re-raise to trigger fallback
    
    async def _call_vision_model_nvidia(self, screenshot_base64: str, page_content: Dict[str, Any], step_num: int, plan_status: str) -> Dict[str, Any]:
        """Call NVIDIA Mistral vision model as fallback"""
        try:
            # Check if task requires image analysis
            task_needs_image_analysis = any(kw in self.task.lower() for kw in ['look at', 'analyze', 'describe', 'pick', 'choose', 'image', 'photo', 'picture'])
            
            image_analysis_instruction = ""
            if task_needs_image_analysis:
                image_analysis_instruction = """
CRITICAL: This task requires IMAGE ANALYSIS!
- Look carefully at ALL images visible in the screenshot
- If task asks to SAVE images, use action "save_images" (not analyze_images)
- If task asks to DESCRIBE/ANALYZE images, use action "analyze_images"
- In "reasoning", provide DETAILED descriptions of what you see:
  * Describe each image separately (Image 1, Image 2, Image 3)
  * Include: car model/type, color, style, design features, condition
  * Explain what makes each image appealing or notable
  * Be specific and detailed (at least 2-3 sentences per image)
  * Example: "Image 1: A pristine 1967 Ford Mustang GT in classic red with white racing stripes..."
"""
            
            prompt = f"""{image_analysis_instruction}

Analyze this webpage screenshot and provide a COMPLETE response with:
1. Bounding boxes of UI elements
2. Next action with coordinates
3. Whether NEXT step needs vision or text-only
4. Plan updates if needed

TASK: {self.task}
{plan_status}

CURRENT PAGE:
URL: {page_content.get('url')}
Title: {page_content.get('title')}
STEP: {step_num}/{self.max_steps}

Respond with ONLY a valid JSON object:
{{
    "bounding_boxes": [
        {{
            "type": "button|input|checkbox|link|dropdown",
            "bbox": {{"x": 100, "y": 200, "width": 80, "height": 40}},
            "text": "visible text",
            "purpose": "what it does"
        }}
    ],
    "action": "click|type|drag|hover|scroll_to|double_click|right_click|extract|navigate|done",
    "reasoning": "what you see and why this action",
    "params": {{
        "x": 100,
        "y": 200,
        "x2": 300,
        "y2": 400,
        "text": "text to type",
        "url": "url to navigate"
    }},
    "completed_subtask": "name of subtask if completed",
    "new_subtask": "new subtask to add if discovered additional work",
    "is_complete": false,
    "next_step_needs_vision": true|false,
    "next_step_reasoning": "why vision is/isn't needed for next step"
}}

MODALITY DECISION (next_step_needs_vision):
- true: If next page will have CAPTCHA, challenge, complex visual layout, or few DOM elements
- false: If next page will be standard HTML with good selectors (most websites)

ACTIONS:
- click: Click at (x, y) - buttons, checkboxes, links
- type: Type text - provide x, y to click first, then text
- drag: Drag from (x, y) to (x2, y2) - sliders, drag-drop
- hover: Hover at (x, y) - dropdowns, tooltips
- scroll_to: Scroll to (x, y)
- extract: Extract visible data
- analyze_images: Analyze images and describe them (put descriptions in reasoning)
- save_images: Save images from page (put descriptions in reasoning)
- navigate: Go to URL
- done: Task complete"""

            # Call NVIDIA vision API
            client = OpenAI(
                api_key=NVIDIA_API_KEY,
                base_url="https://integrate.api.nvidia.com/v1"
            )
            
            response = client.chat.completions.create(
                model="mistralai/mistral-medium-3-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            }
                        }
                    ]
                }],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Log bounding boxes detected
                bboxes = result.get('bounding_boxes', [])
                if bboxes:
                    logger.info(f"✅ Detected {len(bboxes)} UI elements with bounding boxes")
                
                # Log modality decision for next step
                next_needs_vision = result.get('next_step_needs_vision', False)
                next_reasoning = result.get('next_step_reasoning', '')
                if next_needs_vision:
                    logger.info(f"🎨 Next step will use VISION: {next_reasoning}")
                else:
                    logger.info(f"📝 Next step will use TEXT-ONLY: {next_reasoning}")
                
                # Store modality decision for next iteration
                self.next_step_needs_vision = next_needs_vision
                
                # Extract action plan with all fields
                action_plan = {
                    'action': result.get('action'),
                    'reasoning': result.get('reasoning'),
                    'params': result.get('params', {}),
                    'completed_subtask': result.get('completed_subtask'),
                    'new_subtask': result.get('new_subtask'),
                    'is_complete': result.get('is_complete', False)
                }
                
                # Add confidence and bbox to params for multi-strategy retry
                action_plan['params']['confidence'] = result.get('confidence', 0.7)
                
                # Find relevant bbox for the action
                bboxes = result.get('bounding_boxes', [])
                if bboxes and action_plan['action'] in ['click', 'type', 'hover']:
                    # Use first bbox (most relevant)
                    action_plan['params']['bbox'] = bboxes[0].get('bbox')
                
                logger.info(f"✅ NVIDIA vision action: {action_plan.get('action')} (confidence: {action_plan['params']['confidence']:.2f})")
                return action_plan
            else:
                logger.warning("NVIDIA vision didn't return valid JSON")
                return None
                
        except Exception as e:
            logger.error(f"❌ NVIDIA vision failed: {e}")
            raise  # Re-raise to trigger final fallback
    
    async def map_bbox_to_dom_selector(self, bbox: Dict[str, Any]) -> Optional[str]:
        """Map bounding box to DOM selector using coordinates"""
        try:
            x = bbox.get('x', 0) + bbox.get('width', 0) // 2  # Center of bbox
            y = bbox.get('y', 0) + bbox.get('height', 0) // 2
            
            # Get element at bbox center and generate selector
            selector_info = await self.page.evaluate(f"""
                () => {{
                    const element = document.elementFromPoint({x}, {y});
                    if (!element) return null;
                    
                    // Generate unique selector
                    let selector = '';
                    
                    // Try ID first (most specific)
                    if (element.id) {{
                        selector = '#' + element.id;
                    }}
                    // Try name attribute
                    else if (element.name) {{
                        selector = element.tagName.toLowerCase() + '[name="' + element.name + '"]';
                    }}
                    // Try aria-label
                    else if (element.getAttribute('aria-label')) {{
                        selector = element.tagName.toLowerCase() + '[aria-label="' + element.getAttribute('aria-label') + '"]';
                    }}
                    // Try class combination
                    else if (element.className && typeof element.className === 'string') {{
                        const classes = element.className.trim().split(/\\s+/).slice(0, 2).join('.');
                        if (classes) {{
                            selector = element.tagName.toLowerCase() + '.' + classes;
                        }}
                    }}
                    // Fallback to tag + text
                    else {{
                        const text = element.textContent?.trim().substring(0, 30);
                        if (text) {{
                            selector = element.tagName.toLowerCase() + ':has-text("' + text + '")';
                        }} else {{
                            selector = element.tagName.toLowerCase();
                        }}
                    }}
                    
                    return {{
                        selector: selector,
                        tagName: element.tagName.toLowerCase(),
                        text: element.textContent?.trim().substring(0, 50) || '',
                        isClickable: element.tagName.toLowerCase() === 'button' || 
                                    element.tagName.toLowerCase() === 'a' ||
                                    element.tagName.toLowerCase() === 'input' ||
                                    element.getAttribute('role') === 'button' ||
                                    element.onclick !== null,
                        isVisible: element.offsetParent !== null
                    }};
                }}
            """)
            
            if selector_info and selector_info['selector'] and selector_info['isVisible']:
                logger.info(f"✅ Mapped bbox to selector: {selector_info['selector']} ({selector_info['tagName']})")
                return selector_info['selector']
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️  Bbox to DOM mapping failed: {e}")
            return None
    
    async def verify_coordinates(self, x: int, y: int, expected_element_type: str = None) -> Dict[str, Any]:
        """Verify coordinates and return element info with confidence"""
        try:
            # Get element at coordinates using JavaScript
            element_info = await self.page.evaluate(f"""
                () => {{
                    const element = document.elementFromPoint({x}, {y});
                    if (!element) return null;
                    
                    return {{
                        tagName: element.tagName.toLowerCase(),
                        type: element.type || '',
                        role: element.getAttribute('role') || '',
                        ariaLabel: element.getAttribute('aria-label') || '',
                        text: element.textContent?.trim().substring(0, 50) || '',
                        isClickable: element.tagName.toLowerCase() === 'button' || 
                                    element.tagName.toLowerCase() === 'a' ||
                                    element.tagName.toLowerCase() === 'input' ||
                                    element.getAttribute('role') === 'button' ||
                                    element.onclick !== null ||
                                    window.getComputedStyle(element).cursor === 'pointer',
                        isVisible: element.offsetParent !== null,
                        rect: element.getBoundingClientRect()
                    }};
                }}
            """)
            
            if not element_info:
                logger.warning(f"⚠️  No element found at coordinates ({x}, {y})")
                return {'valid': False, 'confidence': 0.0}
            
            # Calculate confidence score
            confidence = 0.0
            if element_info['isVisible']:
                confidence += 0.4
            if element_info['isClickable']:
                confidence += 0.4
            if element_info['text']:
                confidence += 0.1
            if element_info['ariaLabel'] or element_info['role']:
                confidence += 0.1
            
            element_info['valid'] = element_info['isVisible'] and element_info['isClickable']
            element_info['confidence'] = confidence
            
            if element_info['valid']:
                logger.info(f"✅ Verified ({x}, {y}): {element_info['tagName']} - '{element_info['text']}' (confidence: {confidence:.2f})")
            else:
                logger.warning(f"⚠️  Invalid element at ({x}, {y}): visible={element_info['isVisible']}, clickable={element_info['isClickable']}")
            
            return element_info
            
        except Exception as e:
            logger.warning(f"⚠️  Coordinate verification failed: {e}")
            return {'valid': False, 'confidence': 0.0}
    
    async def verify_action_success(self, action: str, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        """Programmatically verify if action succeeded (no API calls)"""
        try:
            verification = {
                'success': False,
                'confidence': 0.0,
                'changes_detected': [],
                'reason': ''
            }
            
            # 1. Visual change detection (screenshot hash)
            if before_state.get('screenshot') and after_state.get('screenshot'):
                import hashlib
                before_hash = hashlib.md5(before_state['screenshot']).hexdigest()
                after_hash = hashlib.md5(after_state['screenshot']).hexdigest()
                
                if before_hash != after_hash:
                    verification['changes_detected'].append('visual_change')
                    verification['confidence'] += 0.3
            
            # 2. URL change detection
            if before_state.get('url') != after_state.get('url'):
                verification['changes_detected'].append('url_change')
                verification['confidence'] += 0.4
                verification['success'] = True
            
            # 3. DOM change detection
            if before_state.get('dom_hash') != after_state.get('dom_hash'):
                verification['changes_detected'].append('dom_change')
                verification['confidence'] += 0.2
            
            # 4. Page title change
            if before_state.get('title') != after_state.get('title'):
                verification['changes_detected'].append('title_change')
                verification['confidence'] += 0.1
            
            # Determine success
            if verification['confidence'] >= 0.3:
                verification['success'] = True
                verification['reason'] = f"Detected: {', '.join(verification['changes_detected'])}"
            else:
                verification['reason'] = "No significant changes detected"
            
            if verification['success']:
                logger.info(f"✅ Action '{action}' verified: {verification['reason']} (confidence: {verification['confidence']:.2f})")
            else:
                logger.warning(f"⚠️  Action '{action}' may have failed: {verification['reason']}")
            
            return verification
            
        except Exception as e:
            logger.warning(f"⚠️  Action verification failed: {e}")
            return {'success': True, 'confidence': 0.5, 'changes_detected': [], 'reason': 'Verification error'}
    
    async def capture_page_state(self) -> Dict[str, Any]:
        """Capture current page state for verification"""
        try:
            state = {
                'url': self.page.url,
                'title': await self.page.title(),
            }
            
            # Capture screenshot
            try:
                state['screenshot'] = await self.page.screenshot(timeout=3000)
            except:
                state['screenshot'] = None
            
            # Capture DOM hash
            try:
                dom_text = await self.page.evaluate("() => document.body.innerText")
                import hashlib
                state['dom_hash'] = hashlib.md5(dom_text.encode()).hexdigest()
            except:
                state['dom_hash'] = None
            
            return state
        except Exception as e:
            logger.warning(f"⚠️  Failed to capture page state: {e}")
            return {}
    
    async def execute_action(self, action_plan: Dict[str, Any]) -> bool:
        """Execute the planned action with coordinate verification and visual feedback"""
        action = action_plan.get("action", "done")
        params = action_plan.get("params", {})
        
        # Capture before screenshot for verification
        before_screenshot = None
        if action in ["click", "type", "drag", "hover"]:
            try:
                before_screenshot = await self.page.screenshot(timeout=3000)
            except:
                pass
        
        try:
            if action == "navigate":
                url = params.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                logger.info(f"🌐 Navigating to: {url}")
                
                # Try multiple wait strategies with shorter timeouts
                try:
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                except:
                    # Fallback: try with networkidle
                    try:
                        await self.page.goto(url, wait_until="networkidle", timeout=10000)
                    except:
                        # Last resort: just load
                        await self.page.goto(url, wait_until="load", timeout=8000)
                
                # Wait for page to stabilize (reduced from 2000ms)
                await self.page.wait_for_timeout(1000)
                
                # Wait for network to be idle (dynamic content)
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=3000)
                except:
                    pass  # Continue even if network doesn't idle
                
            elif action == "click":
                text = params.get("text", "")
                selector = params.get("selector")
                x = params.get("x")
                y = params.get("y")
                bbox = params.get("bbox")  # Bounding box from vision
                confidence = params.get("confidence", 0.5)
                
                # Multi-strategy retry with confidence-based decisions
                strategies = []
                
                # Strategy 1: Vision → DOM mapping (if bbox available and high confidence)
                if bbox and confidence >= 0.7:
                    mapped_selector = await self.map_bbox_to_dom_selector(bbox)
                    if mapped_selector:
                        strategies.append(('dom_from_vision', mapped_selector, None, None))
                        logger.info(f"🎯 Strategy 1: Using DOM selector from vision mapping: {mapped_selector}")
                
                # Strategy 2: Direct selector (if provided and valid)
                if selector:
                    # Validate selector exists before adding to strategies
                    try:
                        count = await self.page.locator(selector).count()
                        if count > 0:
                            strategies.append(('selector', selector, None, None))
                            logger.info(f"🎯 Strategy 2: Using provided selector: {selector} ({count} matches)")
                        else:
                            logger.warning(f"⚠️  Selector '{selector}' not found on page, skipping")
                    except Exception as e:
                        logger.warning(f"⚠️  Invalid selector '{selector}': {e}")
                
                # Strategy 3: Coordinates with verification (if provided)
                if x is not None and y is not None:
                    strategies.append(('coordinates', None, x, y))
                    logger.info(f"🎯 Strategy 3: Using coordinates: ({x}, {y})")
                
                # Strategy 4: Text-based (if provided)
                if text:
                    strategies.append(('text', None, None, None))
                    logger.info(f"🎯 Strategy 4: Using text search: '{text}'")
                
                # Try each strategy
                for strategy_name, strat_selector, strat_x, strat_y in strategies:
                    try:
                        logger.info(f"🔄 Attempting click with strategy: {strategy_name}")
                        
                        # Capture before state
                        before_state = await self.capture_page_state()
                        
                        # Execute click based on strategy
                        if strategy_name in ['dom_from_vision', 'selector']:
                            element = self.page.locator(strat_selector).first
                            await element.wait_for(state="visible", timeout=3000)
                            await element.scroll_into_view_if_needed()
                            await self.page.wait_for_timeout(300)
                            await element.click(timeout=3000)
                            
                        elif strategy_name == 'coordinates':
                            # Verify coordinates first
                            verification = await self.verify_coordinates(strat_x, strat_y)
                            if verification['confidence'] < 0.5:
                                logger.warning(f"⚠️  Low confidence ({verification['confidence']:.2f}), trying next strategy")
                                continue
                            await self.page.mouse.click(strat_x, strat_y)
                            
                        elif strategy_name == 'text':
                            # Text-based click (existing logic)
                            element = self.page.get_by_text(text, exact=False).first
                            await element.wait_for(state="visible", timeout=3000)
                            await element.click(timeout=3000)
                        
                        await self.page.wait_for_timeout(800)
                        
                        # Capture after state and verify
                        after_state = await self.capture_page_state()
                        verification = await self.verify_action_success('click', before_state, after_state)
                        
                        if verification['success'] or verification['confidence'] >= 0.3:
                            logger.info(f"✅ Click succeeded with strategy: {strategy_name}")
                            return True
                        else:
                            logger.warning(f"⚠️  Click with {strategy_name} didn't produce expected changes, trying next strategy")
                            continue
                            
                    except Exception as e:
                        logger.warning(f"⚠️  Strategy {strategy_name} failed: {e}, trying next")
                        continue
                
                # All strategies failed
                logger.error(f"❌ All click strategies failed")
                return False
            
            elif action == "double_click":
                x = params.get("x")
                y = params.get("y")
                selector = params.get("selector")
                
                if x is not None and y is not None:
                    logger.info(f"🖱️  Double-clicking at coordinates: ({x}, {y})")
                    await self.page.mouse.dblclick(x, y)
                    await self.page.wait_for_timeout(1000)
                elif selector:
                    logger.info(f"🖱️  Double-clicking selector: {selector}")
                    element = self.page.locator(selector).first
                    await element.wait_for(state="visible", timeout=5000)
                    await element.dblclick()
                    await self.page.wait_for_timeout(1000)
            
            elif action == "type":
                selector = params.get("selector")
                input_text = params.get("input_text", "")
                logger.info(f"⌨️  Typing: '{input_text}' into {selector or 'auto-detected field'}")
                
                # Enhanced input field detection with multiple strategies
                if selector:
                    try:
                        element = self.page.locator(selector).first
                        await element.wait_for(state="visible", timeout=3000)
                        await element.click()  # Focus the element
                        await self.page.wait_for_timeout(200)
                        await element.fill(input_text)
                        await self.page.wait_for_timeout(200)
                        # Press Enter if it's a search field
                        if "search" in selector.lower() or 'name="q"' in selector or 'type="search"' in selector:
                            await element.press("Enter")
                        logger.info(f"✅ Filled input using provided selector")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed with provided selector: {e}")
                        return False
                else:
                    # Try multiple strategies to find the input field
                    selectors = [
                        'textarea[name="q"]',  # Google search box (textarea)
                        'input[name="q"]',     # Google search box (input)
                        'input[name="search"]',
                        'input[type="search"]',
                        '[role="searchbox"]',  # ARIA searchbox
                        '[role="textbox"]',    # ARIA textbox
                        'input[aria-label*="Search" i]',
                        'input[placeholder*="Search" i]',
                        'textarea[aria-label*="Search" i]',
                        'input[class*="search" i]',
                        'input[id*="search" i]',
                        'textarea[class*="search" i]',
                        'input[type="text"]:visible',  # Any visible text input
                        'textarea:visible'  # Any visible textarea
                    ]
                    
                    filled = False
                    for sel in selectors:
                        try:
                            element = self.page.locator(sel).first
                            await element.wait_for(state="visible", timeout=2000)
                            await element.wait_for(state="attached", timeout=2000)
                            
                            # Try to interact
                            await element.scroll_into_view_if_needed()
                            await self.page.wait_for_timeout(300)
                            await element.click(timeout=3000)
                            await self.page.wait_for_timeout(500)
                            await element.fill(input_text, timeout=5000)
                            await self.page.wait_for_timeout(500)
                            await element.press("Enter")
                            
                            filled = True
                            logger.info(f"✅ Filled input using selector: {sel}")
                            break
                        except Exception as e:
                            logger.debug(f"Failed with selector {sel}: {str(e)[:100]}")
                            continue
                    
                    if not filled:
                        logger.warning(f"Could not find input field to type into after trying {len(selectors)} selectors")
                        return False
                
                # Verify action succeeded by checking for page changes
                await self.page.wait_for_timeout(2000)
                
                # Wait for any navigation or dynamic updates
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=3000)
                except:
                    pass
            
            elif action == "double_click":
                x = params.get("x")
                y = params.get("y")
                selector = params.get("selector")
                
                if x is not None and y is not None:
                    logger.info(f"🖱️  Double-clicking at coordinates: ({x}, {y})")
                    await self.page.mouse.dblclick(x, y)
                    await self.page.wait_for_timeout(1000)
                elif selector:
                    logger.info(f"🖱️  Double-clicking selector: {selector}")
                    element = self.page.locator(selector).first
                    await element.wait_for(state="visible", timeout=5000)
                    await element.dblclick()
                    await self.page.wait_for_timeout(1000)
            
            elif action == "right_click":
                x = params.get("x")
                y = params.get("y")
                selector = params.get("selector")
                
                if x is not None and y is not None:
                    logger.info(f"🖱️  Right-clicking at coordinates: ({x}, {y})")
                    await self.page.mouse.click(x, y, button="right")
                    await self.page.wait_for_timeout(1000)
                elif selector:
                    logger.info(f"🖱️  Right-clicking selector: {selector}")
                    element = self.page.locator(selector).first
                    await element.wait_for(state="visible", timeout=5000)
                    await element.click(button="right")
                    await self.page.wait_for_timeout(1000)
            
            elif action == "hover":
                x = params.get("x")
                y = params.get("y")
                selector = params.get("selector")
                
                if x is not None and y is not None:
                    logger.info(f"🖱️  Hovering at coordinates: ({x}, {y})")
                    await self.page.mouse.move(x, y)
                    await self.page.wait_for_timeout(1000)
                elif selector:
                    logger.info(f"🖱️  Hovering over selector: {selector}")
                    element = self.page.locator(selector).first
                    await element.wait_for(state="visible", timeout=5000)
                    await element.hover()
                    await self.page.wait_for_timeout(1000)
            
            elif action == "drag":
                x = params.get("x")
                y = params.get("y")
                x2 = params.get("x2")
                y2 = params.get("y2")
                
                if x is not None and y is not None and x2 is not None and y2 is not None:
                    logger.info(f"🖱️  Dragging from ({x}, {y}) to ({x2}, {y2})")
                    await self.page.mouse.move(x, y)
                    await self.page.mouse.down()
                    await self.page.wait_for_timeout(500)
                    
                    # Smooth drag motion (important for slider CAPTCHAs)
                    steps = 10
                    for i in range(steps + 1):
                        intermediate_x = x + (x2 - x) * i / steps
                        intermediate_y = y + (y2 - y) * i / steps
                        await self.page.mouse.move(intermediate_x, intermediate_y)
                        await self.page.wait_for_timeout(50)
                    
                    await self.page.mouse.up()
                    await self.page.wait_for_timeout(1000)
                else:
                    logger.warning("Drag action requires x, y, x2, y2 coordinates")
                    return False
            
            elif action == "scroll_to":
                x = params.get("x")
                y = params.get("y")
                selector = params.get("selector")
                
                if x is not None and y is not None:
                    logger.info(f"📜 Scrolling to coordinates: ({x}, {y})")
                    await self.page.evaluate(f"window.scrollTo({x}, {y})")
                    await self.page.wait_for_timeout(1000)
                elif selector:
                    logger.info(f"📜 Scrolling to selector: {selector}")
                    element = self.page.locator(selector).first
                    await element.scroll_into_view_if_needed()
                    await self.page.wait_for_timeout(1000)
                
            elif action == "scroll":
                logger.info("📜 Scrolling page")
                await self.page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
                await self.page.wait_for_timeout(1500)
                
            elif action == "extract":
                logger.info("📊 Extracting data from current page")
                await self.capture_screenshot("extract")
            
            elif action == "analyze_images":
                logger.info("🎨 Analyzing images on current page using vision")
                # Vision model already analyzed in planning step
                # The descriptions are in action_plan['reasoning']
                await self.capture_screenshot("analyze_images")
                # Store analysis in actions_taken for final summary
                self.actions_taken.append({
                    'action': 'analyze_images',
                    'analysis': action_plan.get('reasoning', 'Image analysis completed')
                })
                # Extract action means we got the data we need
                # The completed_subtask should be set in action_plan
                return False  # Continue to next subtask
            
            elif action == "save_images":
                logger.info("💾 Saving images from current page using vision")
                
                # Take screenshot for vision analysis
                screenshot_path = await self.capture_screenshot("save_images")
                
                # Use vision to identify and describe images
                try:
                    import base64
                    with open(screenshot_path, 'rb') as f:
                        screenshot_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Get image URLs and descriptions from page
                    image_data = await self.page.evaluate("""
                        () => {
                            const images = Array.from(document.querySelectorAll('img'))
                                .filter(img => {
                                    const rect = img.getBoundingClientRect();
                                    return rect.width > 100 && rect.height > 100 && 
                                           rect.top >= 0 && rect.top < window.innerHeight;
                                })
                                .slice(0, 5)  // Get first 5 visible images
                                .map(img => ({
                                    src: img.src,
                                    alt: img.alt || '',
                                    width: img.width,
                                    height: img.height
                                }));
                            return images;
                        }
                    """)
                    
                    if not image_data:
                        logger.warning("No images found on page")
                        return False
                    
                    logger.info(f"Found {len(image_data)} images to save")
                    
                    # Download and save each image
                    import httpx
                    saved_count = 0
                    image_descriptions = []
                    
                    for i, img in enumerate(image_data[:3], 1):  # Save first 3 images
                        try:
                            img_url = img['src']
                            if not img_url.startswith('http'):
                                continue
                            
                            # Download image
                            async with httpx.AsyncClient(timeout=10.0) as client:
                                response = await client.get(img_url)
                                if response.status_code == 200:
                                    # Save image
                                    ext = 'jpg' if 'jpeg' in img_url or 'jpg' in img_url else 'png'
                                    filename = f"{self.task_id}_image_{i}.{ext}"
                                    filepath = DOWNLOADS_DIR / filename
                                    
                                    with open(filepath, 'wb') as f:
                                        f.write(response.content)
                                    
                                    self.downloads.append(str(filepath))
                                    saved_count += 1
                                    
                                    # Store image info for description
                                    image_descriptions.append({
                                        'filename': filename,
                                        'alt': img.get('alt', ''),
                                        'size': f"{img.get('width')}x{img.get('height')}"
                                    })
                                    
                                    logger.info(f"💾 Saved image {i}: {filename}")
                        except Exception as e:
                            logger.warning(f"Failed to save image {i}: {e}")
                    
                    # Store descriptions in actions_taken
                    self.actions_taken.append({
                        'action': 'save_images',
                        'saved_count': saved_count,
                        'images': image_descriptions,
                        'analysis': action_plan.get('reasoning', 'Images saved')
                    })
                    
                    logger.info(f"✅ Saved {saved_count} images successfully")
                    return False  # Continue to next subtask
                    
                except Exception as e:
                    logger.error(f"Failed to save images: {e}")
                    return False
            
            elif action == "upload":
                file_path = params.get("file_path", "")
                selector = params.get("selector", 'input[type="file"]')
                
                logger.info(f"📤 Uploading file: {file_path}")
                
                # Check if file exists
                upload_file = Path(file_path)
                if not upload_file.exists():
                    # Try in uploads directory
                    upload_file = UPLOADS_DIR / file_path
                    if not upload_file.exists():
                        logger.warning(f"File not found: {file_path}")
                        return False
                
                # Find file input and upload
                try:
                    file_input = self.page.locator(selector).first
                    await file_input.wait_for(state="attached", timeout=5000)
                    await file_input.set_input_files(str(upload_file))
                    
                    self.uploaded_files.append(str(upload_file))
                    logger.info(f"✅ File uploaded successfully: {upload_file.name}")
                    
                    await self.page.wait_for_timeout(1000)
                except Exception as e:
                    logger.error(f"Failed to upload file: {e}")
                    return False
            
            elif action == "download":
                # Trigger download by clicking a download button/link
                selector = params.get("selector")
                text = params.get("text", "")
                
                logger.info(f"📥 Triggering download...")
                
                # Set up download expectation
                async with self.page.expect_download(timeout=30000) as download_info:
                    if selector:
                        await self.page.locator(selector).first.click()
                    elif text:
                        await self.page.get_by_text(text).first.click()
                    else:
                        logger.warning("No selector or text provided for download")
                        return False
                
                # Download is handled by _handle_download
                logger.info("✅ Download triggered successfully")
                await self.page.wait_for_timeout(2000)
                
            elif action == "skip":
                logger.info("⏭️  Skipping current subtask")
                return False
                
            elif action == "done":
                logger.info("✅ Task marked as complete")
                return True
            
            else:
                # Invalid action - treat as done to avoid loops
                logger.warning(f"⚠️  Invalid action '{action}' - treating as done")
                return True
                
            return False
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error executing {action}: {error_msg[:200]}")
            
            # Capture screenshot on error for debugging
            error_screenshot = None
            try:
                error_screenshot = await self.capture_screenshot(f"error_{action}")
                logger.info(f"� ErTror screenshot captured: {error_screenshot}")
            except:
                pass
            
            # Provide helpful error context
            if "Timeout" in error_msg:
                logger.warning("💡 Tip: Element might be loading slowly or not visible")
            elif "not found" in error_msg.lower():
                logger.warning("💡 Tip: Element selector might be incorrect or page structure changed")
            elif "detached" in error_msg.lower():
                logger.warning("💡 Tip: Element was removed from DOM, page might have changed")
            
            self.actions_taken.append({
                "action": action,
                "status": "failed",
                "error": error_msg[:200],
                "timestamp": datetime.now().isoformat(),
                "url": self.page.url if self.page else "unknown",
                "error_screenshot": error_screenshot
            })
            return False
    
    async def run(self) -> Dict[str, Any]:
        """Main execution loop with task planning and live streaming"""
        self.start_time = datetime.now()
        logger.info(f"🚀 Starting browser agent for task: {self.task}")
        
        # Create initial task plan
        plan_start = datetime.now()
        self.task_plan = await self.create_task_plan()
        self.metrics["llm_calls"] += 1
        
        # Start live screenshot streaming
        self.is_running = True
        if self.enable_streaming:
            self.streaming_task = asyncio.create_task(self.stream_screenshots())
            logger.info("📹 Live screenshot streaming enabled")
        
        result_summary = ""
        
        try:
            for step in range(1, self.max_steps + 1):
                logger.info(f"\n📍 Step {step}/{self.max_steps}")
                
                # Get current page state (always fresh - detects dynamic changes)
                page_content = await self.get_page_content()
                
                # Detect if page has changed (URL or content)
                current_url = page_content.get('url')
                current_hash = hash(page_content.get('bodyText', '')[:500])
                
                if self.last_page_url and (current_url != self.last_page_url or current_hash != self.last_page_hash):
                    logger.info(f"📄 Page changed detected: {self.last_page_url} -> {current_url}")
                    
                    # Auto-complete or skip subtasks that became irrelevant due to page change
                    # For example, if we typed in a search box and the page auto-navigated,
                    # we can skip the "click search button" subtask
                    for subtask in self.task_plan:
                        if subtask['status'] == 'pending':
                            subtask_lower = subtask['subtask'].lower()
                            # If page navigated after typing, skip click/submit subtasks
                            if current_url != self.last_page_url and ('click' in subtask_lower or 'submit' in subtask_lower or 'press' in subtask_lower):
                                if any('type' in a.get('action', '') or 'search' in a.get('action', '') for a in self.actions_taken[-3:]):
                                    subtask['status'] = 'completed'
                                    self.completed_subtasks.append(subtask['subtask'])
                                    logger.info(f"✅ Auto-completed subtask (page navigated): {subtask['subtask']}")
                                    break
                
                self.last_page_url = current_url
                self.last_page_hash = current_hash
                
                # Decide if vision is needed for next step
                use_vision = await self.should_use_vision(page_content)
                
                # Plan next action (with vision if needed)
                action_plan = None
                if use_vision:
                    # Capture screenshot for vision analysis with robust error handling
                    screenshot_bytes = None
                    for timeout_val in [15000, 5000, 3000]:
                        try:
                            screenshot_bytes = await self.page.screenshot(
                                timeout=timeout_val,
                                animations="disabled" if timeout_val > 5000 else None
                            )
                            break
                        except Exception as e:
                            if timeout_val == 3000:  # Last attempt
                                logger.warning(f"⚠️  Vision screenshot failed, falling back to text-only")
                                use_vision = False
                                break
                            continue
                    
                    if screenshot_bytes:
                        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                        
                        # Try vision-based planning
                        action_plan = await self.plan_action_with_vision(screenshot_base64, page_content, step)
                        self.metrics["llm_calls"] += 1
                
                # Fallback to text-only if vision failed or not needed
                if not action_plan:
                    action_plan = self.plan_next_action(page_content, step)
                    self.metrics["llm_calls"] += 1
                
                logger.info(f"💭 Plan: {action_plan.get('action')} - {action_plan.get('reasoning', '')}")
                
                # Check if a subtask was completed
                if action_plan.get("completed_subtask"):
                    subtask_name = action_plan["completed_subtask"]
                    for subtask in self.task_plan:
                        if subtask_name.lower() in subtask['subtask'].lower() and subtask['status'] == 'pending':
                            subtask['status'] = 'completed'
                            self.completed_subtasks.append(subtask['subtask'])
                            logger.info(f"✅ Completed subtask: {subtask['subtask']}")
                            self.consecutive_same_actions = 0
                            break
                
                # Check if LLM wants to add a new subtask mid-way
                if action_plan.get("new_subtask"):
                    new_subtask_text = action_plan["new_subtask"]
                    self.task_plan.append({
                        "subtask": new_subtask_text,
                        "status": "pending"
                    })
                    logger.info(f"➕ Added new subtask: {new_subtask_text}")
                    self.consecutive_same_actions = 0
                
                # Check if a subtask failed/should be skipped
                if action_plan.get("failed_subtask"):
                    subtask_name = action_plan["failed_subtask"]
                    for subtask in self.task_plan:
                        if subtask_name.lower() in subtask['subtask'].lower() and subtask['status'] != 'failed':
                            subtask['status'] = 'failed'
                            self.failed_subtasks.append({
                                "subtask": subtask['subtask'],
                                "reason": action_plan.get("reasoning", "Unknown")
                            })
                            logger.warning(f"❌ Failed/Skipped subtask: {subtask['subtask']}")
                            self.consecutive_same_actions = 0
                            break
                
                # Track consecutive same actions and force skip if stuck
                current_action = action_plan.get("action")
                if current_action == self.last_action_type:
                    self.consecutive_same_actions += 1
                    if self.consecutive_same_actions >= 3:
                        logger.error(f"🚨 STUCK! Same action '{current_action}' repeated {self.consecutive_same_actions} times - FORCING SKIP")
                        # Force mark current pending subtask as failed
                        for subtask in self.task_plan:
                            if subtask['status'] == 'pending':
                                subtask['status'] = 'failed'
                                self.failed_subtasks.append({
                                    "subtask": subtask['subtask'],
                                    "reason": f"Stuck in loop - {current_action} repeated {self.consecutive_same_actions} times"
                                })
                                logger.warning(f"❌ Auto-failed stuck subtask: {subtask['subtask']}")
                                self.consecutive_same_actions = 0
                                break
                else:
                    self.consecutive_same_actions = 1
                    self.last_action_type = current_action
                
                # Record action
                self.actions_taken.append({
                    "step": step,
                    "action": action_plan.get("action"),
                    "reasoning": action_plan.get("reasoning"),
                    "url": page_content["url"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check if ALL subtasks are complete or failed (no pending)
                all_complete = all(t['status'] in ['completed', 'failed'] for t in self.task_plan)
                
                # If all subtasks are done (completed or failed), finish immediately
                if all_complete:
                    completed_count = len([t for t in self.task_plan if t['status'] == 'completed'])
                    failed_count = len([t for t in self.task_plan if t['status'] == 'failed'])
                    result_summary = action_plan.get("result_summary") or f"Completed {completed_count}/{len(self.task_plan)} subtasks ({failed_count} failed)"
                    await self.capture_screenshot("final")
                    logger.info(f"🎯 All subtasks processed: {completed_count} completed, {failed_count} failed")
                    break
                
                # Check if LLM marked complete but subtasks remain
                if action_plan.get("is_complete") and not all_complete:
                    # LLM thinks it's done but there are pending subtasks
                    pending = [t['subtask'] for t in self.task_plan if t['status'] == 'pending']
                    logger.warning(f"⚠️  LLM marked complete but {len(pending)} subtasks pending: {pending}")
                    # Continue execution to finish pending subtasks
                
                # Execute action
                is_done = await self.execute_action(action_plan)
                
                # Auto-complete subtasks based on action if LLM didn't specify
                if not action_plan.get("completed_subtask") and not is_done:
                    current_action = action_plan.get("action")
                    # Try to auto-match action to pending subtask
                    for subtask in self.task_plan:
                        if subtask['status'] == 'pending':
                            subtask_lower = subtask['subtask'].lower()
                            # Match navigate actions
                            if current_action == "navigate" and ("navigate" in subtask_lower or "go to" in subtask_lower or "visit" in subtask_lower):
                                subtask['status'] = 'completed'
                                self.completed_subtasks.append(subtask['subtask'])
                                logger.info(f"✅ Auto-completed subtask: {subtask['subtask']}")
                                break
                            # Match extract actions
                            elif current_action == "extract" and ("extract" in subtask_lower or "get" in subtask_lower or "tell me" in subtask_lower):
                                subtask['status'] = 'completed'
                                self.completed_subtasks.append(subtask['subtask'])
                                logger.info(f"✅ Auto-completed subtask: {subtask['subtask']}")
                                break
                            # Match type/search actions
                            elif current_action == "type" and ("search" in subtask_lower or "type" in subtask_lower):
                                subtask['status'] = 'completed'
                                self.completed_subtasks.append(subtask['subtask'])
                                logger.info(f"✅ Auto-completed subtask: {subtask['subtask']}")
                                break
                
                # Capture screenshot after action
                await self.capture_screenshot(f"step_{step}")
            
            # Build final summary
            if not result_summary:
                completed = len([t for t in self.task_plan if t['status'] == 'completed'])
                result_summary = f"Completed {completed}/{len(self.task_plan)} subtasks"
            
            # Calculate final metrics
            if self.start_time:
                self.metrics["total_time"] = (datetime.now() - self.start_time).total_seconds()
                self.metrics["screenshots_taken"] = len(self.screenshots)
            
            logger.info(f"✅ Browser agent completed: {result_summary}")
            logger.info(f"📊 Metrics: {self.metrics['total_time']:.1f}s total, {self.metrics['llm_calls']} LLM calls, {self.metrics['screenshots_taken']} screenshots")
            
        finally:
            # Stop streaming
            self.is_running = False
            if self.streaming_task:
                self.streaming_task.cancel()
                try:
                    await self.streaming_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up live screenshots for this task
            async with live_screenshots_lock:
                if self.task_id in live_screenshots:
                    del live_screenshots[self.task_id]
        
        return {
            "success": True,
            "summary": result_summary or "Completed all steps",
            "actions": self.actions_taken,
            "screenshots": self.screenshots,
            "downloads": self.downloads,
            "uploaded_files": self.uploaded_files,
            "task_id": self.task_id,
            "metrics": self.metrics
        }

@app.post("/browse")
async def browse(request: BrowseRequest, headless: bool = False, enable_streaming: bool = True, thread_id: Optional[str] = None):
    """Execute browser automation task with optional live streaming (visible browser by default)"""
    logger.info(f"📥 Received task: {request.task} (thread_id: {thread_id})")
    
    agent = None
    try:
        # Use async context manager to ensure proper cleanup
        async with BrowserAgent(
            request.task, 
            request.max_steps, 
            headless=headless, 
            enable_streaming=enable_streaming,
            thread_id=thread_id
        ) as agent:
            result = await agent.run()
            
            return {
                "success": result["success"],
                "task_summary": result["summary"],
                "actions_taken": result["actions"],
                "screenshot_files": result["screenshots"],
                "downloaded_files": result.get("downloads", []),
                "uploaded_files": result.get("uploaded_files", []),
                "extracted_data": None,
                "task_id": result["task_id"],
                "metrics": result.get("metrics", {}),
                "error": None
            }
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure cleanup even on error
        if agent:
            try:
                await agent.__aexit__(None, None, None)
            except:
                pass
        
        return {
            "success": False,
            "task_summary": f"Error: {str(e)}",
            "actions_taken": [],
            "screenshot_files": None,
            "extracted_data": None,
            "task_id": None,
            "error": str(e)
        }

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "custom_browser_agent"}

@app.get("/info")
async def info():
    return AGENT_DEFINITION

@app.get("/live/{task_id}", include_in_schema=False)
async def get_live_screenshot(task_id: str):
    """Get the latest live screenshot for a task (silent endpoint)"""
    async with live_screenshots_lock:
        if task_id in live_screenshots:
            return live_screenshots[task_id]
        else:
            raise HTTPException(status_code=404, detail="Task not found or completed")

@app.get("/live", include_in_schema=False)
async def get_all_live_screenshots():
    """Get all active live screenshots (silent endpoint)"""
    async with live_screenshots_lock:
        return {"active_tasks": list(live_screenshots.keys()), "screenshots": live_screenshots}

if __name__ == "__main__":
    port = int(os.getenv("BROWSER_AGENT_PORT", 8070))
    logger.info(f"Starting Custom Browser Automation Agent on port {port}")
    
    # Configure uvicorn to filter out /live endpoint logs
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Custom filter to exclude /live requests
    class ExcludeLiveFilter(logging.Filter):
        def filter(self, record):
            return '/live' not in record.getMessage()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_config=log_config,
        access_log=True
    )
