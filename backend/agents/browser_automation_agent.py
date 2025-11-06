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

if not CEREBRAS_API_KEY and not GROQ_API_KEY and not NVIDIA_API_KEY:
    raise RuntimeError("At least one of CEREBRAS_API_KEY, GROQ_API_KEY, or NVIDIA_API_KEY must be set")

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
        
        # Handle page errors
        self.page.on("pageerror", lambda err: logger.warning(f"Page error: {err}"))
        
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
        """Capture and save screenshot"""
        try:
            filename = f"{self.task_id}_{name}_{len(self.screenshots)}.png"
            filepath = STORAGE_DIR / filename
            # Use animations: "disabled" to speed up and avoid font loading issues
            await self.page.screenshot(
                path=str(filepath), 
                full_page=False, 
                timeout=10000,
                animations="disabled"
            )
            self.screenshots.append(str(filepath))
            logger.info(f"📸 Screenshot saved: {filename}")
            return str(filepath)
        except Exception as e:
            logger.warning(f"⚠️  Screenshot failed: {e}, continuing without it")
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
                        screenshot_bytes = await self.page.screenshot(
                            full_page=False, 
                            timeout=5000,
                            animations="disabled"
                        )
                        
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
- Each subtask should be something you can DO (navigate, click, type, extract)
- Combine typing and submitting into one subtask (e.g., "Search for X" not "Type X" + "Click search")
- If task mentions multiple sites, create separate subtasks for each
- Keep subtasks specific and actionable
- Maximum 4 subtasks
- Focus on WHAT to do, not HOW to do it"""

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
            logger.warning("Page not fully loaded, continuing anyway")
        
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
                    // Priority: ID > Name > Aria-label > Class + Tag
                    if (el.id) return `#${el.id}`;
                    if (el.name) return `[name="${el.name}"]`;
                    if (el.getAttribute('aria-label')) return `[aria-label="${el.getAttribute('aria-label')}"]`;
                    
                    let path = el.tagName.toLowerCase();
                    if (el.type) path += `[type="${el.type}"]`;
                    if (el.className && typeof el.className === 'string') {
                        const classes = el.className.trim().split(/\\s+/).filter(c => c && !c.match(/^[\\d]/)).slice(0, 2);
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
- ALWAYS use the "selector" field from the INPUT FIELDS or BUTTONS list above
- For typing: Use the exact selector from INPUT FIELDS (e.g., {{"selector": "#searchInput", "input_text": "query"}})
- For clicking: Use the exact selector from BUTTONS (e.g., {{"selector": "[name='btnK']"}})
- For upload: Use {{"action": "upload", "params": {{"file_path": "filename.pdf", "selector": "input[type='file']"}}}}
- For download: Use {{"action": "download", "params": {{"selector": "download-button-selector"}}}}
- DO NOT make up selectors - only use ones provided in the lists above

AVAILABLE ACTIONS (YOU MUST USE EXACTLY ONE OF THESE):
- "navigate": Go to a URL (provide full URL in params.url)
- "click": Click an element (provide text in params.text or selector)
- "type": Type text into input field (provide text in params.input_text and selector)
- "scroll": Scroll down the page
- "extract": Extract data from current page (marks subtask as completed)
- "upload": Upload a file (provide file_path and selector for file input)
- "download": Download a file (provide selector or text for download button)
- "skip": Skip current subtask if blocked/stuck
- "done": Mark entire task as complete (ONLY when ALL subtasks are done)

INVALID ACTIONS (DO NOT USE): "parse", "report", "wait", "analyze", or any other action not listed above!

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
    
    async def execute_action(self, action_plan: Dict[str, Any]) -> bool:
        """Execute the planned action with robust error handling and retries"""
        action = action_plan.get("action", "done")
        params = action_plan.get("params", {})
        
        try:
            if action == "navigate":
                url = params.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                logger.info(f"🌐 Navigating to: {url}")
                
                # Try multiple wait strategies
                try:
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
                except:
                    # Fallback: try with networkidle
                    try:
                        await self.page.goto(url, wait_until="networkidle", timeout=15000)
                    except:
                        # Last resort: just load
                        await self.page.goto(url, wait_until="load", timeout=10000)
                
                # Wait for page to stabilize
                await self.page.wait_for_timeout(2000)
                
                # Wait for network to be idle (dynamic content)
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    pass  # Continue even if network doesn't idle
                
            elif action == "click":
                text = params.get("text", "")
                selector = params.get("selector")
                
                if selector:
                    logger.info(f"🖱️  Clicking selector: {selector}")
                    element = self.page.locator(selector).first
                    
                    # Wait for element to be ready
                    await element.wait_for(state="attached", timeout=5000)
                    await element.wait_for(state="visible", timeout=5000)
                    
                    # Scroll element into view
                    await element.scroll_into_view_if_needed()
                    await self.page.wait_for_timeout(500)
                    
                    # Ensure element is clickable (not covered)
                    try:
                        await element.click(timeout=5000, force=False)
                    except:
                        # Fallback: force click
                        logger.warning("Normal click failed, forcing click")
                        await element.click(timeout=5000, force=True)
                    
                elif text:
                    logger.info(f"🖱️  Clicking text: '{text}'")
                    # Try multiple strategies with better error handling
                    clicked = False
                    
                    # Wait for page to be stable
                    try:
                        await self.page.wait_for_load_state("networkidle", timeout=5000)
                    except:
                        await self.page.wait_for_load_state("domcontentloaded", timeout=3000)
                    
                    strategies = [
                        lambda: self.page.get_by_text(text, exact=False).first,
                        lambda: self.page.get_by_role("link", name=text).first,
                        lambda: self.page.get_by_role("button", name=text).first,
                        lambda: self.page.locator(f'text="{text}"').first,
                        lambda: self.page.locator(f'a:has-text("{text}")').first,
                        lambda: self.page.locator(f'button:has-text("{text}")').first
                    ]
                    
                    for strategy in strategies:
                        try:
                            element = strategy()
                            await element.wait_for(state="visible", timeout=3000)
                            await element.scroll_into_view_if_needed()
                            await element.click(timeout=3000)
                            clicked = True
                            logger.info(f"✅ Successfully clicked element")
                            break
                        except Exception as e:
                            continue
                    
                    if not clicked:
                        logger.warning(f"Could not find clickable element with text: {text}")
                        # Don't raise exception, just log and continue
                        return False
                        
                await self.page.wait_for_timeout(2000)
                
            elif action == "type":
                selector = params.get("selector")
                input_text = params.get("input_text", "")
                logger.info(f"⌨️  Typing: '{input_text}' into {selector or 'auto-detected field'}")
                
                # Enhanced input field detection with multiple strategies
                if selector:
                    try:
                        element = self.page.locator(selector).first
                        await element.wait_for(state="visible", timeout=5000)
                        await element.click()  # Focus the element
                        await self.page.wait_for_timeout(300)
                        await element.fill(input_text)
                        await self.page.wait_for_timeout(300)
                        # Press Enter if it's a search field
                        if "search" in selector.lower() or 'name="q"' in selector or 'type="search"' in selector:
                            await element.press("Enter")
                        logger.info(f"✅ Filled input using provided selector")
                    except Exception as e:
                        logger.warning(f"Failed with provided selector: {e}")
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
                
            elif action == "scroll":
                logger.info("📜 Scrolling page")
                await self.page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
                await self.page.wait_for_timeout(1500)
                
            elif action == "extract":
                logger.info("📊 Extracting data from current page")
                await self.capture_screenshot("extract")
                # Extract action means we got the data we need
                # The completed_subtask should be set in action_plan
                return False  # Continue to next subtask
            
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
                
                # Plan next action
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
