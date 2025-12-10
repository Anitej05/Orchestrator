# agents/custom_browser_agent.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
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
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
except ImportError as exc:
    raise RuntimeError(
        "Playwright is required for browser automation; install it with `pip install playwright` "
        "and run `playwright install` before starting the agent."
    ) from exc
import re

# Import standardized file manager
try:
    from agents.agent_file_manager import AgentFileManager, FileType, FileStatus
except ImportError:
    from agent_file_manager import AgentFileManager, FileType, FileStatus

# Import SOTA improvements
try:
    from agents.browser_agent_improvements import (
        ContextOptimizer,
        SelectorStrategy,
        PageStabilizer,
        DynamicPlanner,
        VisionOptimizer
    )
except ImportError:
    # When running directly, use relative import
    from browser_agent_improvements import (
        ContextOptimizer,
        SelectorStrategy,
        PageStabilizer,
        DynamicPlanner,
        VisionOptimizer
    )

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

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"status": "healthy", "agent": "custom_browser_agent", "message": "Browser Automation Agent is running"}

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all browsers on shutdown"""
    logger.info("🔒 Shutting down - cleaning up all browsers...")
    async with active_browsers_lock:
        for task_id, agent in list(active_browsers.items()):
            try:
                await agent.__aexit__(None, None, None)
            except:
                pass
        active_browsers.clear()
    logger.info("✅ Browser cleanup complete")

# Storage
STORAGE_DIR = Path("storage/browser_screenshots")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS_DIR = Path("storage/browser_downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = Path("storage/browser_uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize standardized file managers for screenshots and downloads
screenshot_file_manager = AgentFileManager(
    agent_id="browser_agent_screenshots",
    storage_dir=str(STORAGE_DIR),
    default_ttl_hours=24,  # Screenshots expire after 24 hours
    auto_cleanup=True,
    cleanup_interval_hours=6
)

download_file_manager = AgentFileManager(
    agent_id="browser_agent_downloads",
    storage_dir=str(DOWNLOADS_DIR),
    default_ttl_hours=72,  # Downloads expire after 3 days
    auto_cleanup=True,
    cleanup_interval_hours=12
)

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

def parse_json_robust(json_str: str, content_preview: str = "") -> Optional[Dict]:
    """
    Robustly parse JSON from LLM responses, handling common formatting issues.
    Returns None if parsing fails completely.
    """
    try:
        # First try direct parsing
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parse failed: {e.msg} at position {e.pos}")
        
        # Clean up common JSON issues
        cleaned = json_str
        
        # Fix trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix unquoted property names (but be careful not to quote values)
        # This regex looks for word characters followed by colon, not already in quotes
        cleaned = re.sub(r'(?<!")(\b[a-zA-Z_]\w*)(?!")(\s*):', r'"\1"\2:', cleaned)
        
        # Fix double-quoted keys that got over-quoted
        cleaned = re.sub(r'""([^"]+)"":', r'"\1":', cleaned)
        
        # Try parsing cleaned version
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e2:
            logger.warning(f"⚠️  JSON parse error: {e2.msg} at position {e2.pos}")
            if content_preview:
                logger.debug(f"Content preview: {content_preview[:200]}")
            
            # Last resort: try to extract key fields manually
            action_match = re.search(r'"action"\s*:\s*"([^"]+)"', cleaned)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', cleaned)
            
            if action_match:
                logger.info("📝 Extracted partial action from malformed JSON")
                return {
                    'action': action_match.group(1),
                    'reasoning': reasoning_match.group(1) if reasoning_match else '',
                    'params': {},
                    'confidence': 0.5,
                    'bounding_boxes': []
                }
            
            return None

# Vision Model Manager with exponential backoff
class VisionManager:
    """Manages vision model providers with intelligent fallback and exponential backoff"""
    
    def __init__(self):
        self.providers = []
        
        # Exponential backoff state for each provider
        self.backoff_state = {}
        self.base_backoff = 3  # Start with 3 seconds for vision (slower models)
        self.max_backoff = 600  # Max 10 minutes for vision
        
        # Build provider chain: Ollama → NVIDIA
        if OLLAMA_API_KEY:
            self.providers.append({
                "name": "ollama",
                "model": "qwen3-vl:235b-cloud",
                "base_url": "https://ollama.com/v1"
            })
            self.backoff_state["ollama"] = {'until': 0, 'backoff_seconds': 0, 'consecutive_failures': 0}
        
        if NVIDIA_API_KEY:
            self.providers.append({
                "name": "nvidia_vision",
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "base_url": "https://integrate.api.nvidia.com/v1"
            })
            self.backoff_state["nvidia_vision"] = {'until': 0, 'backoff_seconds': 0, 'consecutive_failures': 0}
        
        if self.providers:
            logger.info(f"🎨 Initialized vision chain: {' → '.join([p['name'] for p in self.providers])}")
        else:
            logger.info("📝 No vision providers available - text-only mode")
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error is a rate limit error"""
        rate_limit_indicators = ["429", "rate limit", "quota", "too many requests", "rate_limit_exceeded"]
        return any(indicator in error_msg.lower() for indicator in rate_limit_indicators)
    
    def _is_temporary_error(self, error_msg: str) -> bool:
        """Check if error is temporary"""
        temporary_indicators = ["timeout", "connection", "network", "503", "502", "504"]
        return any(indicator in error_msg.lower() for indicator in temporary_indicators)
    
    def _apply_backoff(self, provider_name: str, is_rate_limit: bool = False):
        """Apply exponential backoff to a vision provider"""
        import time
        
        state = self.backoff_state[provider_name]
        state['consecutive_failures'] += 1
        
        # Calculate backoff time
        if is_rate_limit:
            backoff_seconds = min(self.base_backoff * (3 ** state['consecutive_failures']), self.max_backoff)
        else:
            backoff_seconds = min(self.base_backoff * (2 ** state['consecutive_failures']), self.max_backoff)
        
        state['backoff_seconds'] = backoff_seconds
        state['until'] = time.time() + backoff_seconds
        
        logger.warning(f"⏰ Vision {provider_name.upper()} backed off for {backoff_seconds:.1f}s (failure #{state['consecutive_failures']})")
    
    def _reset_backoff(self, provider_name: str):
        """Reset backoff state after successful call"""
        state = self.backoff_state[provider_name]
        if state['consecutive_failures'] > 0:
            logger.info(f"✅ Vision {provider_name.upper()} recovered - resetting backoff")
        state['consecutive_failures'] = 0
        state['backoff_seconds'] = 0
        state['until'] = 0
    
    def get_available_provider(self):
        """Get next available vision provider (not backed off)"""
        import time
        current_time = time.time()
        
        available = []
        backed_off = []
        
        for provider in self.providers:
            name = provider['name']
            state = self.backoff_state[name]
            
            if current_time >= state['until']:
                available.append(provider)
            else:
                remaining = state['until'] - current_time
                backed_off.append({
                    'provider': provider,
                    'remaining': remaining
                })
        
        # Return first available
        if available:
            return available[0], None
        
        # If all backed off, return the one with shortest wait
        if backed_off:
            backed_off.sort(key=lambda x: x['remaining'])
            shortest = backed_off[0]
            return shortest['provider'], shortest['remaining']
        
        return None, None
    
    def record_success(self, provider_name: str):
        """Record successful vision call"""
        self._reset_backoff(provider_name)
    
    def record_failure(self, provider_name: str, error_msg: str):
        """Record failed vision call and apply backoff"""
        is_rate_limit = self._is_rate_limit_error(error_msg)
        is_temporary = self._is_temporary_error(error_msg)
        
        if is_rate_limit:
            logger.warning(f"⚠️  Vision {provider_name.upper()} rate limited")
            self._apply_backoff(provider_name, is_rate_limit=True)
        elif is_temporary:
            logger.warning(f"⚠️  Vision {provider_name.upper()} temporary error")
            self._apply_backoff(provider_name, is_rate_limit=False)
        else:
            # For non-temporary errors, apply minimal backoff
            if self.backoff_state[provider_name]['consecutive_failures'] < 2:
                self._apply_backoff(provider_name, is_rate_limit=False)

# LLM Client with fallback chain
class LLMManager:
    """Manages LLM providers with intelligent fallback and exponential backoff"""
    
    def __init__(self):
        self.providers = []
        self.current_provider_idx = 0
        self.failure_counts = {}
        
        # Exponential backoff state for each provider
        self.backoff_state = {}  # {provider_name: {'until': timestamp, 'backoff_seconds': seconds, 'consecutive_failures': count}}
        self.base_backoff = 2  # Start with 2 seconds
        self.max_backoff = 300  # Max 5 minutes
        
        # Build provider chain: Cerebras → Groq → NVIDIA
        if CEREBRAS_API_KEY:
            self.providers.append({
                "name": "cerebras",
                "client": OpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1"),
                "model": "llama-3.3-70b",
                "max_tokens": 1000
            })
            self.backoff_state["cerebras"] = {'until': 0, 'backoff_seconds': 0, 'consecutive_failures': 0}
        
        if GROQ_API_KEY:
            self.providers.append({
                "name": "groq",
                "client": OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 1000
            })
            self.backoff_state["groq"] = {'until': 0, 'backoff_seconds': 0, 'consecutive_failures': 0}
        
        if NVIDIA_API_KEY:
            self.providers.append({
                "name": "nvidia",
                "client": OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1"),
                "model": "meta/llama-3.1-70b-instruct",
                "max_tokens": 1000
            })
            self.backoff_state["nvidia"] = {'until': 0, 'backoff_seconds': 0, 'consecutive_failures': 0}
        
        if not self.providers:
            raise RuntimeError("No LLM providers available")
        
        logger.info(f"🔧 Initialized LLM chain: {' → '.join([p['name'] for p in self.providers])}")
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error is a rate limit error"""
        rate_limit_indicators = ["429", "rate limit", "quota", "too many requests", "rate_limit_exceeded"]
        return any(indicator in error_msg.lower() for indicator in rate_limit_indicators)
    
    def _is_temporary_error(self, error_msg: str) -> bool:
        """Check if error is temporary (timeout, connection, etc.)"""
        temporary_indicators = ["timeout", "connection", "network", "503", "502", "504"]
        return any(indicator in error_msg.lower() for indicator in temporary_indicators)
    
    def _apply_backoff(self, provider_name: str, is_rate_limit: bool = False):
        """Apply exponential backoff to a provider"""
        import time
        
        state = self.backoff_state[provider_name]
        state['consecutive_failures'] += 1
        
        # Calculate backoff time using exponential backoff
        if is_rate_limit:
            # For rate limits, use more aggressive backoff
            backoff_seconds = min(self.base_backoff * (3 ** state['consecutive_failures']), self.max_backoff)
        else:
            # For other errors, use standard exponential backoff
            backoff_seconds = min(self.base_backoff * (2 ** state['consecutive_failures']), self.max_backoff)
        
        state['backoff_seconds'] = backoff_seconds
        state['until'] = time.time() + backoff_seconds
        
        logger.warning(f"⏰ {provider_name.upper()} backed off for {backoff_seconds:.1f}s (failure #{state['consecutive_failures']})")
    
    def _reset_backoff(self, provider_name: str):
        """Reset backoff state after successful call"""
        state = self.backoff_state[provider_name]
        if state['consecutive_failures'] > 0:
            logger.info(f"✅ {provider_name.upper()} recovered - resetting backoff")
        state['consecutive_failures'] = 0
        state['backoff_seconds'] = 0
        state['until'] = 0
    
    def _get_available_providers(self):
        """Get list of providers that are not currently backed off, sorted by backoff time"""
        import time
        current_time = time.time()
        
        available = []
        backed_off = []
        
        for provider in self.providers:
            name = provider['name']
            state = self.backoff_state[name]
            
            if current_time >= state['until']:
                # Provider is available
                available.append(provider)
            else:
                # Provider is backed off
                remaining = state['until'] - current_time
                backed_off.append({
                    'provider': provider,
                    'remaining': remaining,
                    'backoff_seconds': state['backoff_seconds']
                })
        
        # If no providers available, return the one with shortest remaining backoff
        if not available and backed_off:
            backed_off.sort(key=lambda x: x['remaining'])
            shortest = backed_off[0]
            logger.warning(f"⚠️  All providers backed off. Using {shortest['provider']['name'].upper()} (shortest wait: {shortest['remaining']:.1f}s)")
            
            # Wait for the shortest backoff to expire
            import time
            if shortest['remaining'] > 0:
                logger.info(f"⏳ Waiting {shortest['remaining']:.1f}s for {shortest['provider']['name'].upper()} to become available...")
                time.sleep(shortest['remaining'])
            
            return [shortest['provider']]
        
        return available
    
    def get_completion(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 500):
        """Get completion with automatic fallback and exponential backoff"""
        last_error = None
        attempts = 0
        max_attempts = len(self.providers) * 2  # Allow retries
        
        while attempts < max_attempts:
            attempts += 1
            
            # Get available providers (not backed off)
            available_providers = self._get_available_providers()
            
            if not available_providers:
                # This should not happen due to logic in _get_available_providers
                raise RuntimeError("No providers available and all are backed off")
            
            # Try each available provider
            for provider in available_providers:
                provider_name = provider['name']
                
                try:
                    logger.info(f"🤖 Using {provider_name.upper()} for completion (attempt {attempts}/{max_attempts})")
                    
                    response = provider["client"].chat.completions.create(
                        model=provider["model"],
                        messages=messages,
                        temperature=temperature,
                        max_tokens=min(max_tokens, provider["max_tokens"])
                    )
                    
                    # Success - reset backoff and failure count
                    self._reset_backoff(provider_name)
                    self.failure_counts[provider_name] = 0
                    
                    return response.choices[0].message.content.strip(), provider_name
                    
                except Exception as e:
                    error_msg = str(e)
                    self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
                    
                    # Determine error type and apply appropriate backoff
                    is_rate_limit = self._is_rate_limit_error(error_msg)
                    is_temporary = self._is_temporary_error(error_msg)
                    
                    if is_rate_limit:
                        logger.warning(f"⚠️  {provider_name.upper()} rate limited: {error_msg[:100]}")
                        self._apply_backoff(provider_name, is_rate_limit=True)
                    elif is_temporary:
                        logger.warning(f"⚠️  {provider_name.upper()} temporary error: {error_msg[:100]}")
                        self._apply_backoff(provider_name, is_rate_limit=False)
                    else:
                        logger.warning(f"⚠️  {provider_name.upper()} error: {error_msg[:100]}")
                        # For non-temporary errors, apply minimal backoff
                        if self.backoff_state[provider_name]['consecutive_failures'] < 2:
                            self._apply_backoff(provider_name, is_rate_limit=False)
                    
                    last_error = e
                    
                    # Continue to next provider
                    continue
            
            # If we get here, all available providers failed
            # The next iteration will wait for backed-off providers
            if attempts >= max_attempts:
                break
        
        # All attempts exhausted
        raise RuntimeError(f"All LLM providers failed after {attempts} attempts. Last error: {last_error}")

class BrowserAgent:
    """Custom SOTA browser automation agent with multi-provider LLM fallback and task planning"""
    
    def __init__(self, task: str, max_steps: int = 10, headless: bool = False, enable_streaming: bool = True, thread_id: Optional[str] = None, backend_url: Optional[str] = None):
        self.task = task
        self.max_steps = max_steps
        self.initial_max_steps = max_steps  # Store original for reference
        self.headless = headless
        self.enable_streaming = enable_streaming
        self.browser_pid = None  # Track the browser process ID
        self.thread_id = thread_id  # For pushing updates to backend
        self.backend_url = backend_url or "http://localhost:8000"
        self.actions_taken = []
        self.actions_planned = []  # Track what was planned
        self.actions_succeeded = []  # Track what actually succeeded
        self.actions_failed = []  # Track what failed
        self.screenshots = []
        self.downloads = []
        self.uploaded_files = []
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.llm_manager = LLMManager()
        self.vision_manager = VisionManager()
        self.task_id = str(uuid.uuid4())[:8]
        self.streaming_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Dynamic step adjustment
        self.steps_added = 0  # Track how many steps were added
        self.steps_saved = 0  # Track how many steps were saved
        
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
        
        # SOTA Improvements
        self.context_optimizer = ContextOptimizer()
        self.selector_strategy = SelectorStrategy()
        self.page_stabilizer = PageStabilizer()
        self.dynamic_planner = DynamicPlanner()
        self.vision_optimizer = VisionOptimizer()
        
        logger.info("🚀 SOTA improvements enabled: Context optimization, Multi-strategy selectors, Dynamic planning")
        
    async def __aenter__(self):
        """Initialize browser - ensures only one instance per task"""
        # Track this browser instance
        async with active_browsers_lock:
            active_browsers[self.task_id] = True
            logger.info(f"🌐 Initializing browser for task {self.task_id} (Active: {len(active_browsers)})")
        
        self.playwright = await async_playwright().start()
        
        # Track Chrome PIDs before launch
        import psutil
        chrome_pids_before = set()
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and 'chrome' in proc.info['name'].lower():
                    chrome_pids_before.add(proc.info['pid'])
        except:
            pass
        
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
        
        # Track Chrome PIDs after launch (these are ours)
        self.chrome_pids = set()
        try:
            await asyncio.sleep(0.5)  # Give Chrome time to start
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and 'chrome' in proc.info['name'].lower():
                    pid = proc.info['pid']
                    if pid not in chrome_pids_before:
                        self.chrome_pids.add(pid)
            if self.chrome_pids:
                logger.info(f"🔍 Tracked {len(self.chrome_pids)} Chrome processes for this task")
        except Exception as e:
            logger.debug(f"Could not track Chrome PIDs: {e}")
            self.chrome_pids = set()
        
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
        # Stop streaming first
        if hasattr(self, 'streaming_task') and self.streaming_task:
            try:
                self.streaming_task.cancel()
                try:
                    await self.streaming_task
                except asyncio.CancelledError:
                    pass
                logger.info("🔒 Streaming task stopped")
            except Exception as e:
                logger.warning(f"Error stopping streaming: {e}")
        
        # Close context
        try:
            if self.context:
                await self.context.close()
                logger.info("🔒 Browser context closed")
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
        
        # Close browser
        try:
            if self.browser:
                await self.browser.close()
                logger.info("🔒 Browser closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        
        # Stop playwright
        try:
            if hasattr(self, 'playwright') and self.playwright:
                await self.playwright.stop()
                logger.info("🔒 Playwright stopped")
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
        
        # Force kill any remaining Chrome processes for this task (only our PIDs)
        if hasattr(self, 'chrome_pids') and self.chrome_pids:
            try:
                import psutil
                # Give processes a moment to close gracefully
                await asyncio.sleep(0.5)
                
                # Check which of our tracked PIDs are still running
                still_running = []
                for pid in self.chrome_pids:
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            still_running.append(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Kill only our tracked processes that are still running
                if still_running:
                    logger.warning(f"⚠️  {len(still_running)} Chrome processes still running, force killing")
                    for pid in still_running:
                        try:
                            proc = psutil.Process(pid)
                            proc.kill()
                            logger.debug(f"Killed Chrome process {pid}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    logger.info("✅ Cleaned up tracked Chrome processes")
            except Exception as e:
                logger.debug(f"Chrome process cleanup failed: {e}")
        
        # Remove from active browsers
        async with active_browsers_lock:
            if self.task_id in active_browsers:
                del active_browsers[self.task_id]
            logger.info(f"🔒 Browser cleanup complete (Active: {len(active_browsers)})")
    
    async def _handle_download(self, download):
        """Handle file downloads automatically using standardized file manager"""
        try:
            # Get the download content
            temp_path = await download.path()
            if not temp_path:
                logger.warning("Download path not available")
                return
            
            # Read the downloaded content
            with open(temp_path, 'rb') as f:
                content = f.read()
            
            # Register with standardized file manager
            metadata = await download_file_manager.register_file(
                content=content,
                filename=download.suggested_filename,
                file_type=FileType.DOWNLOAD,
                thread_id=self.thread_id,
                custom_metadata={
                    "task_id": self.task_id,
                    "url": download.url,
                    "source_page": self.page.url if self.page else "",
                    "task": self.task
                },
                tags=["download", "browser", f"task:{self.task_id}"]
            )
            
            # Store file path for backward compatibility
            self.downloads.append(metadata.storage_path)
            
            logger.info(f"📥 Downloaded file: {metadata.file_id} ({download.suggested_filename})")
        except Exception as e:
            logger.warning(f"Failed to save download: {e}")
    
    async def _capture_screenshot_robust(self, timeout_attempts=[15000, 5000, 3000]) -> Optional[bytes]:
        """Capture screenshot with multiple timeout strategies"""
        for timeout_val in timeout_attempts:
            try:
                return await self.page.screenshot(
                    timeout=timeout_val,
                    animations="disabled" if timeout_val > 5000 else None
                )
            except Exception as e:
                if timeout_val == timeout_attempts[-1]:
                    logger.warning(f"⚠️  Screenshot failed after {len(timeout_attempts)} attempts")
                    return None
                continue
        return None
    
    async def capture_screenshot(self, name: str = "screenshot") -> str:
        """Capture and save screenshot using standardized file manager"""
        filename = f"{self.task_id}_{name}_{len(self.screenshots)}.png"
        
        screenshot_bytes = await self._capture_screenshot_robust()
        if screenshot_bytes:
            # Register with standardized file manager
            metadata = await screenshot_file_manager.register_file(
                content=screenshot_bytes,
                filename=filename,
                file_type=FileType.SCREENSHOT,
                mime_type="image/png",
                thread_id=self.thread_id,
                custom_metadata={
                    "task_id": self.task_id,
                    "step": len(self.screenshots),
                    "name": name,
                    "url": self.page.url if self.page else "",
                    "task": self.task
                },
                tags=["screenshot", "browser", f"task:{self.task_id}"]
            )
            
            self.screenshots.append(metadata.storage_path)
            logger.info(f"📸 Screenshot saved: {metadata.file_id} ({filename})")
            return metadata.storage_path
        
        logger.warning(f"⚠️  Screenshot failed, continuing without it")
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
                # For arrays, we need a slightly different approach
                json_str = json_match.group()
                try:
                    plan = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try cleaning
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    plan = json.loads(json_str)
                logger.info(f"📋 Created task plan with {len(plan)} subtasks")
                for i, subtask in enumerate(plan, 1):
                    logger.info(f"   {i}. {subtask.get('subtask', 'Unknown')}")
                
                # Dynamically adjust max_steps based on plan complexity
                self._adjust_steps_for_plan(plan)
                
                return plan
            else:
                logger.warning("Could not parse task plan, using single-task mode")
                return [{"subtask": self.task, "status": "pending"}]
                
        except Exception as e:
            logger.error(f"Error creating task plan: {e}")
            return [{"subtask": self.task, "status": "pending"}]
    
    def _adjust_steps_for_plan(self, plan: List[Dict[str, Any]]):
        """Dynamically adjust max_steps based on task plan complexity"""
        subtask_count = len(plan)
        
        # Estimate steps per subtask (average: 5-8 steps per subtask)
        estimated_steps_per_subtask = 6
        estimated_total_steps = subtask_count * estimated_steps_per_subtask
        
        # Add buffer for retries and navigation (20%)
        estimated_total_steps = int(estimated_total_steps * 1.2)
        
        # Adjust max_steps if needed
        if estimated_total_steps > self.max_steps:
            steps_to_add = estimated_total_steps - self.max_steps
            self.max_steps = estimated_total_steps
            self.steps_added = steps_to_add
            logger.info(f"📈 Increased max_steps by {steps_to_add} → {self.max_steps} (based on {subtask_count} subtasks)")
        elif estimated_total_steps < self.max_steps * 0.6:
            # If estimated steps are much less than max, reduce to save time
            steps_to_save = self.max_steps - estimated_total_steps
            self.max_steps = estimated_total_steps
            self.steps_saved = steps_to_save
            logger.info(f"📉 Reduced max_steps by {steps_to_save} → {self.max_steps} (task simpler than expected)")
        else:
            logger.info(f"✅ max_steps ({self.max_steps}) appropriate for {subtask_count} subtasks")
    
    def _adjust_steps_runtime(self, current_step: int):
        """Dynamically adjust max_steps during execution based on progress"""
        completed_count = len([t for t in self.task_plan if t['status'] == 'completed'])
        pending_count = len([t for t in self.task_plan if t['status'] == 'pending'])
        
        if pending_count == 0:
            # All subtasks done - no adjustment needed
            return
        
        # Calculate progress rate (subtasks per step)
        if current_step > 0:
            progress_rate = completed_count / current_step
            
            # Estimate remaining steps needed
            estimated_remaining_steps = int(pending_count / progress_rate) if progress_rate > 0 else pending_count * 6
            
            # Add buffer (30%)
            estimated_remaining_steps = int(estimated_remaining_steps * 1.3)
            
            # Check if we need more steps
            steps_remaining = self.max_steps - current_step
            
            if estimated_remaining_steps > steps_remaining:
                # Need more steps
                steps_to_add = estimated_remaining_steps - steps_remaining
                # Cap at reasonable limit (don't add more than 20 steps at once)
                steps_to_add = min(steps_to_add, 20)
                
                self.max_steps += steps_to_add
                self.steps_added += steps_to_add
                logger.info(f"📈 Added {steps_to_add} more steps → {self.max_steps} ({pending_count} subtasks remaining, progress rate: {progress_rate:.2f})")
            
            elif estimated_remaining_steps < steps_remaining * 0.5 and steps_remaining > 10:
                # Making great progress - can reduce steps
                steps_to_remove = min(steps_remaining - estimated_remaining_steps, 10)
                
                self.max_steps -= steps_to_remove
                self.steps_saved += steps_to_remove
                logger.info(f"📉 Removed {steps_to_remove} steps → {self.max_steps} (ahead of schedule, {pending_count} subtasks remaining)")
    
    def _is_duplicate_extraction(self, extracted_data: Dict[str, Any]) -> bool:
        """Check if extraction is duplicate using content hashing"""
        import hashlib
        
        if not hasattr(self, 'extracted_data'):
            self.extracted_data = []
        
        # Create content hash
        content_for_hash = f"{extracted_data.get('url')}|{extracted_data.get('text_content', '')[:500]}|{extracted_data.get('vision_analysis', '')[:200]}"
        content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
        
        # Check for duplicates
        for existing in self.extracted_data:
            existing_content = f"{existing.get('url')}|{existing.get('text_content', '')[:500]}|{existing.get('vision_analysis', '')[:200]}"
            existing_hash = hashlib.md5(existing_content.encode()).hexdigest()
            
            if content_hash == existing_hash:
                logger.info(f"⚠️  Skipping duplicate extraction (hash match) for {extracted_data.get('url')}")
                return True
            
            # Check for near-duplicates (same URL and very similar content)
            if existing.get('url') == extracted_data.get('url'):
                existing_text = existing.get('text_content', '')[:200]
                new_text = extracted_data.get('text_content', '')[:200]
                if existing_text == new_text:
                    logger.info(f"⚠️  Skipping near-duplicate extraction for {extracted_data.get('url')}")
                    return True
        
        return False
    
    def _validate_extraction_quality(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if extracted data meets quality thresholds.
        
        Returns:
            dict with 'is_sufficient' (bool) and 'reason' (str)
        """
        # Check for structured items (best quality)
        if extracted_data.get('structured_items'):
            item_count = len(extracted_data['structured_items'])
            if item_count >= 3:
                return {'is_sufficient': True, 'reason': f'Found {item_count} structured items'}
            else:
                return {'is_sufficient': False, 'reason': f'Only {item_count} structured items (need 3+)'}
        
        # Check for vision analysis (good quality for image tasks)
        if extracted_data.get('vision_analysis'):
            analysis_length = len(extracted_data['vision_analysis'])
            if analysis_length >= 100:
                return {'is_sufficient': True, 'reason': 'Vision analysis present'}
        
        # Check text content (minimum quality)
        text_content = extracted_data.get('text_content', '')
        headings = extracted_data.get('headings', [])
        
        if len(text_content) >= 500 and len(headings) >= 3:
            error_indicators = ['error', 'not found', '404', 'access denied', 'forbidden']
            if not any(indicator in text_content.lower()[:200] for indicator in error_indicators):
                return {'is_sufficient': True, 'reason': 'Sufficient text content and headings'}
        
        # Insufficient data
        return {
            'is_sufficient': False,
            'reason': f'Insufficient data: {len(text_content)} chars text, {len(headings)} headings'
        }
    
    async def _extract_with_vision_fallback(self, page_content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enhanced vision-based extraction with intelligent fallback.
        
        Three-phase approach:
        1. Vision analyzes page and suggests selectors
        2. Test suggested selectors and extract via DOM (fast and accurate)
        3. If selectors fail, use pure vision extraction (slower but robust)
        
        No artificial time limits - each phase runs until it succeeds or fails naturally.
        """
        try:
            # Capture screenshot for vision analysis
            screenshot_bytes = await self.page.screenshot(timeout=5000, animations="disabled")
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Phase 1: Vision identifies items and suggests selectors
            logger.info("🔍 Phase 1: Vision analyzing page structure...")
            
            vision_prompt = f"""Analyze this webpage screenshot to help extract structured data.

Look at the page and identify any REPEATED VISUAL PATTERNS - items that appear multiple times with similar layout.
These could be: products, articles, posts, search results, listings, cards, tiles, etc.

For EACH repeated item you can see (count up to 15), extract:
- Any visible text (titles, headings, labels)
- Numbers (prices, ratings, scores, dates)
- Visual characteristics (position, size, layout)

Also analyze the HTML structure and suggest CSS selectors that might match these items.
Look for patterns in:
- data-* attributes (data-asin, data-id, data-component-type)
- class names with patterns (product-, item-, card-, result-, post-)
- semantic HTML (article, section with specific classes)

Current page: {page_content.get('url')}

Respond with JSON (no markdown, just JSON):
{{
    "items_found": <number of repeated items you see>,
    "page_type": "e-commerce|blog|news|social|search-results|gallery|list|other",
    "suggested_selectors": [
        "most specific selector",
        "alternative selector",
        "fallback selector"
    ],
    "items": [
        {{
            "position": 1,
            "title": "main text/heading",
            "price": "price/cost if visible",
            "rating": "rating/score if visible", 
            "description": "secondary text",
            "visual_location": "top|middle|bottom",
            "likely_container_class": "pattern you notice"
        }}
    ]
}}

IMPORTANT: 
- Count ALL repeated items you see, even if some details are unclear
- If you see 10+ similar items, that's good - set items_found accordingly
- Suggest multiple selector options
- If fewer than 3 repeated items visible, set items_found to 0"""

            # Try vision providers with fallback
            provider, wait_time = self.vision_manager.get_available_provider()
            if not provider:
                logger.warning("⚠️  No vision providers available")
                return None
            
            if wait_time and wait_time > 0:
                import time
                time.sleep(wait_time)
            
            provider_name = provider['name']
            
            try:
                if provider_name == "ollama":
                    client = OpenAI(api_key=OLLAMA_API_KEY, base_url="https://ollama.com/v1")
                    model = "qwen3-vl:235b-cloud"
                elif provider_name == "nvidia_vision":
                    client = OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")
                    model = "mistralai/mistral-7b-instruct-v0.3"
                else:
                    logger.warning(f"Unknown vision provider: {provider_name}")
                    return None
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                        ]
                    }],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                
                if not json_match:
                    logger.warning("⚠️  Vision response didn't contain valid JSON")
                    self.vision_manager.record_failure(provider_name, "Invalid JSON response")
                    return None
                
                vision_result = parse_json_robust(json_match.group(), content)
                if not vision_result:
                    logger.warning("⚠️  Failed to parse vision JSON")
                    logger.debug(f"Vision response: {content[:500]}")
                    self.vision_manager.record_failure(provider_name, "JSON parse failed")
                    return None
                
                self.vision_manager.record_success(provider_name)
                
                items_found = vision_result.get('items_found', 0)
                items_list = vision_result.get('items', [])
                
                # Use actual items list length if items_found is 0 but items exist
                if items_found == 0 and items_list:
                    items_found = len(items_list)
                    logger.info(f"📊 Corrected items_found from 0 to {items_found} based on items list")
                
                # Check both reported count and actual list
                actual_count = max(items_found, len(items_list))
                
                if actual_count < 3:
                    logger.info(f"📄 Insufficient items for structured extraction: reported={items_found}, list={len(items_list)}")
                    logger.debug(f"Page type: {vision_result.get('page_type')}, Selectors: {vision_result.get('suggested_selectors')}")
                    return None
                
                logger.info(f"✅ Vision identified {items_found} items on page")
                
                # Phase 2: Test suggested selectors and extract via DOM
                suggested_selectors = vision_result.get('suggested_selectors', [])
                if suggested_selectors:
                    logger.info(f"🔍 Phase 2: Testing {len(suggested_selectors)} suggested selectors...")
                    
                    for selector in suggested_selectors:
                        try:
                            # Test if selector finds elements (reasonable timeout to prevent hanging)
                            elements = await asyncio.wait_for(
                                self.page.query_selector_all(selector),
                                timeout=5.0  # Prevent hanging on bad selectors
                            )
                            
                            if len(elements) >= 3:
                                logger.info(f"✅ Selector '{selector}' found {len(elements)} elements - extracting...")
                                
                                # Extract data using this selector
                                extracted_items = await asyncio.wait_for(
                                    self.page.evaluate(f"""
                                    (selector) => {{
                                        const items = Array.from(document.querySelectorAll(selector));
                                        return items.slice(0, 10).map((item, idx) => {{
                                            const data = {{ position: idx + 1 }};
                                            
                                            // Extract title
                                            const title = item.querySelector('h1, h2, h3, h4, h5, h6, [class*="title"], [class*="name"]');
                                            if (title) data.title = title.innerText.trim();
                                            
                                            // Extract price
                                            const price = item.querySelector('[class*="price"], [data-price]');
                                            if (price) data.price = price.innerText.trim();
                                            
                                            // Extract rating
                                            const rating = item.querySelector('[class*="rating"], [class*="star"], [aria-label*="star"]');
                                            if (rating) data.rating = rating.innerText.trim() || rating.getAttribute('aria-label');
                                            
                                            // Extract description
                                            const desc = item.querySelector('p, [class*="description"], [class*="summary"]');
                                            if (desc) data.description = desc.innerText.trim().substring(0, 200);
                                            
                                            // Extract link
                                            const link = item.querySelector('a[href]');
                                            if (link) data.link = link.href;
                                            
                                            // Extract image
                                            const img = item.querySelector('img');
                                            if (img) {{
                                                data.image = img.src;
                                                data.image_alt = img.alt;
                                            }}
                                            
                                            return data;
                                        }});
                                    }}
                                    """, selector),
                                    timeout=10.0  # Reasonable timeout for extraction
                                )
                                
                                if extracted_items and len(extracted_items) >= 3:
                                    logger.info(f"🎯 Successfully extracted {len(extracted_items)} items using vision-discovered selector!")
                                    return {
                                        'structured_items': extracted_items,
                                        'item_count': len(extracted_items),
                                        'extraction_type': 'vision_assisted_dom',
                                        'selector_used': selector,
                                        'page_type': vision_result.get('page_type', 'unknown')
                                    }
                            else:
                                logger.debug(f"Selector '{selector}' found only {len(elements)} elements - need 3+")
                                
                        except asyncio.TimeoutError:
                            logger.debug(f"Selector '{selector}' timed out - trying next")
                            continue
                        except Exception as e:
                            logger.debug(f"Selector '{selector}' failed: {e}")
                            continue
                    
                    logger.info("📄 Phase 2: No working selectors found - moving to Phase 3")
                
                # Phase 3: Pure vision extraction as final fallback
                logger.info("🎨 Phase 3: Using pure vision extraction...")
                vision_items = vision_result.get('items', [])
                
                if vision_items and len(vision_items) >= 3:
                    logger.info(f"✅ Extracted {len(vision_items)} items using pure vision analysis")
                    return {
                        'structured_items': vision_items,
                        'item_count': len(vision_items),
                        'extraction_type': 'vision_pure',
                        'page_type': vision_result.get('page_type', 'unknown')
                    }
                
                logger.info("📄 Vision extraction found insufficient items")
                return None
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️  Vision extraction failed: {error_msg[:100]}")
                self.vision_manager.record_failure(provider_name, error_msg)
                return None
                
        except Exception as e:
            logger.error(f"❌ Vision fallback extraction failed: {e}")
            return None
    
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
        try:
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
                
                // Helper to escape CSS selector special characters
                function escapeSelector(str) {
                    if (!str) return '';
                    // Escape special CSS characters: !"#$%&'()*+,./:;<=>?@[\]^`{|}~
                    return str.replace(/([!"#$%&'()*+,.\/:;<=>?@\[\\\]^`{|}~])/g, '\\\\$1');
                }
                
                // Helper to validate selector before using it
                function isValidSelector(selector) {
                    try {
                        document.querySelectorAll(selector);
                        return true;
                    } catch (e) {
                        return false;
                    }
                }
                
                // Helper to generate reliable CSS selector
                function getSelector(el) {
                    // Priority: ID > Name > Data attributes > Type+Class combo > Tag only
                    
                    // Try ID (if valid and unique)
                    if (el.id && !el.id.match(/^[\\d]/) && !el.id.match(/[\\s]/)) {
                        const idSelector = `#${CSS.escape(el.id)}`;
                        if (isValidSelector(idSelector)) {
                            try {
                                if (document.querySelectorAll(idSelector).length === 1) {
                                    return idSelector;
                                }
                            } catch (e) {
                                // Skip if selector fails
                            }
                        }
                    }
                    
                    // Try name attribute (if not too common)
                    if (el.name) {
                        const nameSelector = `[name="${CSS.escape(el.name)}"]`;
                        if (isValidSelector(nameSelector)) {
                            try {
                                if (document.querySelectorAll(nameSelector).length <= 3) {
                                    return nameSelector;
                                }
                            } catch (e) {
                                // Skip if selector fails
                            }
                        }
                    }
                    
                    // Try data attributes (most reliable for modern sites)
                    const dataTestId = el.getAttribute('data-test-id') || el.getAttribute('data-testid') || el.getAttribute('data-test');
                    if (dataTestId) {
                        const dataSelector = `[data-testid="${CSS.escape(dataTestId)}"]`;
                        if (isValidSelector(dataSelector)) {
                            return dataSelector;
                        }
                    }
                    
                    // Build combination selector: tag + type + classes
                    let path = el.tagName.toLowerCase();
                    
                    // Add type for inputs
                    if (el.type && ['input', 'button'].includes(path)) {
                        path += `[type="${el.type}"]`;
                    }
                    
                    // Add stable classes (not random hashes)
                    if (el.className && typeof el.className === 'string') {
                        const classes = el.className.trim().split(/\\s+/)
                            .filter(c => c && 
                                    !c.match(/^[\\d]/) && 
                                    !c.match(/[a-f0-9]{8,}/i) && 
                                    !c.match(/^_/) && 
                                    c.length < 30 &&
                                    c.length > 2)
                            .slice(0, 2);
                        
                        for (const cls of classes) {
                            try {
                                path += '.' + CSS.escape(cls);
                            } catch (e) {
                                // Skip invalid class
                            }
                        }
                    }
                    
                    // Validate final selector
                    if (isValidSelector(path)) {
                        return path;
                    }
                    
                    // Fallback: just the tag name
                    return el.tagName.toLowerCase();
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
        except Exception as e:
            logger.error(f"❌ Error extracting page content: {str(e)[:200]}")
            # Return minimal page data on error
            try:
                page_data = {
                    'url': self.page.url,
                    'title': await self.page.title(),
                    'bodyText': await self.page.evaluate('() => document.body.innerText.substring(0, 3000)'),
                    'interactiveElements': [],
                    'headings': [],
                    'forms': [],
                    'elementCount': {'inputs': 0, 'buttons': 0, 'links': 0, 'total': 0},
                    'viewport': {'width': 1280, 'height': 800, 'scrollY': 0}
                }
                logger.warning("⚠️  Using minimal page data due to extraction error")
            except:
                # Ultimate fallback
                page_data = {
                    'url': self.page.url if self.page else 'unknown',
                    'title': 'Error loading page',
                    'bodyText': '',
                    'interactiveElements': [],
                    'headings': [],
                    'forms': [],
                    'elementCount': {'inputs': 0, 'buttons': 0, 'links': 0, 'total': 0},
                    'viewport': {'width': 1280, 'height': 800, 'scrollY': 0}
                }
                logger.error("❌ Could not extract any page data")
        
        return page_data
    
    def plan_next_action(self, page_content: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        """Use LLM with fallback to plan next action - OPTIMIZED with 70% token reduction"""
        
        # Check if we need to replan dynamically
        agent_state = {
            'consecutive_same_actions': self.consecutive_same_actions,
            'actions_failed': self.actions_failed,
            'actions_taken': self.actions_taken,
            'current_subtask': self._get_current_subtask()
        }
        
        should_replan, reason = self.dynamic_planner.should_replan(agent_state)
        if should_replan:
            logger.warning(f"🔄 Dynamic replanning triggered: {reason}")
            # Create alternative plan (will be implemented in create_task_plan)
            asyncio.create_task(self._replan_task(reason))
        
        # Get current subtask for focused context
        current_subtask = self._get_current_subtask()
        
        # Build RICH context with semantic information
        recent_history = [
            {
                'step': a.get('step'),
                'action': a.get('action'),
                'reasoning': a.get('reasoning', '')[:100],  # Why we did it (truncated)
                'attempted': a.get('params', {}),  # What we tried
                'result_url': a.get('result_url', a.get('url')),
                'result_title': a.get('result_title', ''),
                'result_body_preview': a.get('result_body_preview', '')[:150],  # First 150 chars of page
                'url_changed': a.get('url_changed', False),  # Did navigation happen?
                'title_changed': a.get('title_changed', False),  # Did page change?
                'element_count': a.get('result_element_count', {}),  # How many elements on result page?
                'success': a.get('success', True),
                'duration': f"{a.get('duration', 0):.1f}s"
            }
            for a in (self.actions_succeeded + self.actions_failed)[-3:]
        ]
        
        # Detect repeated failures to same target
        repeated_failure_warning = ""
        if len(recent_history) >= 2:
            # Check if last 2 actions tried the same thing
            last_two = recent_history[-2:]
            if len(last_two) == 2:
                action1, action2 = last_two[0], last_two[1]
                # Check if same action with same params
                if (action1.get('action') == action2.get('action') and 
                    action1.get('attempted') == action2.get('attempted')):
                    # Check if both had error indicators (in title OR body)
                    error_indicators = ['404', 'not found', 'error', 'forbidden', 'access denied']
                    title1 = action1.get('result_title', '').lower()
                    title2 = action2.get('result_title', '').lower()
                    body1 = action1.get('result_body_preview', '').lower()
                    body2 = action2.get('result_body_preview', '').lower()
                    
                    has_error1 = any(err in title1 or err in body1 for err in error_indicators)
                    has_error2 = any(err in title2 or err in body2 for err in error_indicators)
                    
                    if has_error1 and has_error2:
                        repeated_failure_warning = f"\n⚠️ WARNING: You tried '{action1.get('action')}' with params {action1.get('attempted')} TWICE and got errors both times! DO NOT REPEAT THIS. Try something DIFFERENT!"
                    elif not action1.get('url_changed') and not action2.get('url_changed'):
                        # Actions didn't cause any page change
                        repeated_failure_warning = f"\n⚠️ WARNING: You tried '{action1.get('action')}' TWICE but the page didn't change either time. This action has NO EFFECT. Try something DIFFERENT!"
        
        recent_actions = {
            'last_success': self.actions_succeeded[-1] if self.actions_succeeded else {},
            'last_failure': self.actions_failed[-1] if self.actions_failed else {},
            'stuck': self.consecutive_same_actions >= 2,
            'recent_history': recent_history,
            'repeated_failure_warning': repeated_failure_warning
        }
        
        # Use ContextOptimizer for compact prompt
        prompt = self.context_optimizer.build_compact_context(
            task=self.task,
            current_subtask=current_subtask or {'subtask': self.task},
            page_content=page_content,
            recent_actions=recent_actions,
            step_num=step_num,
            max_steps=self.max_steps
        )
        
        # Add critical instructions with completed_subtask guidance
        prompt += f"""

TASK PLAN STATUS:
{self._format_task_plan_status()}

CRITICAL RULES:
1. Use EXACT selectors from elements list
2. If stuck (same action 2+ times), try different approach or skip
3. **IMPORTANT**: After EVERY successful action, check if it completes a subtask
4. Use "done" only when ALL subtasks are complete

SUBTASK COMPLETION:
- If your action will complete the current subtask, set "completed_subtask" to the EXACT subtask name
- Examples:
  * After extracting data → mark "extract" or "locate" subtasks as complete
  * After navigating to page → mark "navigate" or "go to" subtasks as complete
  * After clicking button → mark "click" or "select" subtasks as complete
- Be proactive: If action achieves subtask goal, mark it complete!

ACTIONS: navigate, click, type, scroll, extract, done

RESPONSE FORMAT (JSON only, no markdown):
{{
  "action": "navigate|click|type|scroll|extract|done",
  "params": {{}},
  "reasoning": "why this action",
  "completed_subtask": "exact subtask name if this action completes it (or null)",
  "is_complete": false
}}"""
        
        # Now use the optimized prompt (already built above)
        # The old massive prompt code has been removed
        
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
                action_plan = parse_json_robust(json_match.group(), content)
                if not action_plan:
                    logger.warning(f"⚠️  {provider.upper()} returned unparseable JSON")
                    return {"action": "done", "is_complete": True, "result_summary": "Could not parse action"}
                logger.info(f"✅ Action planned by {provider.upper()}")
                return action_plan
            else:
                logger.warning(f"No JSON found in response: {content[:200]}")
                return {"action": "done", "is_complete": True, "result_summary": "Could not parse action"}
                
        except Exception as e:
            logger.error(f"❌ All LLM providers failed: {e}")
            return {"action": "done", "is_complete": True, "result_summary": f"LLM Error: {e}"}
    
    def _get_current_subtask(self) -> Optional[Dict]:
        """Get the current pending subtask"""
        if not self.task_plan:
            return None
        for subtask in self.task_plan:
            if subtask['status'] == 'pending':
                return subtask
        return None
    
    def _format_task_plan_status(self) -> str:
        """Format task plan status for LLM context"""
        if not self.task_plan:
            return "No subtasks defined"
        
        status_lines = []
        for i, subtask in enumerate(self.task_plan, 1):
            status = subtask['status']
            icon = "✅" if status == 'completed' else "❌" if status == 'failed' else "⏳"
            status_lines.append(f"{icon} {i}. {subtask['subtask']} [{status}]")
        
        return "\n".join(status_lines)
    
    async def _replan_task(self, reason: str):
        """Dynamically replan when stuck or failing"""
        logger.info(f"🔄 Replanning task due to: {reason}")
        
        completed = [t['subtask'] for t in self.task_plan if t['status'] == 'completed']
        failed = [t['subtask'] for t in self.task_plan if t['status'] == 'failed']
        
        # Create alternative plan prompt
        replan_prompt = self.dynamic_planner.create_alternative_plan(
            self.task,
            failed,
            completed
        )
        
        try:
            response, _ = self.llm_manager.get_completion([{"role": "user", "content": replan_prompt}])
            new_plan_data = parse_json_robust(response)
            
            if new_plan_data and 'subtasks' in new_plan_data:
                # Update task plan with new approach
                new_plan = [{'subtask': st, 'status': 'pending'} for st in new_plan_data['subtasks']]
                self.task_plan = new_plan
                logger.info(f"✅ Replanned with {len(new_plan)} new subtasks")
        except Exception as e:
            logger.error(f"❌ Replanning failed: {e}")
    
    async def should_use_vision(self, page_content: Dict[str, Any]) -> bool:
        """Decide if vision is needed - OPTIMIZED with rule-based decision (80% cost reduction)"""
        if not VISION_ENABLED:
            return False
        
        # Use VisionOptimizer for smart decision (no LLM call needed)
        agent_state = {
            'consecutive_same_actions': self.consecutive_same_actions
        }
        
        needs_vision = self.vision_optimizer.should_use_vision(
            self.task,
            page_content,
            agent_state
        )
        
        return needs_vision
        
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
        """Single vision API call with fallback chain and exponential backoff: Ollama → NVIDIA → Text-only"""
        
        # Build task plan status
        plan_status = ""
        if self.task_plan:
            plan_status = "\n\nTASK PLAN:\n"
            for i, subtask in enumerate(self.task_plan, 1):
                status_icon = "✅" if subtask['status'] == 'completed' else "⏳"
                plan_status += f"{status_icon} {i}. {subtask['subtask']} [{subtask['status']}]\n"
        
        # Try available vision providers with backoff management
        max_attempts = 3
        for attempt in range(max_attempts):
            provider, wait_time = self.vision_manager.get_available_provider()
            
            if not provider:
                logger.warning("⚠️  No vision providers available, falling back to text-only")
                return None
            
            # If provider is backed off, wait
            if wait_time and wait_time > 0:
                logger.info(f"⏳ Waiting {wait_time:.1f}s for vision {provider['name'].upper()} to become available...")
                import time
                time.sleep(wait_time)
            
            provider_name = provider['name']
            
            try:
                logger.info(f"🎨 Trying vision: {provider_name.upper()} (attempt {attempt + 1}/{max_attempts})")
                
                if provider_name == "ollama":
                    result = await self._call_vision_model_ollama(screenshot_base64, page_content, step_num, plan_status)
                elif provider_name == "nvidia_vision":
                    result = await self._call_vision_model_nvidia(screenshot_base64, page_content, step_num, plan_status)
                else:
                    logger.warning(f"Unknown vision provider: {provider_name}")
                    continue
                
                if result:
                    logger.info(f"✅ Vision success: {provider_name.upper()}")
                    self.vision_manager.record_success(provider_name)
                    return result
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️  Vision {provider_name.upper()} failed: {error_msg[:100]}")
                self.vision_manager.record_failure(provider_name, error_msg)
                
                # Continue to next attempt/provider
                continue
        
        # All attempts exhausted
        logger.warning("⚠️  All vision providers failed after retries, falling back to text-only")
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
            
            # Build adaptive context for stuck situations
            stuck_context = ""
            if self.consecutive_same_actions >= 2:
                # Get last action params for feedback
                last_params = self.actions_taken[-1].get('params', {}) if self.actions_taken else {}
                last_coords = f"({last_params.get('x', 'N/A')}, {last_params.get('y', 'N/A')})"
                
                stuck_context = f"""

⚠️ ADAPTIVE REASONING REQUIRED - YOU ARE STUCK!
Consecutive same actions: {self.consecutive_same_actions}/5 (circuit breaker at 5)
Last action: {self.last_action_type}
Last coordinates: {last_coords}
Recent failures: {json.dumps([{'action': a.get('action'), 'url': a.get('url', '')[:50]} for a in self.actions_failed[-3:]], indent=2) if self.actions_failed else 'None'}

🚨 CRITICAL: The element you clicked before did NOT work!

You MUST try a DIFFERENT approach:
1. Click a DIFFERENT element (not the same coordinates)
2. Try a DIFFERENT method (navigate directly to URL instead of clicking)
3. Use text search or keyboard shortcuts instead
4. SKIP this step if it's blocking progress

DO NOT:
- Click the same element again
- Use the same coordinates
- Repeat the same failed action

ALTERNATIVE STRATEGIES:
- Navigate directly: Use "navigate" action with full URL
- Try JavaScript: Some elements need JS interaction
- Skip and continue: Mark subtask as impossible and move on
- Use different selector: Try text-based or XPath selectors
"""
            
            prompt = f"""{image_analysis_instruction}

Analyze this webpage screenshot and provide a COMPLETE response with:
1. Bounding boxes of UI elements
2. Next action with coordinates
3. Whether NEXT step needs vision or text-only
4. Plan updates if needed

TASK: {self.task}
{plan_status}
{stuck_context}

CURRENT PAGE:
URL: {page_content.get('url')}
Title: {page_content.get('title')}
STEP: {step_num}/{self.max_steps}

ACTION HISTORY:
Recent successes: {json.dumps([{'action': a.get('action')} for a in self.actions_succeeded[-3:]], indent=2) if self.actions_succeeded else 'None'}
Recent failures: {json.dumps([{'action': a.get('action')} for a in self.actions_failed[-3:]], indent=2) if self.actions_failed else 'None'}

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

SUBTASK COMPLETION (completed_subtask):
- **CRITICAL**: After EVERY action, check if it completes a subtask from the task plan above
- Set "completed_subtask" to the EXACT subtask name if your action achieves its goal
- Examples:
  * After extracting/locating data → mark "extract", "locate", "find", "identify", "get", "retrieve" subtasks complete
  * After navigating → mark "navigate", "go to", "visit", "open" subtasks complete  
  * After clicking → mark "click", "select", "choose", "press" subtasks complete
- Be proactive! If action achieves the subtask goal, mark it complete immediately

PLAN UPDATES:
- Use "new_subtask" if you discover additional work needed
- Use "completed_subtask" when you finish a subtask (BE AGGRESSIVE about marking complete!)

ACTIONS:
- click: Click at (x, y) - buttons, checkboxes, links
- type: Type text into input field - MUST include "text" parameter with what to type
- drag: Drag from (x, y) to (x2, y2) - sliders, drag-drop
- hover: Hover at (x, y) - dropdowns, tooltips
- scroll_to: Scroll to (x, y)
- extract: Extract visible data
- navigate: Go to URL
- done: Task complete

CRITICAL FOR TYPE ACTION:
- ALWAYS include "text" parameter with the actual text to type
- Example: {{"action": "type", "params": {{"x": 100, "y": 200, "text": "gaming laptops"}}, ...}}
- The "text" field is REQUIRED and must contain the search query or input text

EXAMPLES:
Cloudflare: {{"action": "click", "params": {{"x": 50, "y": 300}}, "next_step_needs_vision": false, "reasoning": "Click checkbox, next page will be normal"}}
Search: {{"action": "type", "params": {{"x": 100, "y": 200, "text": "gaming laptops"}}, "next_step_needs_vision": false, "reasoning": "Type search query"}}
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
            
            # Extract JSON with robust parsing
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = parse_json_robust(json_match.group(), content)
                if not result:
                    logger.error("❌ Ollama returned unparseable JSON")
                    raise ValueError("Failed to parse Ollama vision response")
                
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
            
            # Build adaptive context for stuck situations
            stuck_context = ""
            if self.consecutive_same_actions >= 2:
                # Get last action params for feedback
                last_params = {}
                if self.actions_taken:
                    last_action_data = self.actions_taken[-1]
                    # Try to get params from the action data
                    if isinstance(last_action_data, dict):
                        last_params = last_action_data.get('params', {})
                
                last_coords = f"({last_params.get('x', 'N/A')}, {last_params.get('y', 'N/A')})"
                
                stuck_context = f"""

⚠️ ADAPTIVE REASONING REQUIRED - YOU ARE STUCK!
Consecutive same actions: {self.consecutive_same_actions}/5 (circuit breaker at 5)
Last action: {self.last_action_type}
Last coordinates: {last_coords}
Recent failures: {json.dumps([{'action': a.get('action'), 'url': a.get('url', '')[:50]} for a in self.actions_failed[-3:]], indent=2) if self.actions_failed else 'None'}

🚨 CRITICAL: The element you clicked before did NOT work!

You MUST try a DIFFERENT approach:
1. Click a DIFFERENT element (not the same coordinates)
2. Navigate directly to URL (e.g., https://www.bestbuy.com/site/searchpage.jsp?st=gaming+laptops)
3. Try keyboard shortcuts or text input instead
4. SKIP this step if it's blocking progress

DO NOT:
- Click the same element again
- Use the same coordinates
- Repeat the same failed action

ALTERNATIVE STRATEGIES:
- Navigate directly: Use "navigate" action with search URL
- Try different element: Look for alternative buttons or links
- Skip and continue: Mark subtask as impossible and move on
"""
            
            prompt = f"""{image_analysis_instruction}

Analyze this webpage screenshot and provide a COMPLETE response with:
1. Bounding boxes of UI elements
2. Next action with coordinates
3. Whether NEXT step needs vision or text-only
4. Plan updates if needed

TASK: {self.task}
{plan_status}
{stuck_context}

CURRENT PAGE:
URL: {page_content.get('url')}
Title: {page_content.get('title')}
STEP: {step_num}/{self.max_steps}

ACTION HISTORY:
Recent successes: {json.dumps([{'action': a.get('action')} for a in self.actions_succeeded[-3:]], indent=2) if self.actions_succeeded else 'None'}
Recent failures: {json.dumps([{'action': a.get('action')} for a in self.actions_failed[-3:]], indent=2) if self.actions_failed else 'None'}

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
    "completed_subtask": "EXACT subtask name if this action completes it (or null)",
    "new_subtask": "new subtask to add if discovered additional work",
    "is_complete": false,
    "next_step_needs_vision": true|false,
    "next_step_reasoning": "why vision is/isn't needed for next step"
}}

SUBTASK COMPLETION (completed_subtask):
- **CRITICAL**: After EVERY action, check if it completes a subtask from the task plan above
- Set "completed_subtask" to the EXACT subtask name if your action achieves its goal
- Examples:
  * After extracting/locating data → mark "extract", "locate", "find", "identify", "get", "retrieve" subtasks complete
  * After navigating → mark "navigate", "go to", "visit", "open" subtasks complete  
  * After clicking → mark "click", "select", "choose", "press" subtasks complete
- Be proactive! If action achieves the subtask goal, mark it complete immediately

MODALITY DECISION (next_step_needs_vision):
- true: If next page will have CAPTCHA, challenge, complex visual layout, or few DOM elements
- false: If next page will be standard HTML with good selectors (most websites)

ACTIONS:
- click: Click at (x, y) - buttons, checkboxes, links
- type: Type text into input - MUST include "text" parameter with what to type
- drag: Drag from (x, y) to (x2, y2) - sliders, drag-drop
- hover: Hover at (x, y) - dropdowns, tooltips
- scroll_to: Scroll to (x, y)
- extract: Extract visible data
- analyze_images: Analyze images and describe them (put descriptions in reasoning)
- save_images: Save images from page (put descriptions in reasoning)
- navigate: Go to URL
- done: Task complete

CRITICAL FOR TYPE ACTION:
- ALWAYS include "text" parameter with the actual text to type
- Example: {{"action": "type", "params": {{"x": 100, "y": 200, "text": "gaming laptops"}}, ...}}
- The "text" field is REQUIRED - do not leave it empty!"""

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
            
            # Extract JSON with robust parsing
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = parse_json_robust(json_match.group(), content)
                if not result:
                    logger.error("❌ NVIDIA returned unparseable JSON")
                    raise ValueError("Failed to parse NVIDIA vision response")
                
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
    

    async def _execute_click_robust(self, params: Dict[str, Any]) -> bool:
        """Execute click with SOTA multi-strategy approach"""
        # Wait for page to be stable first
        await self.page_stabilizer.wait_for_stable(self.page)
        
        # Try to dismiss any overlays
        await self.page_stabilizer.dismiss_overlays(self.page)
        
        # Check if we have any usable parameters
        if not any([params.get('selector'), params.get('text'), params.get('x')]):
            logger.error("❌ No selector, text, or coordinates provided for click")
            return False
        
        # Build element info for selector strategy
        element_info = {
            'selector': params.get('selector'),
            'text': params.get('text'),
            'type': params.get('type'),
            'name': params.get('name'),
            'id': params.get('id'),
            'tag': params.get('tag', 'button'),
            'position': {
                'x': params.get('x'),
                'y': params.get('y')
            }
        }
        
        # Use multi-strategy clicking
        success, strategy = await self.selector_strategy.click_with_fallback(
            self.page,
            element_info,
            timeout=5000
        )
        
        if success:
            logger.info(f"✅ Click succeeded using {strategy}")
            await self.page.wait_for_timeout(800)  # Wait for action to complete
            return True
        
        logger.error(f"❌ All click strategies failed")
        return False
    
    def _infer_text_from_context(self) -> str:
        """Smart fallback: Infer what text to type based on task and subtask context"""
        # Get current subtask
        current_subtask = None
        for subtask in self.task_plan:
            if subtask['status'] == 'pending':
                current_subtask = subtask['subtask']
                break
        
        # Common search patterns
        search_patterns = [
            # Pattern: "search for X"
            (r'search for ["\']?([^"\']+)["\']?', 1),
            (r'search ["\']?([^"\']+)["\']?', 1),
            # Pattern: "find X"
            (r'find ["\']?([^"\']+)["\']?', 1),
            # Pattern: "look for X"
            (r'look for ["\']?([^"\']+)["\']?', 1),
            # Pattern: "type X"
            (r'type ["\']?([^"\']+)["\']?', 1),
            # Pattern: "enter X"
            (r'enter ["\']?([^"\']+)["\']?', 1),
            # Pattern: quoted text
            (r'["\']([^"\']{3,})["\']', 1),
        ]
        
        # Try to extract from current subtask first
        if current_subtask:
            for pattern, group in search_patterns:
                match = re.search(pattern, current_subtask, re.IGNORECASE)
                if match:
                    text = match.group(group).strip()
                    logger.info(f"📝 Extracted from subtask: '{text}'")
                    return text
        
        # Try to extract from main task
        for pattern, group in search_patterns:
            match = re.search(pattern, self.task, re.IGNORECASE)
            if match:
                text = match.group(group).strip()
                logger.info(f"📝 Extracted from main task: '{text}'")
                return text
        
        # Check for common product/item names in task
        product_keywords = ['laptop', 'phone', 'headphone', 'mouse', 'keyboard', 'monitor', 'tablet', 'camera']
        for keyword in product_keywords:
            # Look for "gaming laptops", "wireless headphones", etc.
            pattern = rf'(\w+\s+{keyword}s?)'
            match = re.search(pattern, self.task, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                logger.info(f"📝 Extracted product search: '{text}'")
                return text
        
        logger.warning("⚠️  Could not infer text from context")
        return ""
    
    async def _execute_type_robust(self, params: Dict[str, Any]) -> bool:
        """Execute typing with SOTA multi-strategy approach"""
        text = params.get('input_text', '') or params.get('text', '')
        
        # SMART FALLBACK: Infer text from task context if not provided
        if not text:
            logger.warning("⚠️  No text provided in params - attempting to infer from task context")
            text = self._infer_text_from_context()
            
            if text:
                logger.info(f"✅ Inferred text from context: '{text}'")
            else:
                logger.error("❌ No text provided for typing and could not infer from context")
                return False
        
        # Wait for page to be stable
        await self.page_stabilizer.wait_for_stable(self.page)
        
        # Build element info
        element_info = {
            'selector': params.get('selector'),
            'type': params.get('type', 'text'),
            'name': params.get('name'),
            'id': params.get('id'),
            'tag': 'input'
        }
        
        # Use multi-strategy typing
        success, strategy = await self.selector_strategy.type_with_fallback(
            self.page,
            element_info,
            text,
            timeout=5000
        )
        
        if success:
            logger.info(f"✅ Typing succeeded using {strategy}")
            await self.page.wait_for_timeout(500)
            return True
        
        logger.error(f"❌ All typing strategies failed")
        return False
    
    async def execute_action(self, action_plan: Dict[str, Any]) -> bool:
        """Execute the planned action with SOTA robustness and multi-strategy fallbacks"""
        action = action_plan.get("action", "done")
        params = action_plan.get("params", {})
        
        # Adaptive timeout based on action type
        # Extraction can take longer due to vision analysis and selector testing
        action_timeouts = {
            'extract': 120.0,  # 2 minutes for extraction (vision + selectors)
            'analyze_images': 60.0,  # 1 minute for image analysis
            'navigate': 30.0,
            'click': 30.0,
            'type': 30.0,
            'scroll': 15.0,
            'default': 30.0
        }
        
        timeout = action_timeouts.get(action, action_timeouts['default'])
        
        # Wrap execution in timeout to prevent hanging
        try:
            return await asyncio.wait_for(
                self._execute_action_impl(action_plan),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Action '{action}' timed out after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"❌ Action '{action}' failed: {e}")
            return False
    
    async def _execute_action_impl(self, action_plan: Dict[str, Any]) -> bool:
        """Internal implementation of execute_action"""
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
                navigation_succeeded = False
                try:
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                    navigation_succeeded = True
                except Exception as e1:
                    logger.debug(f"domcontentloaded failed: {e1}, trying networkidle")
                    # Fallback: try with networkidle
                    try:
                        await self.page.goto(url, wait_until="networkidle", timeout=10000)
                        navigation_succeeded = True
                    except Exception as e2:
                        logger.debug(f"networkidle failed: {e2}, trying load")
                        # Last resort: just load
                        try:
                            await self.page.goto(url, wait_until="load", timeout=8000)
                            navigation_succeeded = True
                        except Exception as nav_error:
                            logger.error(f"❌ Navigation failed completely: {nav_error}")
                            return False
                
                if not navigation_succeeded:
                    logger.error(f"❌ Navigation to {url} failed")
                    return False
                
                # Wait for page to stabilize (reduced from 2000ms)
                try:
                    await self.page.wait_for_timeout(1000)
                except:
                    pass
                
                # Wait for network to be idle (dynamic content) - don't fail if this times out
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=3000)
                except Exception as e:
                    logger.debug(f"networkidle wait timed out (this is OK): {e}")
                    pass  # Continue even if network doesn't idle
                
                # SEMANTIC VALIDATION: Check if we actually got a valid page
                try:
                    page_title = await self.page.title()
                    page_text = await self.page.inner_text('body')
                    
                    # Check for error indicators
                    error_indicators = [
                        'page not found',
                        '404',
                        'not found',
                        'error',
                        'access denied',
                        'forbidden'
                    ]
                    
                    title_lower = page_title.lower()
                    text_lower = page_text[:500].lower()
                    
                    is_error_page = any(indicator in title_lower or indicator in text_lower 
                                       for indicator in error_indicators)
                    
                    if is_error_page:
                        logger.warning(f"⚠️  Navigation succeeded but landed on ERROR page: '{page_title}'")
                        logger.warning(f"⚠️  This URL appears to be invalid: {url}")
                        # Still return True but log the semantic failure
                        # The LLM will see this in the page title and recent_history
                except Exception as e:
                    logger.debug(f"Could not validate page content: {e}")
                
                logger.info(f"✅ Successfully navigated to {url}")
                return True  # Navigation succeeded
                
            elif action == "click":
                # Use SOTA robust clicking
                logger.info(f"🖱️  Executing SOTA robust click")
                return await self._execute_click_robust(params)
            
            elif action == "type":
                # Use SOTA robust typing
                logger.info(f"⌨️  Executing SOTA robust typing")
                success = await self._execute_type_robust(params)
                if success:
                    # Press Enter after typing (common use case)
                    try:
                        await self.page.keyboard.press("Enter")
                        await self.page.wait_for_timeout(1000)
                        logger.info("✅ Pressed Enter after typing")
                    except:
                        pass
                return success
            
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
                    return True
                elif selector:
                    logger.info(f"📜 Scrolling to selector: {selector}")
                    element = self.page.locator(selector).first
                    await element.scroll_into_view_if_needed()
                    await self.page.wait_for_timeout(1000)
                    return True
                else:
                    logger.error("❌ Scroll_to requires coordinates or selector")
                    return False
                
            elif action == "scroll":
                logger.info("📜 Scrolling page")
                await self.page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
                await self.page.wait_for_timeout(1500)
                return True  # Scroll succeeded
                
            elif action == "extract":
                logger.info("📊 Extracting data from current page")
                
                # CRITICAL FIX: Check if vision was used for this extraction
                vision_reasoning = action_plan.get('reasoning', '')
                vision_keywords = ['image', 'visual', 'doodle', 'logo', 'picture', 'photo', 'illustration', 'graphic']
                used_vision = any(keyword in vision_reasoning.lower() for keyword in vision_keywords)
                
                # Get comprehensive page content
                page_content = await self.get_page_content()
                
                # Build extracted data with proper metadata
                extracted_data = {
                    'url': page_content.get('url'),
                    'title': page_content.get('title'),
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'vision' if used_vision else 'dom',
                }
                
                # CRITICAL FIX: Include vision analysis if available
                if used_vision and vision_reasoning:
                    extracted_data['analysis_type'] = 'image_analysis'
                    extracted_data['vision_analysis'] = vision_reasoning
                    logger.info(f"🎨 Including vision analysis in extracted data")
                else:
                    extracted_data['analysis_type'] = 'text_extraction'
                
                # ENHANCED: Universal structured data extraction with lazy-load handling
                # Step 1: Wait for dynamic content to load
                logger.info("⏳ Waiting for dynamic content to load...")
                try:
                    # Wait for common loading indicators to disappear
                    await self.page.wait_for_function("""
                        () => {
                            const loadingIndicators = document.querySelectorAll('[class*="loading"], [class*="spinner"], [class*="skeleton"]');
                            return loadingIndicators.length === 0 || 
                                   Array.from(loadingIndicators).every(el => el.offsetParent === null);
                        }
                    """, timeout=3000)
                    logger.info("✅ Loading indicators cleared")
                except:
                    logger.debug("No loading indicators found or timeout - continuing")
                
                # Step 2: Scroll to trigger lazy-loaded content
                logger.info("📜 Scrolling to trigger lazy-loaded content...")
                scroll_success = False
                try:
                    # Scroll down in steps to trigger lazy loading
                    for i in range(3):
                        try:
                            # Get current scroll position before
                            before_scroll = await self.page.evaluate("window.pageYOffset")
                            
                            # Scroll down
                            await self.page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
                            await self.page.wait_for_timeout(800)  # Wait for scroll and content load
                            
                            # Verify scroll happened
                            after_scroll = await self.page.evaluate("window.pageYOffset")
                            if after_scroll > before_scroll:
                                logger.debug(f"Scrolled from {before_scroll} to {after_scroll}")
                                scroll_success = True
                            else:
                                logger.debug(f"Scroll {i+1}: No movement detected")
                        except Exception as e:
                            logger.debug(f"Scroll {i+1} failed: {e}")
                    
                    # Scroll back to top to see all loaded content
                    if scroll_success:
                        try:
                            await self.page.evaluate("window.scrollTo(0, 0)")
                            await self.page.wait_for_timeout(500)
                            logger.info("✅ Scrolling complete - content loaded")
                        except Exception as e:
                            logger.warning(f"⚠️  Scroll to top failed: {e}")
                    else:
                        logger.warning("⚠️  Scrolling did not work - page may not be scrollable")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Scrolling failed: {e} - continuing anyway")
                
                # Step 3: Attempt universal structured data extraction
                logger.info("🔍 Attempting universal structured data extraction...")
                try:
                    structured_items = await self.page.evaluate("""
                        () => {
                            // Universal pattern detection - finds repeated structures
                            function findRepeatedPatterns() {
                                const items = [];
                                
                                // Common container patterns for repeated items (expanded)
                                const containerSelectors = [
                                    // E-commerce specific
                                    '[data-asin]', '[data-component-type*="result"]', '[data-component-type*="item"]',
                                    '[class*="s-result"]', '[class*="product"]', '[class*="item-container"]',
                                    // Generic patterns
                                    '[class*="item"]', '[class*="card"]', '[class*="result"]', 
                                    '[class*="listing"]', '[class*="post"]', '[class*="article"]', 
                                    '[class*="entry"]', '[class*="tile"]', '[class*="grid-item"]',
                                    // Data attributes
                                    '[data-testid*="item"]', '[data-testid*="card"]', '[data-testid*="product"]',
                                    '[data-id]', '[data-item-id]', '[data-product-id]',
                                    // Semantic HTML
                                    'article', 'li[class]', '[role="article"]', '[role="listitem"]',
                                    // Div with specific patterns
                                    'div[class*="col-"]', 'div[class*="grid"]', 'div[class*="flex"]'
                                ];
                                
                                // Find containers with multiple similar children
                                let bestContainer = null;
                                let maxSimilarChildren = 0;
                                
                                for (const selector of containerSelectors) {
                                    try {
                                        const elements = Array.from(document.querySelectorAll(selector));
                                        
                                        // Group by parent to find repeated patterns
                                        const parentGroups = {};
                                        elements.forEach(el => {
                                            const parent = el.parentElement;
                                            if (!parent) return;
                                            
                                            const parentKey = parent.tagName + (parent.className || '');
                                            if (!parentGroups[parentKey]) {
                                                parentGroups[parentKey] = { parent, children: [] };
                                            }
                                            parentGroups[parentKey].children.push(el);
                                        });
                                        
                                        // Find parent with most similar children
                                        for (const key in parentGroups) {
                                            const group = parentGroups[key];
                                            if (group.children.length > maxSimilarChildren && group.children.length >= 3) {
                                                maxSimilarChildren = group.children.length;
                                                bestContainer = { selector, elements: group.children };
                                            }
                                        }
                                    } catch (e) {
                                        // Skip invalid selectors
                                    }
                                }
                                
                                if (!bestContainer || bestContainer.elements.length < 3) {
                                    return [];
                                }
                                
                                // Extract data from each item
                                bestContainer.elements.slice(0, 15).forEach((item, idx) => {
                                    const rect = item.getBoundingClientRect();
                                    
                                    // Skip if not visible or too small
                                    if (rect.height < 50 || rect.width < 100) return;
                                    
                                    const itemData = {
                                        index: idx + 1,
                                        type: bestContainer.selector
                                    };
                                    
                                    // Extract text content
                                    const headings = Array.from(item.querySelectorAll('h1, h2, h3, h4, h5, h6, [class*="title"], [class*="heading"], [class*="name"]'));
                                    if (headings.length > 0) {
                                        itemData.title = headings[0].innerText.trim().substring(0, 200);
                                    }
                                    
                                    // Extract description/body text
                                    const descriptions = Array.from(item.querySelectorAll('p, [class*="description"], [class*="summary"], [class*="excerpt"]'));
                                    if (descriptions.length > 0) {
                                        itemData.description = descriptions[0].innerText.trim().substring(0, 300);
                                    }
                                    
                                    // Extract price (if present)
                                    const pricePatterns = ['[class*="price"]', '[data-price]', '[class*="cost"]', '[class*="amount"]'];
                                    for (const pattern of pricePatterns) {
                                        const priceEl = item.querySelector(pattern);
                                        if (priceEl) {
                                            itemData.price = priceEl.innerText.trim();
                                            break;
                                        }
                                    }
                                    
                                    // Extract rating/score (if present)
                                    const ratingPatterns = ['[class*="rating"]', '[class*="star"]', '[class*="score"]', '[aria-label*="star"]'];
                                    for (const pattern of ratingPatterns) {
                                        const ratingEl = item.querySelector(pattern);
                                        if (ratingEl) {
                                            itemData.rating = ratingEl.innerText.trim() || ratingEl.getAttribute('aria-label');
                                            break;
                                        }
                                    }
                                    
                                    // Extract date/time (if present)
                                    const datePatterns = ['time', '[class*="date"]', '[class*="time"]', '[datetime]'];
                                    for (const pattern of datePatterns) {
                                        const dateEl = item.querySelector(pattern);
                                        if (dateEl) {
                                            itemData.date = dateEl.innerText.trim() || dateEl.getAttribute('datetime');
                                            break;
                                        }
                                    }
                                    
                                    // Extract author/source (if present)
                                    const authorPatterns = ['[class*="author"]', '[class*="by"]', '[class*="source"]', '[rel="author"]'];
                                    for (const pattern of authorPatterns) {
                                        const authorEl = item.querySelector(pattern);
                                        if (authorEl) {
                                            itemData.author = authorEl.innerText.trim();
                                            break;
                                        }
                                    }
                                    
                                    // Extract primary link
                                    const linkEl = item.querySelector('a[href]');
                                    if (linkEl) {
                                        itemData.link = linkEl.href;
                                    }
                                    
                                    // Extract image
                                    const imgEl = item.querySelector('img');
                                    if (imgEl) {
                                        itemData.image = imgEl.src;
                                        itemData.image_alt = imgEl.alt;
                                    }
                                    
                                    // Extract any data attributes
                                    const dataAttrs = {};
                                    for (const attr of item.attributes) {
                                        if (attr.name.startsWith('data-')) {
                                            dataAttrs[attr.name] = attr.value;
                                        }
                                    }
                                    if (Object.keys(dataAttrs).length > 0) {
                                        itemData.data_attributes = dataAttrs;
                                    }
                                    
                                    // Extract all text as fallback
                                    if (!itemData.title && !itemData.description) {
                                        itemData.text = item.innerText.trim().substring(0, 500);
                                    }
                                    
                                    // Only add if we extracted meaningful data
                                    if (itemData.title || itemData.description || itemData.text || itemData.link) {
                                        items.push(itemData);
                                    }
                                });
                                
                                return items;
                            }
                            
                            return findRepeatedPatterns();
                        }
                    """)
                    
                    if structured_items and len(structured_items) >= 3:
                        extracted_data['structured_items'] = structured_items
                        extracted_data['item_count'] = len(structured_items)
                        extracted_data['extraction_type'] = 'structured'
                        logger.info(f"✨ Extracted {len(structured_items)} structured items using universal pattern detection")
                    else:
                        logger.info("📄 No repeated patterns detected with DOM extraction")
                        
                        # Step 4: Enhanced vision-based extraction with selector discovery
                        if VISION_ENABLED and not used_vision:
                            logger.info("🎨 Attempting enhanced vision-based extraction...")
                            vision_extraction_result = await self._extract_with_vision_fallback(page_content)
                            
                            if vision_extraction_result:
                                extracted_data.update(vision_extraction_result)
                                logger.info(f"✅ Vision extraction successful: {vision_extraction_result.get('extraction_type')}")
                            else:
                                extracted_data['extraction_type'] = 'unstructured'
                                logger.info("📄 Vision extraction failed - using unstructured extraction")
                        else:
                            extracted_data['extraction_type'] = 'unstructured'
                            logger.info("📄 Using unstructured extraction")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Universal structured extraction failed: {e}")
                    extracted_data['extraction_type'] = 'unstructured'
                
                # Include DOM content
                extracted_data['text_content'] = page_content.get('bodyText', '')[:3000]  # Increased to 3000 chars
                extracted_data['headings'] = page_content.get('headings', [])
                extracted_data['links'] = [elem for elem in page_content.get('interactiveElements', []) if elem.get('isLink')][:15]  # Increased to 15
                extracted_data['images'] = []
                
                # Extract image information with timeout
                try:
                    images = await asyncio.wait_for(
                        self.page.evaluate("""
                            () => {
                                const imgs = Array.from(document.querySelectorAll('img'));
                                return imgs.slice(0, 20).map(img => ({
                                    src: img.src,
                                    alt: img.alt || '',
                                    title: img.title || '',
                                    width: img.width,
                                    height: img.height
                                })).filter(img => img.width > 50 && img.height > 50);
                            }
                        """),
                        timeout=5.0  # 5 second timeout
                    )
                    extracted_data['images'] = images[:15]
                    logger.info(f"📷 Found {len(images)} images on page")
                except asyncio.TimeoutError:
                    logger.warning("⚠️  Image extraction timed out - skipping")
                    extracted_data['images'] = []
                except Exception as e:
                    logger.warning(f"⚠️  Could not extract images: {e}")
                    extracted_data['images'] = []
                
                # Robust deduplication using content hashing
                if not hasattr(self, 'extracted_data'):
                    self.extracted_data = []
                
                is_duplicate = self._is_duplicate_extraction(extracted_data)
                
                if not is_duplicate:
                    # Validate extraction quality
                    extraction_quality = self._validate_extraction_quality(extracted_data)
                    
                    if extraction_quality['is_sufficient']:
                        self.extracted_data.append(extracted_data)
                        # Log what was extracted
                        if used_vision:
                            logger.info(f"✅ Extracted vision analysis: {vision_reasoning[:100]}...")
                        logger.info(f"✅ Extracted data: {len(extracted_data['text_content'])} chars text, {len(extracted_data['images'])} images, {len(extracted_data['headings'])} headings")
                        if extracted_data.get('structured_items'):
                            logger.info(f"📊 Structured items: {len(extracted_data['structured_items'])} items")
                        logger.debug(f"📝 Text preview: {extracted_data['text_content'][:200]}...")
                    else:
                        # Extraction quality is poor - save what we have
                        logger.warning(f"⚠️  Extraction quality insufficient: {extraction_quality['reason']}")
                        self.extracted_data.append(extracted_data)
                        logger.warning("⚠️  Saving low-quality extraction")
                else:
                    logger.info(f"ℹ️  Data already extracted, skipping duplicate")
                
                await self.capture_screenshot("extract")
                return True  # Extract succeeded
            
            elif action == "analyze_images":
                logger.info("🎨 Analyzing images on current page using vision")
                # Vision model already analyzed in planning step
                # The descriptions are in action_plan['reasoning']
                await self.capture_screenshot("analyze_images")
                
                # Store analysis as extracted data
                analysis_text = action_plan.get('reasoning', 'Image analysis completed')
                if not hasattr(self, 'extracted_data'):
                    self.extracted_data = []
                
                # Robust deduplication for analyze_images
                analysis_data = {
                    'url': self.page.url,
                    'title': await self.page.title(),
                    'data_source': 'vision',
                    'analysis_type': 'image_analysis',
                    'vision_analysis': analysis_text,
                    'timestamp': datetime.now().isoformat()
                }
                
                is_duplicate = self._is_duplicate_extraction(analysis_data)
                
                if not is_duplicate:
                    self.extracted_data.append(analysis_data)
                    logger.info(f"✅ Image analysis completed: {analysis_text[:100]}...")
                else:
                    logger.info(f"ℹ️  Analysis already recorded, skipping duplicate")
                
                return True  # Analysis succeeded
            
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
                return True  # Skip is a successful action
                
            elif action == "done":
                logger.info("✅ Task marked as complete")
                return True
            
            else:
                # Invalid action - treat as failure
                logger.error(f"❌ Invalid/unknown action '{action}'")
                return False
                
            # Default: if we reach here, action succeeded
            return True
            
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
        
        result_summary = ""
        
        try:
            # Create initial task plan
            plan_start = datetime.now()
            self.task_plan = await self.create_task_plan()
            self.metrics["llm_calls"] += 1
            
            # Start live screenshot streaming
            self.is_running = True
            if self.enable_streaming:
                self.streaming_task = asyncio.create_task(self.stream_screenshots())
                logger.info("📹 Live screenshot streaming enabled")
            for step in range(1, self.max_steps + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"📍 Step {step}/{self.max_steps}")
                logger.info(f"{'='*60}")
                
                # Get current page state (always fresh - detects dynamic changes)
                page_content = await self.get_page_content()
                logger.info(f"🌐 Current URL: {page_content.get('url')}")
                logger.info(f"📄 Page title: {page_content.get('title')}")
                
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
                
                # Log subtask status
                pending_subtasks = [t for t in self.task_plan if t['status'] == 'pending']
                completed_subtasks = [t for t in self.task_plan if t['status'] == 'completed']
                logger.info(f"📊 Subtask Progress: {len(completed_subtasks)}/{len(self.task_plan)} completed, {len(pending_subtasks)} pending")
                if pending_subtasks:
                    logger.info(f"⏭️  Next pending: {pending_subtasks[0]['subtask']}")
                
                # Plan next action (with vision if needed)
                action_plan = None
                if use_vision:
                    logger.info("🎨 Using VISION for next action planning")
                    # Capture screenshot for vision analysis with robust error handling
                    screenshot_bytes = await self._capture_screenshot_robust()
                    
                    if not screenshot_bytes:
                        logger.warning(f"⚠️  Vision screenshot failed, falling back to text-only")
                        use_vision = False
                    
                    if screenshot_bytes:
                        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                        
                        # Try vision-based planning
                        action_plan = await self.plan_action_with_vision(screenshot_base64, page_content, step)
                        self.metrics["llm_calls"] += 1
                
                # Fallback to text-only if vision failed or not needed
                if not action_plan:
                    logger.info("📝 Using TEXT-ONLY for next action planning")
                    action_plan = self.plan_next_action(page_content, step)
                    self.metrics["llm_calls"] += 1
                
                logger.info(f"💭 Planned Action: {action_plan.get('action')}")
                logger.info(f"💡 Reasoning: {action_plan.get('reasoning', 'No reasoning provided')}")
                
                # Log action parameters
                if action_plan.get('params'):
                    logger.debug(f"🔧 Action params: {action_plan.get('params')}")
                
                # CRITICAL FIX: Prevent planning the same action repeatedly
                current_action_type = action_plan.get('action')
                if len(self.actions_planned) >= 2:
                    last_two_actions = [a.get('action') for a in self.actions_planned[-2:]]
                    if all(a == current_action_type for a in last_two_actions):
                        # Same action planned 3 times in a row
                        if current_action_type in ['analyze_images', 'extract']:
                            logger.warning(f"⚠️  Detected repeated {current_action_type} action - forcing completion")
                            # Force complete the pending subtask
                            for subtask in self.task_plan:
                                if subtask['status'] == 'pending':
                                    subtask['status'] = 'completed'
                                    self.completed_subtasks.append(subtask['subtask'])
                                    logger.info(f"✅ Force-completed subtask to break loop: {subtask['subtask']}")
                                    break
                            # Change action to 'done' if no more pending
                            if not any(t['status'] == 'pending' for t in self.task_plan):
                                action_plan['action'] = 'done'
                                action_plan['is_complete'] = True
                                logger.info(f"🔄 Changed action to 'done' to break loop")
                
                # CRITICAL FIX: Do NOT mark subtasks as completed until AFTER action executes
                # Store the completed_subtask for later processing
                pending_completed_subtask = action_plan.get("completed_subtask")
                
                # Check if LLM wants to add a new subtask mid-way
                if action_plan.get("new_subtask"):
                    new_subtask_text = action_plan["new_subtask"]
                    
                    # CRITICAL FIX: Check for duplicate subtasks before adding
                    is_duplicate_subtask = False
                    for existing in self.task_plan:
                        # Check for exact match or very similar subtasks
                        if existing['subtask'].lower() == new_subtask_text.lower():
                            is_duplicate_subtask = True
                            logger.info(f"⚠️  Skipping duplicate subtask: {new_subtask_text}")
                            break
                        # Check for semantic similarity (same key words)
                        existing_words = set(existing['subtask'].lower().split())
                        new_words = set(new_subtask_text.lower().split())
                        overlap = len(existing_words & new_words) / max(len(existing_words), len(new_words))
                        if overlap > 0.7:  # 70% word overlap = duplicate
                            is_duplicate_subtask = True
                            logger.info(f"⚠️  Skipping similar subtask: {new_subtask_text} (similar to: {existing['subtask']})")
                            break
                    
                    # CRITICAL FIX: Limit total subtasks to prevent runaway task creation
                    MAX_SUBTASKS = 8
                    if len(self.task_plan) >= MAX_SUBTASKS:
                        logger.warning(f"⚠️  Maximum subtasks ({MAX_SUBTASKS}) reached, not adding: {new_subtask_text}")
                    elif not is_duplicate_subtask:
                        self.task_plan.append({
                            "subtask": new_subtask_text,
                            "status": "pending"
                        })
                        logger.info(f"➕ Added new subtask: {new_subtask_text}")
                        self.consecutive_same_actions = 0
                    else:
                        logger.info(f"ℹ️  Subtask already exists, not adding duplicate")
                
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
                
                # Track consecutive same actions - let LLM decide recovery
                current_action = action_plan.get("action")
                current_url = page_content.get('url')
                
                # Only count as stuck if same action on same URL
                if current_action == self.last_action_type and current_url == getattr(self, 'last_action_url', None):
                    self.consecutive_same_actions += 1
                    
                    # CRITICAL FIX: Circuit breaker to prevent infinite loops
                    if self.consecutive_same_actions >= 5:
                        logger.error(f"🛑 CIRCUIT BREAKER: Same action '{current_action}' repeated {self.consecutive_same_actions} times on {current_url}")
                        
                        # Mark current subtask as failed
                        for subtask in self.task_plan:
                            if subtask['status'] == 'pending':
                                subtask['status'] = 'failed'
                                self.failed_subtasks.append({
                                    "subtask": subtask['subtask'],
                                    "reason": f"Stuck in loop - same action repeated {self.consecutive_same_actions} times"
                                })
                                logger.error(f"❌ Marking subtask as FAILED (circuit breaker): {subtask['subtask']}")
                                break
                        
                        # Reset counter
                        self.consecutive_same_actions = 0
                        self.last_action_type = None
                        self.last_action_url = None
                        
                        # Check if any subtasks remain
                        if all(t['status'] != 'pending' for t in self.task_plan):
                            logger.error("🛑 No more subtasks to try - exiting")
                            break
                        
                        # Skip to next iteration to try next subtask
                        logger.info("🔄 Circuit breaker triggered - moving to next subtask")
                        continue
                else:
                    self.consecutive_same_actions = 1
                    self.last_action_type = current_action
                    self.last_action_url = current_url
                
                # Record action WITH SEMANTIC CONTEXT
                self.actions_taken.append({
                    "step": step,
                    "action": action_plan.get("action"),
                    "reasoning": action_plan.get("reasoning"),
                    "url": page_content["url"],
                    "page_title": page_content.get("title", ""),  # ← Add page title
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
                
                # Track planned action
                planned_action = {
                    'step': step,
                    'action': action_plan.get('action'),
                    'reasoning': action_plan.get('reasoning'),
                    'params': action_plan.get('params', {})
                }
                self.actions_planned.append(planned_action)
                
                # Execute action
                logger.info(f"⚡ Executing action: {action_plan.get('action')}")
                action_start_time = datetime.now()
                is_done = await self.execute_action(action_plan)
                action_duration = (datetime.now() - action_start_time).total_seconds()
                
                # Track result - FIXED: properly handle action success/failure
                # is_done can be: True (success), False (failure), or None (extract/done action)
                action_success = is_done if is_done is not None else True
                
                # Get page state AFTER action for semantic context
                post_action_page = await self.get_page_content()
                
                # Extract key semantic indicators from page content
                result_body_text = post_action_page.get('bodyText', '')[:500]  # First 500 chars
                result_element_count = post_action_page.get('elementCount', {})
                
                action_result = {
                    'step': step,
                    'action': action_plan.get('action'),
                    'reasoning': action_plan.get('reasoning', ''),  # ← WHY did we do this?
                    'success': action_success,
                    'duration': action_duration,
                    # BEFORE state
                    'url': page_content.get('url'),
                    'page_title': page_content.get('title', ''),
                    # AFTER state  
                    'result_url': post_action_page.get('url'),
                    'result_title': post_action_page.get('title', ''),
                    'result_body_preview': result_body_text,  # ← Can detect error messages!
                    'result_element_count': result_element_count,  # ← How many interactive elements?
                    # ATTEMPTED action
                    'params': action_plan.get('params', {}),
                    # OUTCOME indicators
                    'url_changed': page_content.get('url') != post_action_page.get('url'),
                    'title_changed': page_content.get('title') != post_action_page.get('title')
                }
                
                if action_success:
                    self.actions_succeeded.append(action_result)
                    logger.info(f"✅ Action succeeded in {action_duration:.2f}s")
                    
                    # CRITICAL FIX: Detect "successful action but no effect"
                    if current_action in ['click', 'type']:
                        # Wait for page to respond
                        await asyncio.sleep(1.5)
                        
                        # Get new page state
                        new_page_content = await self.get_page_content()
                        new_url = new_page_content.get('url')
                        new_hash = hash(new_page_content.get('bodyText', '')[:500])
                        
                        # Check if ANYTHING changed
                        if new_url == current_url and new_hash == current_hash:
                            logger.warning(f"⚠️  Action '{current_action}' succeeded but page didn't change!")
                            
                            # Initialize counter if needed
                            if not hasattr(self, 'no_effect_count'):
                                self.no_effect_count = 0
                            self.no_effect_count += 1
                            
                            if self.no_effect_count >= 3:
                                logger.error(f"🛑 Action has no effect after {self.no_effect_count} attempts - marking as failed")
                                action_success = False
                                self.no_effect_count = 0
                                
                                # Move this to failed actions
                                self.actions_succeeded.pop()  # Remove from succeeded
                                self.actions_failed.append(action_result)
                        else:
                            # Page changed - reset counter
                            if hasattr(self, 'no_effect_count'):
                                self.no_effect_count = 0
                            logger.info(f"✅ Page changed after action - progress detected")
                else:
                    self.actions_failed.append(action_result)
                    logger.error(f"❌ Action failed after {action_duration:.2f}s")
                    
                    # CRITICAL FIX: Mark subtask as FAILED when action fails
                    for subtask in self.task_plan:
                        if subtask['status'] == 'pending':
                            # Track attempts per subtask
                            if 'attempts' not in subtask:
                                subtask['attempts'] = 0
                            subtask['attempts'] += 1
                            
                            # Only mark as failed after multiple attempts
                            if subtask['attempts'] >= 3:
                                subtask['status'] = 'failed'
                                self.failed_subtasks.append({
                                    "subtask": subtask['subtask'],
                                    "reason": f"Action '{action_plan.get('action')}' failed {subtask['attempts']} times"
                                })
                                logger.error(f"❌ Subtask failed after {subtask['attempts']} attempts: {subtask['subtask']}")
                            else:
                                logger.warning(f"⚠️  Subtask attempt {subtask['attempts']}/3 failed: {subtask['subtask']}")
                            break
                
                # CRITICAL FIX: Mark subtasks as completed AFTER action succeeds
                if action_success:
                    # First, check if LLM explicitly marked a subtask as completed
                    if pending_completed_subtask:
                        subtask_name = pending_completed_subtask
                        for subtask in self.task_plan:
                            if subtask_name.lower() in subtask['subtask'].lower() and subtask['status'] == 'pending':
                                subtask['status'] = 'completed'
                                self.completed_subtasks.append(subtask['subtask'])
                                logger.info(f"✅ Completed subtask: {subtask['subtask']}")
                                self.consecutive_same_actions = 0
                                
                                # Dynamic step adjustment based on progress
                                self._adjust_steps_runtime(step)
                                
                                break
                    # Otherwise, try to auto-match action to pending subtask INTELLIGENTLY
                    else:
                        current_action = action_plan.get("action")
                        action_reasoning = action_plan.get("reasoning", "").lower()
                        
                        # CRITICAL FIX: Smart subtask matching based on intent
                        for subtask in self.task_plan:
                            if subtask['status'] == 'pending':
                                subtask_lower = subtask['subtask'].lower()
                                matched = False
                                
                                # Match navigate actions
                                if current_action == "navigate" and any(word in subtask_lower for word in ["navigate", "go to", "visit", "open"]):
                                    matched = True
                                
                                # Match vision/image analysis subtasks
                                elif current_action in ["extract", "analyze_images"]:
                                    # Check if subtask is about image/vision analysis
                                    vision_keywords = ["analyze", "doodle", "image", "visual", "picture", "logo", "graphic", "look at", "determine", "subject", "about", "what", "identify", "describe"]
                                    is_vision_subtask = any(word in subtask_lower for word in vision_keywords)
                                    
                                    # Check if action used vision
                                    used_vision = any(word in action_reasoning for word in ["image", "visual", "doodle", "logo", "picture", "analyze", "standard google logo", "multicolored"])
                                    
                                    # CRITICAL: analyze_images ALWAYS completes ANY pending vision subtask
                                    if current_action == "analyze_images":
                                        if is_vision_subtask:
                                            matched = True
                                            logger.info(f"🎨 Matched vision subtask '{subtask['subtask']}' with analyze_images action")
                                        else:
                                            # Even if not obviously vision-related, analyze_images should complete SOMETHING
                                            # This prevents infinite loops
                                            logger.warning(f"⚠️  analyze_images completing non-vision subtask: {subtask['subtask']}")
                                            matched = True
                                    # Match if both are vision-related
                                    elif is_vision_subtask and used_vision and current_action == "extract":
                                        matched = True
                                        logger.info(f"🎨 Matched vision subtask with vision-based extract action")
                                    elif not is_vision_subtask and current_action == "extract":
                                        # Text extraction subtask
                                        if any(word in subtask_lower for word in ["extract", "get", "tell me", "find", "content", "locate", "identify", "retrieve", "collect", "gather"]):
                                            matched = True
                                
                                # Match type/search actions
                                elif current_action == "type" and any(word in subtask_lower for word in ["search", "type", "enter", "input"]):
                                    matched = True
                                
                                # Match click actions
                                elif current_action == "click" and any(word in subtask_lower for word in ["click", "select", "choose", "press"]):
                                    matched = True
                                
                                if matched:
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
            
            # Validation: Check if extract actions were performed but no data was extracted
            extract_actions = [a for a in self.actions_taken if a.get('action') == 'extract']
            if extract_actions and not hasattr(self, 'extracted_data'):
                logger.error("🚨 VALIDATION FAILED: Extract actions were performed but no data was extracted!")
                logger.error("🚨 This indicates the LLM may be hallucinating responses!")
            elif extract_actions:
                logger.info(f"✅ Validation passed: {len(extract_actions)} extract actions, {len(getattr(self, 'extracted_data', []))} data entries")
        
        except Exception as e:
            # CRITICAL FIX: Catch any exception in the main loop and return a proper error result
            logger.error(f"❌ Exception in browser agent run loop: {e}")
            import traceback
            traceback.print_exc()
            result_summary = f"Error during execution: {str(e)}"
            
        finally:
            # Stop streaming
            self.is_running = False
            if hasattr(self, 'streaming_task') and self.streaming_task:
                self.streaming_task.cancel()
                try:
                    await self.streaming_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up live screenshots for this task
            async with live_screenshots_lock:
                if self.task_id in live_screenshots:
                    del live_screenshots[self.task_id]
        
        # Prepare extracted data summary
        extracted_data_summary = None
        if hasattr(self, 'extracted_data') and self.extracted_data:
            logger.info(f"📦 Returning {len(self.extracted_data)} extracted data entries")
            extracted_data_summary = {
                'entries': self.extracted_data,
                'total_entries': len(self.extracted_data)
            }
        else:
            logger.warning("⚠️  No data was extracted during this task")
        
        # Determine success based on whether we had an error
        is_success = not result_summary.startswith("Error")
        
        return {
            "success": is_success,
            "task_summary": result_summary or "Completed all steps",
            "summary": result_summary or "Completed all steps",  # Keep for backward compatibility
            "actions": getattr(self, 'actions_taken', []),
            "actions_planned": getattr(self, 'actions_planned', []),
            "actions_succeeded": getattr(self, 'actions_succeeded', []),
            "actions_failed": getattr(self, 'actions_failed', []),
            "action_success_rate": f"{len(getattr(self, 'actions_succeeded', []))}/{len(getattr(self, 'actions_planned', []))}" if getattr(self, 'actions_planned', []) else "0/0",
            "screenshots": getattr(self, 'screenshots', []),
            "downloads": getattr(self, 'downloads', []),
            "uploaded_files": getattr(self, 'uploaded_files', []),
            "extracted_data": extracted_data_summary,
            "task_id": self.task_id,
            "metrics": getattr(self, 'metrics', {})
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
            
            # CRITICAL FIX: Handle None result
            if result is None:
                logger.error("❌ agent.run() returned None - this should not happen")
                return {
                    "success": False,
                    "task_summary": "Error: Browser agent returned no result",
                    "actions_taken": [],
                    "screenshot_files": None,
                    "extracted_data": None,
                    "task_id": None,
                    "error": "Browser agent returned no result"
                }
            
            # Log extracted data if available
            extracted_data = result.get("extracted_data")
            if extracted_data:
                logger.info(f"📦 Task completed with {extracted_data.get('total_entries', 0)} data extractions")
            else:
                logger.warning("⚠️  Task completed but no data was extracted")
            
            return {
                "success": result.get("success", False),
                "task_summary": result.get("summary", result.get("task_summary", "Unknown")),
                "actions_taken": result.get("actions", []),
                "actions_planned": result.get("actions_planned", []),
                "actions_succeeded": result.get("actions_succeeded", []),
                "actions_failed": result.get("actions_failed", []),
                "action_success_rate": result.get("action_success_rate", "0/0"),
                "screenshot_files": result.get("screenshots", []),
                "downloaded_files": result.get("downloads", []),
                "uploaded_files": result.get("uploaded_files", []),
                "extracted_data": extracted_data,
                "task_id": result.get("task_id"),
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


# ============== STANDARDIZED FILE MANAGEMENT ENDPOINTS ==============

class FileListResponse(BaseModel):
    success: bool
    files: List[Dict[str, Any]]
    count: int
    error: Optional[str] = None

class FileStatsResponse(BaseModel):
    success: bool
    screenshots: Dict[str, Any]
    downloads: Dict[str, Any]
    error: Optional[str] = None

@app.get("/files/screenshots", response_model=FileListResponse)
async def list_screenshots(
    status: Optional[str] = None,
    thread_id: Optional[str] = None
):
    """
    List all screenshot files managed by this agent.
    """
    try:
        file_status = FileStatus(status) if status else FileStatus.ACTIVE
        files = screenshot_file_manager.list_files(
            status=file_status,
            thread_id=thread_id
        )
        
        return FileListResponse(
            success=True,
            files=[f.to_orchestrator_format() for f in files],
            count=len(files)
        )
    except Exception as e:
        logger.error(f"Failed to list screenshots: {e}")
        return FileListResponse(success=False, files=[], count=0, error=str(e))


@app.get("/files/downloads", response_model=FileListResponse)
async def list_downloads(
    status: Optional[str] = None,
    thread_id: Optional[str] = None
):
    """
    List all downloaded files managed by this agent.
    """
    try:
        file_status = FileStatus(status) if status else FileStatus.ACTIVE
        files = download_file_manager.list_files(
            status=file_status,
            thread_id=thread_id
        )
        
        return FileListResponse(
            success=True,
            files=[f.to_orchestrator_format() for f in files],
            count=len(files)
        )
    except Exception as e:
        logger.error(f"Failed to list downloads: {e}")
        return FileListResponse(success=False, files=[], count=0, error=str(e))


@app.get("/files/{file_type}/{file_id}")
async def get_file_info(file_type: str, file_id: str):
    """
    Get detailed information about a specific file.
    
    Parameters:
        file_type: Either 'screenshots' or 'downloads'
        file_id: The file ID
    """
    try:
        if file_type == "screenshots":
            metadata = screenshot_file_manager.get_file(file_id)
        elif file_type == "downloads":
            metadata = download_file_manager.get_file(file_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid file_type. Use 'screenshots' or 'downloads'")
        
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found or expired")
        
        return {"success": True, "file": metadata.to_orchestrator_format()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/files/{file_type}/{file_id}")
async def delete_file(file_type: str, file_id: str):
    """
    Delete a file from the agent's storage.
    """
    try:
        if file_type == "screenshots":
            success = screenshot_file_manager.delete_file(file_id)
        elif file_type == "downloads":
            success = download_file_manager.delete_file(file_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid file_type. Use 'screenshots' or 'downloads'")
        
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {"success": True, "message": f"File {file_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        return {"success": False, "error": str(e)}


@app.post("/cleanup")
async def cleanup_files(max_age_hours: int = 24):
    """
    Clean up old/expired files from both screenshots and downloads.
    """
    try:
        # Cleanup screenshots
        screenshot_expired = screenshot_file_manager.cleanup_expired()
        screenshot_old = screenshot_file_manager.cleanup_old(max_age_hours=max_age_hours)
        
        # Cleanup downloads (use longer default for downloads)
        download_expired = download_file_manager.cleanup_expired()
        download_old = download_file_manager.cleanup_old(max_age_hours=max_age_hours * 3)
        
        return {
            "success": True,
            "screenshots": {
                "expired_removed": screenshot_expired,
                "old_removed": screenshot_old
            },
            "downloads": {
                "expired_removed": download_expired,
                "old_removed": download_old
            }
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/stats", response_model=FileStatsResponse)
async def get_file_stats():
    """
    Get file management statistics for both screenshots and downloads.
    """
    try:
        screenshot_stats = screenshot_file_manager.get_stats()
        download_stats = download_file_manager.get_stats()
        
        return FileStatsResponse(
            success=True,
            screenshots=screenshot_stats,
            downloads=download_stats
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return FileStatsResponse(
            success=False,
            screenshots={},
            downloads={},
            error=str(e)
        )

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
    port = int(os.getenv("BROWSER_AGENT_PORT", 8090))
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
