"""
Browser Agent - Main Agent Orchestrator

Stateful, SOTA browser automation with memory, planning, and vision.
"""

import time
import uuid
import base64
import logging
import asyncio
import psutil
import os
import json
from typing import Dict, Any, List, Optional
import httpx
import websockets

# Suppress noisy httpx logging (canvas updates)
logging.getLogger("httpx").setLevel(logging.WARNING)

from .browser import Browser
from .dom import DOMExtractor
from .actions import ActionExecutor
from .llm import LLMClient
from .vision import VisionClient
from .config import CONFIG

try:
    from agents.utils.agent_file_manager import AgentFileManager, FileType, FileStatus
except ImportError:
    class FileType:
        DOWNLOAD = "download"
        SCREENSHOT = "screenshot"
    class FileStatus:
        ACTIVE = "active"
    AgentFileManager = None
from pathlib import Path
from .schemas import ActionPlan, ActionResult, BrowserResult
from .state import AgentMemory
from .planner import Planner
from .persistent_memory import get_persistent_memory

# Configure logger for this module and children (agents.browser_agent.*)
class IndentedFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        # Check for step header (contains "📍 Step" or starts with heavy separator)
        # We don't indent the main step headers to keep them prominent
        if "📍 Step" in msg or msg.startswith("="*10):
            return msg
        # Indent everything else with one tabspace as requested
        # Also handle multi-line messages so they align nicely
        return "\t" + msg.replace("\n", "\n\t")

logger = logging.getLogger("agents.browser_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = IndentedFormatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)


class BrowserAgent:
    """SOTA browser automation agent with Memory, Planning, and Vision"""
    
    def __init__(self, task: str, headless: bool = False, thread_id: Optional[str] = None, backend_url: Optional[str] = "http://localhost:8000"):
        self.task = task
        self.headless = headless
        self.task_id = str(uuid.uuid4())[:8]
        self.thread_id = thread_id
        self.backend_url = backend_url
        
        # Components
        self.browser = Browser()
        self.dom = DOMExtractor()
        self.llm = LLMClient()
        self.vision = VisionClient()
        self.planner = Planner(self.llm)
        
        # State
        self.memory = AgentMemory(task=task)
        self.start_time = None
        self.next_mode = "text"  # Default start mode
        self.current_action_description = "Initializing..."
        self.is_running = False
        self._recovering = False  # Flag to pause background tasks during recovery
        # Robust Lock for Page Access (prevents race conditions)
        self.page_access_lock = asyncio.Lock()
        self.streaming_task = None
        self.screenshot_ws = None  # WebSocket connection for screenshot streaming
        self.stuck_count = 0  # Track consecutive stuck warnings
        self.previous_url = ""  # Track URL changes to detect progress
        self.recent_downloads = [] # Track downloads in the current step
        self.known_elements = {} # Memory of elements by URL: {url: elem}
        self._active_downloads = set() # Track active background downloads
        
        # Repeated Action Detection
        self._last_action_signature = None  # Hash of last action for duplicate detection
        self._repeated_action_count = 0  # Count of consecutive same-action attempts
        self._last_no_effect_action = None  # Track last action that had no effect
        self._no_effect_count = 0  # Count of consecutive no-effect actions
        self._last_executed_action = None  # Track last action for blocking
        
        self.action_history: List[Dict[str, Any]] = [] # Added this line for blocking
        
        # Persistent Memory (survives across sessions)
        self.persistent = get_persistent_memory()
        logger.info(f"📚 {self.persistent.get_summary()}")
        
        # Initialize File Managers
        self.download_manager = None
        self.screenshot_manager = None
        if AgentFileManager:
            try:
                self.download_manager = AgentFileManager(
                    agent_id="browser_agent_downloads",
                    storage_dir=str(CONFIG.DOWNLOADS_DIR),
                    default_ttl_hours=72,
                    auto_cleanup=True
                )
                self.screenshot_manager = AgentFileManager(
                    agent_id="browser_agent_screenshots",
                    storage_dir=str(CONFIG.SCREENSHOTS_DIR),
                    default_ttl_hours=72,
                    auto_cleanup=True
                )
            except Exception as e:
                logger.warning(f"Failed to init file managers: {e}")
        
        # ActionExecutor with screenshot capability
        self.executor = ActionExecutor(
            screenshot_manager=self.screenshot_manager,
            thread_id=self.thread_id
        )
        
        # Metrics tracking
        self._metrics_start_time = time.time()
        self.metrics = {
            "actions": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "click": 0,
                "type": 0,
                "navigate": 0,
                "scroll": 0,
                "wait": 0,
                "extract": 0,
                "other": 0
            },
            "llm_calls": {
                "total": 0,
                "planning": 0,
                "vision": 0,
                "failures": 0
            },
            "performance": {
                "total_latency_ms": 0,
                "avg_action_ms": 0,
                "actions_completed": 0
            },
            "navigation": {
                "pages_visited": 0,
                "unique_urls": set(),
                "successful_navigations": 0,
                "failed_navigations": 0
            },
            "vision": {
                "screenshots_taken": 0,
                "vision_analyses": 0
            },
            "errors": {
                "total": 0,
                "action_errors": 0,
                "llm_errors": 0,
                "browser_errors": 0
            },
            "resource": {
                "peak_memory_mb": 0,
                "current_memory_mb": 0
            },
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            },
            "dom": {
                "total_elements": 0,
                "snapshots": 0,
                "avg_elements": 0
            }
        }

    def _update_page_knowledge(self, url: str, elements: List[Dict]):
        """Update memory of elements for this URL"""
        if not url: return

        if url not in self.known_elements:
            self.known_elements[url] = {}
        
        memory = self.known_elements[url]
        
        for el in elements:
            # Mark it visible=True since it's in current viewport
            el['visible'] = True 
            el['last_seen'] = time.time()
            
            # Key: prefer xpath but fallback to something unique
            key = el.get('xpath')
            if key:
                memory[key] = el
        
    def _merge_known_elements(self, url: str, current_elements: List[Dict]) -> List[Dict]:
        """Merge current viewport elements with off-screen memory"""
        if not url or url not in self.known_elements:
            # Just ensure visible flag is set
            for el in current_elements: el['visible'] = True
            return current_elements

        memory = self.known_elements[url]
        current_xpaths = {el.get('xpath') for el in current_elements}
        
        merged = []
        
        # 1. Add current elements (Priority: Fresh, definitely visible)
        merged.extend(current_elements)
        
        # 2. Add memory elements NOT in current view
        for xpath, el in memory.items():
            if xpath not in current_xpaths:
                # This element is known but not in current view
                el_copy = el.copy()
                el_copy['visible'] = False
                merged.append(el_copy)
        
        # 3. Sort by Y position (Top to Bottom) to maintain logical flow
        # Use safe get because off-screen elements might have stale coordinates?
        # Actually coordinates are page-absolute, so they are correct relative to page top.
        merged.sort(key=lambda x: x.get('y', 0))
        
        return merged

    async def _wait_for_downloads(self, timeout: int = 30):
        """Wait for active downloads to complete"""
        if not self._active_downloads:
            return

        logger.info(f"⏳ Waiting for {len(self._active_downloads)} active downloads...")
        start_time = time.time()
        
        while self._active_downloads:
            if time.time() - start_time > timeout:
                logger.warning(f"⚠️ Timeout waiting for downloads: {len(self._active_downloads)} remaining")
                break
            await asyncio.sleep(0.5)
        
        if not self._active_downloads:
            logger.info("✅ All downloads finished")

    async def _handle_download(self, download):
        """Handle file download event"""
        download_id = str(uuid.uuid4())
        self._active_downloads.add(download_id)
        try:
            filename = download.suggested_filename
            logger.info(f"📥 Starting download: {filename}")
            
            storage_dir = Path(self.download_manager.storage_dir) if self.download_manager else CONFIG.DOWNLOADS_DIR
            storage_dir.mkdir(parents=True, exist_ok=True)
            target_path = storage_dir / filename
            
            await download.save_as(str(target_path))
            
            # Track this download for the current step logic
            self.recent_downloads.append(str(target_path))
            
            if self.download_manager:
                try:
                    await self.download_manager.register_file(
                        content=None,
                        filename=filename,
                        file_type=FileType.DOWNLOAD,
                        file_path=str(target_path),
                        thread_id=self.thread_id,
                        custom_metadata={"task": self.task}
                    )
                except Exception as reg_err:
                     logger.warning(f"Failed to register download: {reg_err}")

            logger.info(f"✅ Download complete: {filename}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
        finally:
            self._active_downloads.discard(download_id)

    async def _stream_loop(self):
        """Background task for smooth visual streaming (1fps)"""
        logger.error("📹 STARTING STREAM LOOP CHECK")
        logger.info("📹 Starting background stream loop")
        while self.is_running:
            try:
                # CRITICAL: Acquire lock to prevent race with main loop actions
                # Concurrent page access between stream and action causes IPC pipe crashes!
                if self._recovering:
                    await asyncio.sleep(0.5)
                    continue

                async with self.page_access_lock:
                    if self.browser.page:
                        # Capture screenshot to memory (fast, no file save)
                        try:
                            # Use JPEG with aggressive compression for streaming (25% quality, 50% scale)
                            screenshot_bytes = await self.browser.page.screenshot(
                                timeout=2000, 
                                type='jpeg',
                                quality=25,
                                full_page=False,
                                scale='css'  # Use CSS pixels (smaller on high DPI)
                            )
                            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                            
                            # Push update
                            current_step_count = len(self.memory.history)
                            await self._push_state_update(screenshot_b64, current_step_count)
                        except Exception as loop_e:
                            # Ignore screenshot timeouts or page close races
                            pass
                
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Stream loop error: {e}")
                await asyncio.sleep(1.0)
        logger.info("🛑 Stream loop stopped")

    async def _push_state_update(self, screenshot_b64: Optional[str], step: int):
        """Push update to orchestrator via WebSocket. Pass screenshot_b64=None to clear canvas."""
        if not self.thread_id or not self.screenshot_ws:
            return
            
        try:
            # Build payload
            payload = {
                "screenshot_data": screenshot_b64 or "",
                "url": self.browser.page.url if self.browser.page and screenshot_b64 else "",
                "step": step,
                "task_plan": [
                    {"subtask": t.description, "status": t.status} 
                    for t in self.memory.plan
                ] if screenshot_b64 else [],
                "current_action": self.current_action_description if screenshot_b64 else "Session Ended"
            }
            
            # Send via WebSocket (fast, no HTTP overhead)
            await self.screenshot_ws.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Screenshot WebSocket disconnected, attempting reconnect...")
            await self._connect_screenshot_ws()
        except Exception as e:
            logger.error(f"Streaming failed: {e}")

    async def _connect_screenshot_ws(self):
        """Connect to orchestrator's screenshot WebSocket for streaming"""
        if not self.thread_id or not self.backend_url:
            return
        
        try:
            # Convert http:// to ws:// for WebSocket
            ws_url = self.backend_url.replace("http://", "ws://").replace("https://", "wss://")
            ws_endpoint = f"{ws_url}/ws/screenshots/{self.thread_id}"
            
            logger.info(f"📸 Connecting to screenshot WebSocket: {ws_endpoint}")
            self.screenshot_ws = await websockets.connect(ws_endpoint, ping_interval=None)
            logger.info(f"📸 Screenshot WebSocket connected for thread {self.thread_id}")
        except Exception as e:
            logger.error(f"Failed to connect screenshot WebSocket: {e}")
            self.screenshot_ws = None

    async def run(self) -> BrowserResult:
        """Execute the browser automation task"""
        self.start_time = time.time()
        self.is_running = True
        logger.info(f"🚀 Starting SOTA Agent [{self.task_id}]: {self.task}")
        
        # 1. Initialize Browser
        if not await self.browser.launch(headless=self.headless, on_download=self._handle_download):
            return BrowserResult(success=False, task_summary="Browser launch failed", error="Browser launch failed")
        
        # Connect screenshot WebSocket for streaming
        if self.thread_id:
            await self._connect_screenshot_ws()
        
        # Start Streaming Loop
        if self.thread_id:
            self.streaming_task = asyncio.create_task(self._stream_loop())
        
        try:
            # 2. Create Initial Plan
            self.current_action_description = "Planning..."
            logger.info("🧠 Generating execution plan...")
            initial_subtasks = await self.planner.create_initial_plan(self.task)
            self.memory.plan = initial_subtasks
            if not self.memory.plan:
                logger.error("Failed to generate plan")
                self.is_running = False
                return BrowserResult(success=False, task_summary="Planning failed")
            
            logger.info(f"📋 Plan: {[t.description for t in self.memory.plan]}")

            # 3. Execution Loop
            step = 0
            while True:
                step += 1
                logger.info(f"{'='*50}\n📍 Step {step}\n{'='*50}")
                
                # Clear previous step's downloads
                self.recent_downloads = []
                
                # Get Active Subtask
                current_subtask = self.memory.get_active_subtask()
                if not current_subtask:
                    logger.info("🎉 No more subtasks! Task Complete.")
                    break
                logger.info(f"🎯 Current Goal: {current_subtask.description}")

                # Context Gathering - Always use the ACTIVE page (handles tabs)
                self.current_action_description = "Observing page..."
                active_page = self.browser.get_active_page()
                
                if not active_page:
                    logger.warning("⚠️ No active page at step start - recovering...")
                    self._recovering = True
                    try:
                        last_known_url = self.previous_url if self.previous_url.startswith('http') else None
                        active_page = await self.browser.recover_page(last_known_url)
                    finally:
                        self._recovering = False
                
                # CRITICAL: Capture URL immediately for recovery purposes
                # This MUST happen before any operations that might cause context to become stale
                last_error = None  # Track last error for prompt injection
                
                if active_page:
                    try:
                        immediate_url = active_page.url
                        if immediate_url and immediate_url.startswith('http'):
                            self.previous_url = immediate_url
                    except Exception:
                        pass  # Page might already be stale, that's ok
                
                # Brief wait to let any recent tab switch settle
                await asyncio.sleep(0.3)
                
                # Try to get page content with retry if page context is stale
                try:
                    async with self.page_access_lock:
                        page_content = await self.dom.get_page_content(active_page) if active_page else {'url': '', 'elements': [], 'extraction_failed': True}
                except ValueError as e:
                    if "closed pipe" in str(e) or "target closed" in str(e):
                        logger.error(f"⚠️ Browser pipe closed ({e}) - Triggering recovery...")
                        self.browser.page = None # Force recovery next loop
                        await asyncio.sleep(1)
                        continue
                    raise e
                except Exception as e:
                     logger.warning(f"DOM extraction error: {e}")
                     page_content = {'url': '', 'elements': [], 'extraction_failed': True}
                
                # RETRY MECHANISM: If DOM extraction failed (stale context), retry with fresh page reference
                # Check for: empty URL, extraction_failed flag, or 0 elements on non-blank page
                extraction_failed = (
                    page_content.get('extraction_failed', False) or
                    not page_content.get('url') or 
                    page_content.get('url') == ''
                )
                
                if extraction_failed:
                    logger.warning("⚠️ DOM extraction failed - context may be stale, retrying...")
                    
                    # Try to get the last known URL for recovery
                    last_known_url = self.previous_url if self.previous_url.startswith('http') else None
                    
                    # Multiple retries with increasing wait times
                    for retry_wait in [0.5, 1.0, 2.0]:
                        await asyncio.sleep(retry_wait)
                        active_page = self.browser.get_active_page()
                        
                        # If no active page found, try to recover with a new page
                        if not active_page:
                            logger.warning(f"⚠️ All pages are stale - attempting recovery...")
                            self._recovering = True
                            try:
                                active_page = await self.browser.recover_page(last_known_url)
                            finally:
                                self._recovering = False
                            
                            if not active_page:
                                logger.error("❌ Page recovery failed!")
                                continue
                        
                        async with self.page_access_lock:
                            page_content = await self.dom.get_page_content(active_page)
                        if not page_content.get('extraction_failed', False) and page_content.get('url'):
                            logger.info(f"✅ Retry succeeded after {retry_wait}s wait")
                            break
                    else:
                        logger.warning("⚠️ All retries failed, continuing with partial data")

                # MEMORY UPDATE: Persist known elements
                # Merge current viewport elements with previously seen off-screen elements
                current_url_val = page_content.get('url', '')
                if current_url_val:
                    self._update_page_knowledge(current_url_val, page_content.get('elements', []))
                    page_content['elements'] = self._merge_known_elements(current_url_val, page_content.get('elements', []))
                    
                    # UPDATE CACHE for ACTIONS (CRITICAL for index-based clicks)
                    self.executor.set_cached_elements(page_content['elements'])
                    
                    # BUILD UNIFIED PAGE TREE (combines a11y hierarchy + elements + selectors)
                    try:
                        page_content['unified_page_tree'] = await self.dom.build_unified_page_tree(
                            active_page, 
                            page_content,
                            mode='text'
                        )
                    except Exception as tree_err:
                        logger.warning(f"Unified tree build failed: {tree_err}")
                        page_content['unified_page_tree'] = ""
                
                # Get URL early for blank page detection
                current_url = page_content.get('url', '')
                is_blank_page = current_url in ['about:blank', ''] or not current_url.startswith('http')
                
                # Capture screenshot for Logic (Vision/Analysis) - Skip for blank pages
                if is_blank_page:
                    logger.info(f"📸 Skipping screenshot for blank page: {current_url}")
                    screenshot_bytes = None
                    screenshot_b64 = None
                else:
                    try:
                        # CRITICAL: Use active_page for screenshot, not self.browser.page
                        # They can be different when tabs switch!
                        logger.info(f"📸 Taking screenshot from active_page (URL: {current_url[:50]}...)")
                        
                        # Take screenshot directly from active_page to avoid page mismatch
                        ss_start = time.time()
                        # CRITICAL: Lock to prevent race with background stream
                        async with self.page_access_lock:
                            screenshot_bytes = await active_page.screenshot(type='jpeg', quality=70, timeout=15000)
                        ss_elapsed = time.time() - ss_start
                        
                        if screenshot_bytes:
                            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                            logger.info(f"📸 ✅ Screenshot SUCCESS in {ss_elapsed:.2f}s, size: {len(screenshot_bytes)} bytes")
                        else:
                            screenshot_b64 = None
                            logger.warning(f"📸 Screenshot returned None after {ss_elapsed:.2f}s")
                    except Exception as e:
                        logger.warning(f"📸 Screenshot failed (non-critical): {e}")
                        screenshot_bytes = None
                        screenshot_b64 = None
                
                logger.info(f"🌐 URL: {page_content.get('url')} | 📄 Title: {page_content.get('title')}")

                # NOTE: Modal handling is now done by the LLM - overlay info is sent in the prompt
                # so the LLM can intelligently decide whether to dismiss or interact with modals


                # AUTO-COMPLETION CHECK: Detect if subtask is already done via URL patterns
                if self._check_url_based_completion(current_subtask.description, current_url):
                    self.memory.mark_completed(current_subtask.id, f"Auto-detected via URL: {current_url[:60]}...")
                    logger.info(f"✅ Subtask '{current_subtask.description}' auto-completed via URL detection")
                    continue  # Move to next subtask

                # CONTEXT CHANGE DETECTION: If URL changed significantly, we made progress!
                url_changed = current_url != self.previous_url and self.previous_url != ""
                if url_changed:
                    logger.info(f"🔄 URL changed: {self.previous_url[:40]}... → {current_url[:40]}...")
                    self.stuck_count = 0  # Reset on URL change
                self.previous_url = current_url
                
                # LLM-BASED PROGRESS AND STUCK DETECTION (every 5 steps)
                is_stuck = False
                stuck_suggestion = ""
                if step >= 5:
                    progress_check = await self._check_progress_and_stuck(current_subtask.description, step)
                    
                    # If we have enough data, auto-complete the task
                    if progress_check.get("has_enough_data", False):
                        logger.info(f"✅ LLM determined we have enough data: {progress_check.get('reasoning', '')[:80]}")
                        self.stuck_count = 0  # Reset on success
                        self.memory.mark_completed(current_subtask.id, f"LLM: Collected sufficient data - {len(self.memory.extracted_items)} items")
                        continue  # Move to next subtask
                    
                    # If stuck, get the suggestion for the prompt
                    if progress_check.get("is_stuck", False):
                        is_stuck = True
                        stuck_suggestion = progress_check.get("suggestion", "")
                        self.stuck_count += 1
                        logger.warning(f"🔄 LLM detected stuck: {progress_check.get('reasoning', '')[:80]}")
                        
                        # Force fail after 2 LLM-detected stuck situations
                        if self.stuck_count >= 2:
                            logger.error(f"❌ Stuck 2+ times per LLM. Force-completing with available data.")
                            self.memory.mark_completed(current_subtask.id, f"Auto-completed due to stuck - {len(self.memory.extracted_items)} items collected")
                            self.stuck_count = 0
                            continue
                    else:
                        # Not stuck - reset counter
                        self.stuck_count = 0
                
                # Let the LLM decide via next_mode prediction
                # Only force vision if we are stuck and text actions failed
                use_vision = (
                    self.vision.available and 
                    screenshot_b64 and 
                    (
                        self.next_mode == "vision" or  # LLM explicitly requested vision
                        is_stuck  # Need visual understanding when stuck text-only
                    )
                )

                # Action Planning
                self.current_action_description = "Planning action..."
                
                # Log saved data state for debugging stateful execution
                if self.memory.extracted_items:
                    logger.info(f"📦 Saved data available: {self.memory.get_saved_summary()}")
                
                action_prompt_context = self.memory.to_prompt_context()
                
                # Add persistent memory context (cross-session data)
                current_site = ""
                try:
                    current_site = page_content.get('url', '').split('/')[2] if page_content.get('url') else ""
                except:
                    pass
                
                # Pass full task description for semantic retrieval
                persistent_context = self.persistent.to_prompt_context(
                    current_site=current_site,
                    task_description=self.task  # Full task for semantic matching
                )
                if persistent_context:
                    action_prompt_context += f"\n\n{persistent_context}"
                
                # INJECT STUCK WARNING (with LLM suggestion and BLOCKED actions)
                if is_stuck:
                    warning_msg = "\n\n" + "="*60 + "\n"
                    warning_msg += "🚨 CRITICAL: YOU ARE STUCK - TRY A DIFFERENT APPROACH 🚨\n"
                    warning_msg += "="*60 + "\n\n"
                    
                    # Include LLM's suggestion if available
                    if stuck_suggestion:
                        warning_msg += f"💡 SUGGESTED APPROACH: {stuck_suggestion}\n\n"
                    
                    warning_msg += "OPTIONS:\n"
                    warning_msg += "  1. If you already have enough data → call 'done' immediately\n"
                    warning_msg += "  2. If the current approach isn't working → try a completely different method\n"
                    warning_msg += "  3. Use run_js to directly interact with elements via JavaScript\n"
                    warning_msg += "  4. If this subtask is impossible → use 'skip_subtask'\n\n"
                    warning_msg += "Review your ACTION HISTORY to avoid repeating failed steps!\n"
                     
                    logger.warning(f"🚫 Injecting STUCK WARNING into prompt")
                    action_prompt_context += warning_msg
                
                action = None
                if use_vision:
                    logger.info("🎨 Using VISION for action planning")
                    # Strongly emphasize: Complete CURRENT subtask, don't worry about future ones
                    vision_task_context = f"""Main Task (for context only): {self.task}

⚡ YOUR CURRENT FOCUS - Subtask #{current_subtask.id}: {current_subtask.description}

🎯 IMPORTANT: You must ONLY focus on completing Subtask #{current_subtask.id}. 
   - If you have achieved the goal of THIS subtask (e.g., described an image, found info), use 'done' action IMMEDIATELY
   - Do NOT worry about other subtasks yet - they will be handled after you mark this one done
   - Set 'completed_subtasks': [{current_subtask.id}] when using 'done'

{action_prompt_context}"""
                    action = await self.vision.plan_action_with_vision(
                        vision_task_context, screenshot_b64, page_content, self.memory.history, step
                    )
                    
                    # VISION OBSERVATION CAPTURE: Extract findings from vision reasoning
                    # This fixes the issue where vision describes what it sees but doesn't call save_info
                    if action and action.reasoning and len(action.reasoning) > 50:
                        # Check if reasoning contains extractable data patterns
                        data_triggers = ['price is', 'costs', 'found', 'the product', 'i can see', 
                                        'shows', 'displays', 'contains', 'lists', 'priced at']
                        if any(trigger in action.reasoning.lower() for trigger in data_triggers):
                            logger.info("📷 Vision provided observation - auto-adding to memory")
                            self.memory.add_observation(f"vision_step_{step}", action.reasoning[:500])
                
                if not action:
                    if use_vision: logger.info("⚠️ Vision failed, falling back to TEXT")
                    logger.info("📝 Using TEXT LLM for action planning")
                    text_task_context = f"Main Task: {self.task}\nCurrent Subtask: {current_subtask.description}\n{action_prompt_context}"
                    action = await self.llm.plan_action(
                        text_task_context, page_content, self.memory.history, step, last_error=last_error
                    )
                    self.metrics["llm_calls"]["planning"] += 1
                    self.metrics["llm_calls"]["total"] += 1
                    
                    # Track Token Usage
                    if action.usage:
                        self.metrics["tokens"]["prompt"] += action.usage.get("prompt_tokens", 0)
                        self.metrics["tokens"]["completion"] += action.usage.get("completion_tokens", 0)
                        self.metrics["tokens"]["total"] += action.usage.get("total_tokens", 0)

                action_names = [a.name for a in action.actions]
                logger.info(f"💭 Action Sequence: {action_names} | 💡 {action.reasoning[:100]}...")
                
                self.current_action_description = f"{action.reasoning[:60]}..."
                
                logger.info(f"🔮 Next Mode Prediction: {action.next_mode}")
                self.next_mode = action.next_mode

                # Dynamic Replanning: Handle Full Plan Update
                if action.updated_plan:
                    logger.warning(f"🔄 DYNAMIC REPLANNING: Replacing pending tasks with: {action.updated_plan}")
                    if self.memory.get_active_subtask(): # Mark current as skipped/replaced if valid
                        self.memory.mark_completed(current_subtask.id, "Replaced by new plan")
                    
                    self.memory.update_plan(action.updated_plan)
                    continue

                # Execute Action Sequence
                # Intelligent Replanning: Capture State Before Action (including visual hash)
                try:
                    pre_state = {
                        'url': active_page.url,
                        'title': await active_page.title(),
                        'screenshot_hash': None
                    }
                    # Capture pre-action screenshot hash for visual comparison
                    try:
                        async with self.page_access_lock:
                            pre_screenshot = await active_page.screenshot(type='jpeg', quality=50, timeout=5000)
                        pre_state['screenshot_hash'] = hash(pre_screenshot)  # Simple hash for comparison
                    except Exception:
                        pass  # Screenshot failed, will rely on URL comparison
                except Exception as e:
                    # Browser/page might have been closed
                    logger.warning(f"⚠️ Could not capture pre-state (browser may be closed): {e}")
                    # Attempt recovery in next loop iteration instead of crashing
                    self.browser.page = None  # Force fresh page
                    continue

                # Cache elements and page text on executor for verification
                self.executor._cached_elements = page_content.get('elements', [])
                
                # Update DOM Metrics
                element_count = len(self.executor._cached_elements)
                self.metrics['dom']['total_elements'] += element_count
                self.metrics['dom']['snapshots'] += 1
                if self.metrics['dom']['snapshots'] > 0:
                    self.metrics['dom']['avg_elements'] = self.metrics['dom']['total_elements'] / self.metrics['dom']['snapshots']
                self.executor.set_cached_page_text(page_content.get('body_text', ''))
                
                # Track this action for blocklist (in case we get stuck later)
                action_name = action.actions[0].name if action.actions else "unknown"
                self._last_executed_action = f"{action_name}: {str(action)[:80]}"
                
                # Track start time for metrics
                action_start = time.time()
                self.metrics["actions"]["total"] += 1
                
                # CRITICAL: Use Lock to prevent stream loop from accessing page during action
                # This prevents the race condition that was causing browser crashes!
                async with self.page_access_lock:
                    result = await self.executor.execute(active_page, action)
                    
                    # Refresh page reference IMMEDIATELY after action while holding lock if possible?
                    # No, executing action releases control. We need to be careful.
                    # Actually async with lock will hold it during await. Perfect.
                    
                    # CRITICAL: Refresh page reference IMMEDIATELY after action
                    # Navigation/click actions may have changed the page context
                    # get_active_page() will find the valid page AND update self.browser.page
                    refreshed_page = self.browser.get_active_page()
                
                # Check for reference update outside lock (to allow streaming to potentially resume or check)
                if refreshed_page and refreshed_page != active_page:
                    logger.info(f"🔄 Page reference updated after action: {refreshed_page.url[:50] if refreshed_page.url else 'new page'}")
                    active_page = refreshed_page
                    # Wait for the new page to fully load before continuing
                    try:
                        await refreshed_page.wait_for_load_state('domcontentloaded', timeout=10000)
                        await asyncio.sleep(1.5)  # Extra wait for SPAs/dynamic content
                        logger.info(f"✅ Page loaded and ready")
                    except Exception as load_err:
                        logger.debug(f"Page load wait skipped: {load_err}")
                
                # Track action timing
                action_time = (time.time() - action_start) * 1000
                self.metrics["performance"]["total_latency_ms"] += action_time
                self.metrics["performance"]["actions_completed"] += 1
                
                # Wait for any background downloads trigger by clicks
                await self._wait_for_downloads()
                
                # Intelligent Replanning: Capture State After Action & Verify with VISUAL comparison
                if result.success and not result.timeout_occurred:
                    try:
                        # Small delay to let page update
                        await asyncio.sleep(0.3)
                        
                        # Get FRESH page reference in case context changed during navigation
                        fresh_page = self.browser.get_active_page() or active_page
                        
                        post_state = {
                            'url': fresh_page.url,
                            'title': await fresh_page.title(),
                            'screenshot_hash': None
                        }
                        # Capture post-action screenshot hash
                        try:
                            async with self.page_access_lock:
                                post_screenshot = await fresh_page.screenshot(type='jpeg', quality=50, timeout=5000)
                            post_state['screenshot_hash'] = hash(post_screenshot)
                        except Exception:
                            pass
                            
                    except Exception as e:
                        # "Execution context destroyed" or "closed" = navigation happened = SUCCESS!
                        if "destroyed" in str(e).lower() or "closed" in str(e).lower() or "navigation" in str(e).lower():
                            logger.info("✅ Navigation detected (context changed) - this is successful!")
                            post_state = {'url': 'navigated', 'title': 'navigated', 'screenshot_hash': 'changed'}
                        else:
                            logger.warning(f"⚠️ Post-state capture failed: {e}")
                            post_state = pre_state  # Fallback to pre-state
                    
                    # Determine if ANY change happened (URL OR visual)
                    url_changed = pre_state['url'] != post_state['url']
                    visual_changed = (pre_state['screenshot_hash'] != post_state['screenshot_hash']) if (pre_state['screenshot_hash'] and post_state['screenshot_hash']) else None
                    
                    # Detect "No Effect" for state-changing actions
                    action_types = [a.name for a in action.actions]
                    expect_change = any(t in ['click', 'navigate', 'type', 'press', 'go_back'] for t in action_types)
                    
                    if expect_change:
                        if url_changed:
                            logger.info(f"✅ URL changed: {pre_state['url'][:50]} → {post_state['url'][:50]}")
                            self._last_no_effect_action = None  # Clear any previous no-effect flag
                            self._no_effect_count = 0  # Reset counter
                            # CRITICAL: Update previous_url IMMEDIATELY so specific recovery uses the NEW url
                            # If we crash after this point but before next loop, we want to restore THIS page, not the old one
                            self.previous_url = post_state['url']
                        elif visual_changed:
                            logger.info(f"✅ Visual change detected (modal/overlay/content update)")
                            self._last_no_effect_action = None  # Clear - action had effect
                            self._no_effect_count = 0  # Reset counter
                        elif visual_changed is None:
                            logger.info(f"⚠️ Could not verify visual change (screenshot comparison unavailable)")
                        else:
                            # BOTH URL and visual unchanged = TRUE no effect
                            failed_action_desc = f"{action_types} on {[a.params for a in action.actions]}"
                            
                            # Track repeated failures
                            if self._last_no_effect_action == failed_action_desc:
                                self._no_effect_count += 1
                            else:
                                self._no_effect_count = 1
                                self._last_no_effect_action = failed_action_desc
                            
                            warning = f"\n⚠️ NO EFFECT (#{self._no_effect_count}): Action {action_types} had no visual or URL change. The element might be non-interactive."
                            logger.warning(warning)
                            result.message += warning
                            result.data['state_unchanged'] = True
                            
                            # FORCE STUCK MODE after 3 repeated failures - need completely different approach
                            if self._no_effect_count >= 3:
                                logger.error(f"❌ STUCK: Same action '{failed_action_desc[:60]}' failed {self._no_effect_count}x - forcing stuck recovery")
                                self.stuck_count = 5  # Force stuck mode
                                # Add guidance to memory for next action
                                self.memory.add_observation(f"🚫 STUCK ALERT: Clicking '{[a.params for a in action.actions]}' does NOT work! Must try: 1) Different element, 2) JavaScript, 3) Direct URL navigation, or 4) Skip this subtask.")

                # Check for background downloads (e.g. PDF links that don't navigate)
                if self.recent_downloads:
                    logger.info(f"✅ Download detected during action: {self.recent_downloads}")
                    self.stuck_count = 0 # Reset stuck count as progress was made
                    result.success = True
                    result.message = result.message.replace("⚠️ CRITICAL", "✅") # Clear warning if download happened
                    result.message += f" (Triggered {len(self.recent_downloads)} downloads)"
                    
                    # Add download info to result data
                    if not result.data: result.data = {}
                    result.data['downloaded_files'] = [str(p) for p in self.recent_downloads]
                    
                    # Auto-complete if subtask explicitly asked for download
                    if 'download' in current_subtask.description.lower():
                        self.memory.mark_completed(current_subtask.id, f"Downloaded {len(self.recent_downloads)} files: {self.recent_downloads}")

                # RECORD ACTION IN HISTORY for complete LLM context
                try:
                    action_type = action.actions[0].name if action.actions else "unknown"
                    action_target = str(action.actions[0].params)[:80] if action.actions else ""
                    error_msg = None
                    
                    # Capture error if action failed
                    if not result.success:
                        error_msg = result.message[:100] if result.message else "Action failed"
                    
                    self.memory.add_action(
                        step=step,
                        url=page_content.get('url', ''),
                        title=page_content.get('title', ''),
                        goal=current_subtask.description if current_subtask else "",
                        reasoning=action.reasoning[:200] if action.reasoning else "",
                        action_type=action_type,
                        target=action_target,
                        result="✅ SUCCESS" if result.success else "❌ FAILED",
                        error=error_msg,
                        stuck=is_stuck,
                        mode="vision" if use_vision else "text"
                    )
                except Exception as history_err:
                    logger.debug(f"Could not record action history: {history_err}")

                # AUTO-SAVE EXTRACTED DATA FROM run_js RESULTS
                # If the action was run_js and returned structured data, save it to memory automatically
                if result.success and result.data and result.data.get('auto_extracted'):
                    js_result = result.data.get('result')
                    if js_result:
                        # Handle array of products/items
                        if isinstance(js_result, list) and len(js_result) > 0:
                            # Check if items have product-like structure
                            sample = js_result[0] if js_result else {}
                            if isinstance(sample, dict) and any(k in str(sample).lower() for k in ['name', 'title', 'price', 'product']):
                                logger.info(f"📦 AUTO-SAVING {len(js_result)} extracted items from run_js")
                                # Save to memory with a sensible key
                                key = 'extracted_products' if 'product' in str(js_result).lower() else 'extracted_items'
                                self.memory.safe_add_extracted({key: js_result})
                                # Also add individual items for traceability
                                for item in js_result:  # No limit - save all
                                    self.memory.extracted_items.append({'run_js_data': item, 'url': page_content.get('url', '')})
                        # Handle dict with products/items inside
                        elif isinstance(js_result, dict):
                            if 'products' in js_result or 'items' in js_result:
                                items = js_result.get('products') or js_result.get('items', [])
                                if isinstance(items, list) and len(items) > 0:
                                    logger.info(f"📦 AUTO-SAVING {len(items)} extracted items from run_js dict")
                                    self.memory.safe_add_extracted({'extracted_products': items})
                            elif any(k in str(js_result).lower() for k in ['name', 'title', 'price', 'product']):
                                # Single product/item
                                logger.info(f"📦 AUTO-SAVING single extracted item from run_js")
                                self.memory.safe_add_extracted({'extracted_item': js_result})


                # Dynamic Replanning: Handle Skip
                if result.action == "skip_subtask":
                    reason = result.data.get('reason', 'Skipped by agent')
                    logger.warning(f"⏭️ Skipping subtask {current_subtask.id}: {reason}")
                    self.memory.mark_failed(current_subtask.id, f"SKIPPED: {reason}")
                    continue
                
                # Handle Adaptive Timeout
                if not result.success and result.timeout_occurred:
                    retry_count = 0
                    while retry_count < 2 and result.timeout_occurred:
                        logger.warning(f"🕒 Timeout detected (attempt {retry_count+1}). Asking LLM decision...")
                        
                        decision = await self.llm.should_extend_timeout(
                            self.task, 
                            step, 
                            result.action, 
                            result.timeout_context, 
                            retry_count
                        )
                        
                        logger.info(f"🤔 Timeout Decision: {decision.get('decision')} ({decision.get('reasoning')})")
                        
                        if decision.get('decision') == 'EXTEND':
                            multiplier = decision.get('multiplier', 1.5)
                            # Re-execute with extended timeout
                            # NOTE: We need to pass this multiplier to executor, but for now we'll just try again
                            # ideally executor should accept custom timeout
                            logger.info(f"🔄 Retrying {result.action} with extended wait...")
                            async with self.page_access_lock:
                                result = await self.executor.execute(active_page, action)
                            retry_count += 1
                        elif decision.get('decision') == 'SKIP':
                            logger.warning("⏭️ LLM decided to SKIP failed action.")
                            result.success = True # Treat as success to continue
                            result.message += " (Skipped after timeout)"
                            break
                        else:
                            logger.error("❌ LLM decided to FAIL task due to timeout.")
                            break

                # Update State
                self.memory.history.append({
                    'step': step,
                    'subtask_id': current_subtask.id,
                    'action': action.model_dump(),
                    'result': result.model_dump(),
                    'url': page_content.get('url'),
                    'observation': page_content.get('observation_summary', ''),  # What was seen on page
                    'overlays': page_content.get('overlays', {}).get('hasOverlay', False),  # Were there popups?
                    'timestamp': time.time()
                })

                # Persist session state (cookies/storage) to survive crashes
                try:
                    if not self._recovering:
                        await self.browser.save_session()
                except Exception:
                    pass

                if result.success:
                    self.metrics["actions"]["successful"] += 1
                    logger.info(f"✅ Sequence Succeeded: {result.message}")
                    
                    if action.completed_subtasks:
                        for tid in action.completed_subtasks:
                            self.memory.mark_completed(tid, f"Completed via sequence '{action_names}'")

                    # Reset error state on success
                    last_error = None

                    has_done = any(a.name == "done" for a in action.actions)
                    has_extract = any(a.name == "extract" for a in action.actions)
                    has_save = any(a.name == "save_info" for a in action.actions)

                    # IMPROVED DATA CAPTURE: Capture data from ANY action that returns data
                    # This fixes the issue where only save_info data was captured
                    if result.data:
                        # Enrich data with context
                        result.data['url'] = page_content.get('url', '')
                        result.data['step'] = step
                        result.data['action_type'] = result.action
                        
                        if action and action.reasoning:
                            result.data['llm_reasoning'] = action.reasoning
                        
                        # For save_info actions, always accumulate
                        if has_save:
                            # Handle multiple save_info actions from a single sequence
                            if result.data.get('all_saved_items'):
                                for item in result.data['all_saved_items']:
                                    item_data = {
                                        'structured_info': item,
                                        'url': result.data.get('url', page_content.get('url', '')),
                                        'step': step,
                                        'action_type': result.action,
                                        'llm_reasoning': action.reasoning if action else ''
                                    }
                                    self.memory.extracted_items.append(item_data)
                                    verified = item.get('verified', False)
                                    key_name = item.get('key', 'unknown')
                                    status = "✅ VERIFIED" if verified else "⚠️ UNVERIFIED"
                                    logger.info(f"💾 Data saved: {key_name} [{status}]")
                            else:
                                # Single save_info (backward compatibility)
                                self.memory.extracted_items.append(result.data)
                                verified = result.data.get('structured_info', {}).get('verified', False)
                                key_name = result.data.get('structured_info', {}).get('key', 'unknown')
                                status = "✅ VERIFIED" if verified else "⚠️ UNVERIFIED"
                                logger.info(f"💾 Data saved: {key_name} [{status}]")
                            
                            self.memory.extracted_data.update(result.data)
                        elif has_extract and result.data.get('text_content'):
                            # Extract action - save the content
                            self.memory.extracted_data.update(result.data)
                            self.memory.extracted_items.append(result.data)
                            logger.info(f"💾 Extracted page content ({len(result.data.get('text_content', ''))} chars)")
                    
                    # THEN: Handle task completion
                    if has_done:
                        # CRITICAL FIX: Check if we need fallback data capture before marking done
                        task_needs_data = any(kw in self.task.lower() for kw in ['extract', 'find', 'get', 'save', 'what is', 'tell me', 'price', 'name'])
                        
                        if task_needs_data and not self.memory.extracted_items:
                            logger.warning("⚠️ Task requires data but none saved - triggering fallback capture")
                            fallback_data = await self._capture_fallback_data(active_page, current_subtask.description)
                            if fallback_data:
                                self.memory.extracted_items.append(fallback_data)
                                self.memory.extracted_data.update(fallback_data)
                                logger.info("📋 Fallback data captured before marking done")
                        
                        # Also capture any valuable reasoning from the done action
                        if action.reasoning and len(action.reasoning) > 50:
                            # Check if reasoning contains data that wasn't saved
                            if not self.memory.extracted_items or not any(
                                item.get('structured_info', {}).get('verified', False) 
                                for item in self.memory.extracted_items
                            ):
                                self.memory.add_observation("final_reasoning", action.reasoning[:500])
                        
                        self.memory.mark_completed(current_subtask.id, action.reasoning)
                        logger.info(f"✅ Subtask '{current_subtask.description}' marked complete (Done).")
                    elif has_extract or has_save:
                        # Mark complete if we have data
                        if result.data:
                            self.memory.mark_completed(current_subtask.id, "Data extracted")
                            logger.info(f"✅ Subtask '{current_subtask.description}' marked complete (Data extracted).")
                else:
                    self.metrics["actions"]["failed"] += 1
                    self.metrics["errors"]["action_errors"] += 1
                    self.metrics["errors"]["total"] += 1
                    logger.warning(f"⚠️ Sequence Failed at {result.action}: {result.message}")
                    
                    # Capture error for next prompt iteration
                    last_error = f"Action '{result.action}' failed: {result.message}"
                    
                    # Check if we should trigger intelligent replanning
                    should_replan = await self.planner.should_replan_after_failure(
                        self.memory, 
                        result.action, 
                        result.message
                    )
                    
                    if should_replan:
                        logger.info("📋 Triggering intelligent replanning...")
                        failure_context = f"Action '{result.action}' failed: {result.message}"
                        did_replan, new_subtasks = await self.planner.update_plan(
                            self.memory, 
                            failure_context
                        )
                        
                        if did_replan and new_subtasks:
                            logger.info(f"📋 Revised plan: {new_subtasks}")
                            self.memory.update_plan(new_subtasks)
                            # Reset stuck counter since we have a new approach
                            self.stuck_count = 0
                    
                    # NOTE: Stuck detection is now LLM-based via _check_progress_and_stuck earlier in the loop
                    # No legacy stuck check needed here
                
            self.is_running = False
            
            # Update resource metrics
            process = psutil.Process(os.getpid())
            self.metrics["resource"]["current_memory_mb"] = process.memory_info().rss / 1024 / 1024
            self.metrics["resource"]["peak_memory_mb"] = max(
                self.metrics["resource"]["peak_memory_mb"],
                self.metrics["resource"]["current_memory_mb"]
            )
            
            # Calculate averages
            if self.metrics["performance"]["actions_completed"] > 0:
                self.metrics["performance"]["avg_action_ms"] = (
                    self.metrics["performance"]["total_latency_ms"] / 
                    self.metrics["performance"]["actions_completed"]
                )
            
            # Log execution metrics
            final_result = self._build_final_result()
            self._log_execution_metrics(final_result.success)
            return final_result

        except Exception as e:
            self.is_running = False
            self.metrics["errors"]["total"] += 1
            self.metrics["errors"]["browser_errors"] += 1
            logger.error(f"❌ Critical Agent Failure: {e}", exc_info=True)
            self._log_execution_metrics(False)
            return BrowserResult(
                success=False, 
                task_summary=f"Critical failure: {str(e)}", 
                error=str(e),
                extracted_data={"merged": {}, "items": [], "stats": {}, "persistent_memory": {}}
            )
        finally:
            # Final wait for downloads before closing
            await self._wait_for_downloads()
            
            self.is_running = False
            if self.streaming_task:
                self.streaming_task.cancel()
                try:
                    await self.streaming_task
                except asyncio.CancelledError:
                    pass
            
            # Clear Canvas on exit
            if self.thread_id:
                try:
                    await self._push_state_update(None, 0)
                except Exception as e:
                    logger.warning(f"Failed to clear canvas: {e}")
            
            # Clear cached data (no longer needed after task)
            self.executor._cached_page_text = ""
            self.executor._cached_elements = []

            await self.browser.close()

    # NOTE: _needs_image_analysis and _is_stuck methods removed - they were dead code
    # Vision decision is now made in the run() loop based on next_mode
    # Stuck detection is now LLM-based via _check_progress_and_stuck()
    
    async def _check_progress_and_stuck(self, current_subtask_desc: str, step: int) -> Dict[str, Any]:
        """
        LLM-based progress and stuck detection.
        
        Returns:
            {
                "has_enough_data": bool,  # Should we complete the task?
                "is_stuck": bool,         # Are we in a unproductive loop?
                "suggestion": str,        # What to do differently
                "reasoning": str          # Why this decision
            }
        """
        # Only run this check every 5 steps to save API calls
        if step % 5 != 0 and step > 5:
            return {"has_enough_data": False, "is_stuck": False, "suggestion": "", "reasoning": "Skipped check"}
        
        # Build history summary
        history_lines = []
        for i, h in enumerate(self.memory.history[-10:]):
            actions = h.get('action', {}).get('actions', [])
            result = h.get('result', {})
            action_names = [a['name'] for a in actions] if actions else ['?']
            success = "✅" if result.get('success', False) else "❌"
            msg = result.get('message', '')[:60]
            history_lines.append(f"  Step {h.get('step', '?')}: {action_names} {success} - {msg}")
        
        history_str = "\n".join(history_lines) if history_lines else "(no history)"
        
        # Build saved data summary
        saved_data = []
        for item in self.memory.extracted_items[:10]:
            key = item.get('key', 'unknown')
            value = str(item.get('value', ''))[:80]
            verified = "✓" if item.get('verified', False) else "?"
            saved_data.append(f"  [{verified}] {key}: {value}")
        saved_str = "\n".join(saved_data) if saved_data else "(no data saved yet)"
        
        prompt = f"""You are a browser automation progress analyzer. Review the current state and decide:

ORIGINAL TASK: {self.task}
CURRENT SUBTASK: {current_subtask_desc}
CURRENT STEP: {step}

SAVED DATA ({len(self.memory.extracted_items)} items):
{saved_str}

RECENT ACTION HISTORY:
{history_str}

═══════════════════════════════════════════════════════════════════════════════
ANALYZE THE SITUATION
═══════════════════════════════════════════════════════════════════════════════

1. **HAS ENOUGH DATA?**
   - Does the saved data sufficiently answer the user's original question?
   - For "find best X" tasks: Do we have good options to recommend?
   - For "describe/analyze" tasks: Do we have the key findings?
   - Still need more info? → has_enough_data: false

2. **IS STUCK?**
   - Repeating the same ineffective action? (click failing, same JS code) → is_stuck: true
   - Productive scrolling to find more items? → is_stuck: FALSE (this is useful!)
   - Trying to apply filters that don't exist? → is_stuck: true
   - Making progress (new URLs, new data saved)? → is_stuck: false

3. **SUGGESTION** (only if stuck):
   - What different approach should we try?
   - Should we skip this subtask and move on?

Respond with JSON ONLY:
{{
    "has_enough_data": true/false,
    "is_stuck": true/false,
    "suggestion": "what to do differently (empty if not stuck)",
    "reasoning": "brief explanation"
}}"""

        try:
            response, _ = await self.llm.call_llm_direct(prompt)  # No token limit
            if response:
                import json
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                    logger.info(f"🧠 Progress check: has_enough={data.get('has_enough_data')}, stuck={data.get('is_stuck')}")
                    if data.get('reasoning'):
                        logger.info(f"   Reasoning: {data.get('reasoning')[:100]}")
                    return data
        except Exception as e:
            logger.warning(f"Progress check failed: {e}")
        
        return {"has_enough_data": False, "is_stuck": False, "suggestion": "", "reasoning": "Check failed"}

    async def _capture_fallback_data(self, page, subtask_description: str) -> Optional[Dict]:
        """Auto-capture visible page data as fallback when no explicit save_info was called.
        
        This ensures we capture data even if the LLM forgot to call save_info before done.
        """
        try:
            url = page.url
            title = await page.title()
            
            # Extract visible text (first 2000 chars for context)
            visible_text = await page.evaluate("document.body.innerText.substring(0, 2000)")
            
            # Try to extract key data patterns
            import re
            extracted_patterns = {}
            
            # Look for prices
            prices = re.findall(r'[\$₹€£]\s*[\d,]+(?:\.\d{2})?|\d+[\d,]*\s*(?:LPA|lpa|USD|INR|Rs\.?)', visible_text)
            if prices:
                extracted_patterns['prices_found'] = list(set(prices[:5]))  # Unique, max 5
            
            # Look for product names (capitalized phrases)
            product_patterns = re.findall(r'(?:[A-Z][a-zA-Z0-9]+\s+)+(?:Pro|Max|Plus|Ultra|SE)?', visible_text[:1000])
            if product_patterns:
                extracted_patterns['potential_products'] = list(set(p.strip() for p in product_patterns[:5]))
            
            logger.info(f"📋 Fallback data capture from {url[:50]}: {len(visible_text)} chars, patterns: {list(extracted_patterns.keys())}")
            
            return {
                "fallback_capture": True,
                "structured_info": {
                    "key": "page_content",
                    "value": visible_text[:500],  # Summary
                    "source": url,
                    "title": title,
                    "verified": True,  # It's from the actual page
                    "extracted_patterns": extracted_patterns
                },
                "url": url,
                "subtask": subtask_description,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.warning(f"Fallback data capture failed: {e}")
            return None

    def _check_url_based_completion(self, subtask_description: str, url: str) -> bool:
        """Check if a subtask is already completed based on URL patterns.
        
        IMPORTANT: Be VERY SPECIFIC to avoid false positives.
        Only return True if URL CLEARLY shows the EXACT task is done.
        """
        desc_lower = subtask_description.lower()
        url_lower = url.lower()
        
        # Sort detection - ONLY match if URL has explicit sort parameter
        # AND subtask is specifically about sorting (not searching or clicking)
        if 'sort' in desc_lower and 'low' in desc_lower:
            # Must have EXPLICIT sort indicators, not just search params
            if 'price-asc' in url_lower or 's=price-asc' in url_lower:
                logger.info(f"✅ URL-based completion: Sort by Low to High detected")
                return True
            # Don't auto-complete sort just because search is done - need actual sort URL param
            return False
        
        if 'sort' in desc_lower and 'high' in desc_lower:
            if 'price-desc' in url_lower or 's=price-desc' in url_lower:
                logger.info(f"✅ URL-based completion: Sort by High to Low detected")
                return True
            return False
        
        # Search completion - detect when "navigate and search" task is done
        # Only if subtask mentions BOTH navigating AND searching
        if ('navigate' in desc_lower or 'go to' in desc_lower) and 'search' in desc_lower:
            # Check if we're on a search results page (has search query in URL)
            if 'k=' in url_lower or 'q=' in url_lower or 'query=' in url_lower or 'search=' in url_lower:
                # Verify we're not on about:blank or login page
                if url_lower.startswith('http') and 'signin' not in url_lower and 'login' not in url_lower:
                    logger.info(f"✅ URL-based completion: Search results page detected")
                    return True
        
        return False

    def _build_final_result(self) -> BrowserResult:
        """Build a clean, human-readable result for the orchestrator/user."""
        
        success_count = sum(1 for t in self.memory.plan if t.status == 'completed')
        total_tasks = len(self.memory.plan)
        
        # === SECTION 1: TASK SUMMARY (human readable) ===
        summary = f"✅ Task {'Completed' if success_count == total_tasks else 'Partially Completed'} ({success_count}/{total_tasks} subtasks)\n\n"
        
        # List subtasks with status
        for t in self.memory.plan:
            icon = "✓" if t.status == 'completed' else "✗" if t.status == 'failed' else "○"
            summary += f"  {icon} {t.description}\n"
        
        # === SECTION 2: EXTRACTED DATA (the actual findings) ===
        if self.memory.extracted_data:
            summary += "\n📋 **Extracted Information:**\n"
            for key, value in self.memory.extracted_data.items():
                # Truncate very long values for readability
                display_value = str(value)[:200] + "..." if len(str(value)) > 200 else value
                summary += f"  • {key}: {display_value}\n"
        
        # === SECTION 3: ACTION LOG (simple list) ===
        action_log = []
        for step in self.memory.history:
            step_num = step.get("step", "?")
            action_data = step.get("action", {})
            result_data = step.get("result", {})
            
            # Get action name(s)
            actions = action_data.get("actions", [])
            action_names = [a.get("name", "unknown") for a in actions] if actions else ["unknown"]
            action_str = " → ".join(action_names)
            
            # Get result
            success = result_data.get("success", False)
            status = "✓" if success else "✗"
            
            # Get brief description from reasoning (first 80 chars)
            reasoning = action_data.get("reasoning", "")[:80]
            if len(action_data.get("reasoning", "")) > 80:
                reasoning += "..."
            
            action_log.append(f"  {step_num}. [{status}] {action_str}: {reasoning}")
        
        # Verbose action log removed from summary to prevent flooding
        # Structured history is already in 'actions_taken' field
        pass
        
        # === BUILD RESULT (schema-compatible) ===
        # actions_taken must be List[Dict] per BrowserResult schema
        # Keep it minimal: just step number, action name, success status
        minimal_actions = []
        for step in self.memory.history:
            step_num = step.get("step", 0)
            action_data = step.get("action", {})
            result_data = step.get("result", {})
            actions = action_data.get("actions", [])
            action_names = [a.get("name", "unknown") for a in actions] if actions else ["unknown"]
            
            minimal_actions.append({
                "step": step_num,
                "actions": action_names,
                "success": result_data.get("success", False)
            })
        
        result = BrowserResult(
            success=(success_count == total_tasks),
            task_summary=summary,
            actions_taken=minimal_actions,  # List[Dict] as schema requires
            # Sanitize extracted data to prevent output flooding
            extracted_data={
                k: (v[:500] + "...(truncated)" if isinstance(v, str) and len(v)>500 else 
                    str(v)[:1000] + "...(truncated)" if len(str(v))>1000 else v)
                for k, v in self.memory.extracted_data.items()
            },  
            metrics={
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'steps': len(self.memory.history),
                'verified_items': sum(1 for i in self.memory.extracted_items if i.get('structured_info', {}).get('verified'))
            }
        )
        
        logger.info(f"📊 Final Result: {success_count}/{total_tasks} subtasks, {len(self.memory.extracted_data)} data items")
        
        # Debug logging (not in response)
        logger.info("🕵️ DEBUG: Extracted Item Details:")
        for idx, item in enumerate(self.memory.extracted_items):
            verified = item.get('structured_info', {}).get('verified', 'N/A')
            logger.info(f"  Item {idx+1}: {item.get('url')} [verified={verified}]")
            logger.info(f"    Reasoning: {item.get('llm_reasoning', 'N/A')[:100]}")
             
        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent metrics."""
        uptime_seconds = time.time() - self._metrics_start_time if hasattr(self, '_metrics_start_time') else 0
        
        total_actions = self.metrics["actions"]["total"]
        success_rate = (
            (self.metrics["actions"]["successful"] / total_actions * 100) 
            if total_actions > 0 else 0
        )
        
        return {
            "uptime_seconds": uptime_seconds,
            "actions": self.metrics["actions"].copy(),
            "success_rate": success_rate,
            "llm_calls": self.metrics["llm_calls"].copy(),
            "performance": self.metrics["performance"].copy(),
            "navigation": {
                "pages_visited": self.metrics["navigation"]["pages_visited"],
                "unique_urls_count": len(self.metrics["navigation"]["unique_urls"]),
                "successful_navigations": self.metrics["navigation"]["successful_navigations"],
                "failed_navigations": self.metrics["navigation"]["failed_navigations"]
            },
            "vision": self.metrics["vision"].copy(),
            "errors": self.metrics["errors"].copy(),
            "resource": self.metrics["resource"].copy()
        }

    def _log_execution_metrics(self, success: bool):
        """Log execution metrics with clean formatting."""
        status_emoji = "✅" if success else "❌"
        
        logger.info("")
        logger.info(f"{status_emoji} BROWSER AGENT EXECUTION METRICS")
        logger.info("")
        
        # Performance
        logger.info("Performance:")
        logger.info(f"  Total Actions: {self.metrics['performance']['actions_completed']}")
        logger.info(f"  Total Time: {self.metrics['performance']['total_latency_ms']:.0f} ms")
        logger.info(f"  Avg Action Time: {self.metrics['performance']['avg_action_ms']:.0f} ms")
        
        # Actions
        logger.info("")
        logger.info("Actions:")
        logger.info(f"  Total: {self.metrics['actions']['total']}")
        logger.info(f"  Successful: {self.metrics['actions']['successful']}")
        logger.info(f"  Failed: {self.metrics['actions']['failed']}")
        success_rate = (self.metrics['actions']['successful'] / self.metrics['actions']['total'] * 100) if self.metrics['actions']['total'] > 0 else 0
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        
        # Action breakdown
        logger.info("")
        logger.info("Action Types:")
        for action_type in ['click', 'type', 'navigate', 'scroll', 'wait', 'extract', 'other']:
            count = self.metrics['actions'].get(action_type, 0)
            if count > 0:
                logger.info(f"  {action_type.capitalize()}: {count}")
        
        # Navigation
        logger.info("")
        logger.info("Navigation:")
        logger.info(f"  Pages Visited: {self.metrics['navigation']['pages_visited']}")
        logger.info(f"  Unique URLs: {len(self.metrics['navigation']['unique_urls'])}")
        
        # LLM Calls & Tokens
        logger.info("")
        logger.info("LLM & Cost:")
        logger.info(f"  Calls: {self.metrics['llm_calls']['total']} (Plan: {self.metrics['llm_calls']['planning']}, Vision: {self.metrics['llm_calls']['vision']})")
        if self.metrics['tokens']['total'] > 0:
            logger.info(f"  Tokens: {self.metrics['tokens']['total']:,} (In: {self.metrics['tokens']['prompt']:,}, Out: {self.metrics['tokens']['completion']:,})")
        if self.metrics['llm_calls']['failures'] > 0:
            logger.info(f"  Failures: {self.metrics['llm_calls']['failures']}")
        
        # Vision
        if self.metrics['vision']['screenshots_taken'] > 0 or self.metrics['vision']['vision_analyses'] > 0:
            logger.info("")
            logger.info("Vision:")
            logger.info(f"  Screenshots: {self.metrics['vision']['screenshots_taken']}")
            logger.info(f"  Analyses: {self.metrics['vision']['vision_analyses']}")
            
        # DOM & Complexity
        if self.metrics['dom']['snapshots'] > 0:
            avg_dom = self.metrics['dom']['avg_elements']
            logger.info("")
            logger.info("Page Complexity:")
            logger.info(f"  Avg Elements: {avg_dom:.0f}")
        
        # Errors
        if self.metrics['errors']['total'] > 0:
            logger.info("")
            logger.info("Errors:")
            logger.info(f"  Total: {self.metrics['errors']['total']}")
            logger.info(f"  Action Errors: {self.metrics['errors']['action_errors']}")
            logger.info(f"  LLM Errors: {self.metrics['errors']['llm_errors']}")
            logger.info(f"  Browser Errors: {self.metrics['errors']['browser_errors']}")
        
        # Resources
        logger.info("")
        logger.info("Resources:")
        logger.info(f"  Current Memory: {self.metrics['resource']['current_memory_mb']:.1f} MB")
        logger.info(f"  Peak Memory: {self.metrics['resource']['peak_memory_mb']:.1f} MB")
        logger.info("")


