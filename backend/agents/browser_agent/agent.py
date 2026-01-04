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
from typing import Dict, Any, List, Optional
import httpx

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

logger = logging.getLogger(__name__)


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
        self.streaming_task = None
        self.stuck_count = 0  # Track consecutive stuck warnings
        self.previous_url = ""  # Track URL changes to detect progress
        self.recent_downloads = [] # Track downloads in the current step
        self.known_elements = {} # Memory of elements by URL: {url: {xpath: elem}}
        self._active_downloads = set() # Track active background downloads
        
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
        logger.info("📹 Starting background stream loop")
        while self.is_running:
            try:
                if self.browser.page:
                    # Capture screenshot to memory (fast, no file save)
                    try:
                        screenshot_bytes = await self.browser.page.screenshot(
                            timeout=2000, 
                            animations="disabled",
                            full_page=False
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
        """Push update to orchestrator. Pass screenshot_b64=None to clear canvas."""
        if not self.thread_id or not self.backend_url:
            return
            
        try:
            # Serialize plan (Legacy format for Frontend: 'subtask', 'status')
            plan_data = []
            if screenshot_b64:
                plan_data = [
                    {"subtask": t.description, "status": t.status} 
                    for t in self.memory.plan
                ]
            
            payload = {
                "thread_id": self.thread_id,
                "screenshot_data": screenshot_b64,
                "url": self.browser.page.url if self.browser.page and screenshot_b64 else "",
                "step": step,
                "task": self.task if screenshot_b64 else "Session Ended",
                "task_plan": plan_data, # Send empty if clearing
                "current_action": self.current_action_description if screenshot_b64 else "Session Ended"
            }

            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.backend_url}/api/canvas/update",
                    json=payload,
                    timeout=2.0
                )
        except Exception as e:
            logger.debug(f"Streaming failed: {e}")

    async def run(self) -> BrowserResult:
        """Execute the browser automation task"""
        self.start_time = time.time()
        self.is_running = True
        logger.info(f"🚀 Starting SOTA Agent [{self.task_id}]: {self.task}")
        
        # 1. Initialize Browser
        if not await self.browser.launch(headless=self.headless, on_download=self._handle_download):
            return BrowserResult(success=False, task_summary="Browser launch failed", error="Browser launch failed")
        
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
                logger.info(f"\n{'='*50}\n📍 Step {step}\n{'='*50}")
                
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
                active_page = self.browser.get_active_page() or self.browser.page
                
                # Brief wait to let any recent tab switch settle
                await asyncio.sleep(0.3)
                
                page_content = await self.dom.get_page_content(active_page)

                # MEMORY UPDATE: Persist known elements
                # Merge current viewport elements with previously seen off-screen elements
                current_url_val = page_content.get('url', '')
                if current_url_val:
                    self._update_page_knowledge(current_url_val, page_content.get('elements', []))
                    page_content['elements'] = self._merge_known_elements(current_url_val, page_content.get('elements', []))
                    
                    # UPDATE CACHE for ACTIONS (CRITICAL for index-based clicks)
                    self.executor.set_cached_elements(page_content['elements'])
                
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

                # AUTO-COMPLETION CHECK: Detect if subtask is already done via URL patterns
                if self._check_url_based_completion(current_subtask.description, current_url):
                    self.memory.mark_completed(current_subtask.id, f"Auto-detected via URL: {current_url[:60]}...")
                    logger.info(f"✅ Subtask '{current_subtask.description}' auto-completed via URL detection")
                    continue  # Move to next subtask

                # Decision: Text vs Vision based on LLM prediction
                # If stuck, force Vision AND inject warning
                is_stuck = self._is_stuck()
                
                # CONTEXT CHANGE DETECTION: If URL changed significantly, we made progress!
                url_changed = current_url != self.previous_url and self.previous_url != ""
                if url_changed:
                    logger.info(f"🔄 URL changed: {self.previous_url[:40]}... → {current_url[:40]}...")
                    # Reset stuck counter on meaningful progress
                    if is_stuck:
                        logger.info("✅ URL change detected - resetting stuck counter")
                    self.stuck_count = 0
                self.previous_url = current_url
                
                # FORCE STUCK RECOVERY: If stuck detected multiple times, force subtask fail
                # BUT: Don't count stuck on blank pages - those just need navigation first
                if is_stuck and not is_blank_page:
                    self.stuck_count += 1
                    logger.warning(f"🔄 Stuck count: {self.stuck_count}/3")
                    if self.stuck_count >= 3:
                        logger.error("❌ Stuck 3+ times. Force-failing current subtask to move forward.")
                        self.memory.mark_failed(current_subtask.id, "Force-failed after repeated stuck detection")
                        self.stuck_count = 0
                        continue  # Move to next subtask
                        
                use_vision = (
                    self.vision.available and 
                    screenshot_b64 and 
                    (self.next_mode == "vision" or is_stuck)
                )

                # Action Planning
                self.current_action_description = "Planning action..."
                action_prompt_context = self.memory.to_prompt_context()
                
                # INJECT STUCK WARNING
                if is_stuck:
                    # Check recent history for context
                    recent_actions = [h['action']['actions'][0]['name'] for h in self.memory.history[-3:] if h['action'].get('actions')]
                    
                    warning_msg = "\n\n" + "="*60 + "\n"
                    warning_msg += "🚨 CRITICAL SYSTEM ALERT: YOU ARE IN A LOOP 🚨\n"
                    warning_msg += "="*60 + "\n\n"
                    
                    # Include explicit failed action if available
                    if hasattr(self, '_last_no_effect_action') and self._last_no_effect_action:
                        warning_msg += f"YOUR ACTION: {self._last_no_effect_action}\n"
                        warning_msg += "RESULT: ❌ NO EFFECT - The page state did not change.\n\n"
                    
                    warning_msg += "🚫 IT IS FORBIDDEN TO REPEAT ACTIONS THAT HAD NO EFFECT.\n"
                    warning_msg += "There is NO POINT trying the same action again - it will fail again.\n"
                    warning_msg += "Clicking different elements at similar coordinates will also fail.\n\n"
                    
                    # AGGRESSIVE WARNING for stuck_count >= 2
                    if self.stuck_count >= 2:
                        warning_msg += "="*60 + "\n"
                        warning_msg += "🛑 MANDATORY: YOU MUST TAKE A DIFFERENT APPROACH\n"
                        warning_msg += "="*60 + "\n\n"
                        warning_msg += "STEP 1 - VERIFY THE ELEMENT:\n"
                        warning_msg += "Use 'run_js' to check if the element is actually interactive:\n"
                        warning_msg += "  Example: run_js with code: document.querySelector('a[href]').href\n"
                        warning_msg += "  This tells you what the link URL is, so you can 'navigate' to it directly.\n\n"
                        warning_msg += "STEP 2 - DECIDE BASED ON VERIFICATION:\n"
                        warning_msg += "  • If JS reveals a URL → Use 'navigate' to go there directly\n"
                        warning_msg += "  • If the element is an image/non-link → The content is already visible. Use 'save_info' to save what you see, then 'done'\n"
                        warning_msg += "  • If the goal is impossible → Use 'skip_subtask' to move on\n\n"
                        warning_msg += "⛔ DO NOT OUTPUT 'click'. IT IS FORBIDDEN AND WILL FAIL.\n"
                    elif 'go_back' in recent_actions:
                        warning_msg += "Your 'go_back' action is failing. Use 'navigate' to the previous URL instead.\n"
                    elif 'click' in recent_actions:
                        warning_msg += "Your 'click' is not causing navigation. STOP CLICKING.\n"
                        warning_msg += "Instead: Use 'run_js' to extract the href/onclick, then decide.\n"
                    else:
                        warning_msg += "DO NOT repeat the same action. Use a different approach.\n"
                         
                    logger.warning(f"🚫 Injecting STUCK WARNING into prompt: {warning_msg}")
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
                    self.metrics["llm_calls"]["vision"] += 1
                    self.metrics["llm_calls"]["total"] += 1
                    self.metrics["vision"]["vision_analyses"] += 1
                
                if not action:
                    if use_vision: logger.info("⚠️ Vision failed, falling back to TEXT")
                    logger.info("📝 Using TEXT LLM for action planning")
                    text_task_context = f"Main Task: {self.task}\nCurrent Subtask: {current_subtask.description}\n{action_prompt_context}"
                    action = await self.llm.plan_action(
                        text_task_context, page_content, self.memory.history, step
                    )
                    self.metrics["llm_calls"]["planning"] += 1
                    self.metrics["llm_calls"]["total"] += 1

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
                pre_state = {
                    'url': active_page.url,
                    'title': await active_page.title(),
                    'screenshot_hash': None
                }
                # Capture pre-action screenshot hash for visual comparison
                try:
                    pre_screenshot = await active_page.screenshot(type='jpeg', quality=50, timeout=5000)
                    pre_state['screenshot_hash'] = hash(pre_screenshot)  # Simple hash for comparison
                except Exception:
                    pass  # Screenshot failed, will rely on URL comparison

                # Cache elements on executor for index-based clicking
                self.executor._cached_elements = page_content.get('elements', [])
                
                # Track actions by type
                action_start = time.time()
                for act in action.actions:
                    action_type = act.name
                    if action_type in self.metrics["actions"]:
                        self.metrics["actions"][action_type] += 1
                    else:
                        self.metrics["actions"]["other"] += 1
                    self.metrics["actions"]["total"] += 1
                
                result = await self.executor.execute(active_page, action)
                
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
                        
                        post_state = {
                            'url': active_page.url,
                            'title': await active_page.title(),
                            'screenshot_hash': None
                        }
                        # Capture post-action screenshot hash
                        try:
                            post_screenshot = await active_page.screenshot(type='jpeg', quality=50, timeout=5000)
                            post_state['screenshot_hash'] = hash(post_screenshot)
                        except Exception:
                            pass
                            
                    except Exception as e:
                        # "Execution context destroyed" = navigation happened = SUCCESS!
                        if "destroyed" in str(e).lower() or "navigation" in str(e).lower():
                            logger.info("✅ Navigation detected (context destroyed) - this is successful!")
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
                        elif visual_changed:
                            logger.info(f"✅ Visual change detected (modal/overlay/content update)")
                            self._last_no_effect_action = None  # Clear - action had effect
                        elif visual_changed is None:
                            logger.info(f"⚠️ Could not verify visual change (screenshot comparison unavailable)")
                        else:
                            # BOTH URL and visual unchanged = TRUE no effect
                            failed_action_desc = f"{action_types} on {[a.params for a in action.actions]}"
                            self._last_no_effect_action = failed_action_desc
                            warning = f"\n⚠️ NO EFFECT: Action {action_types} had no visual or URL change. The element might be non-interactive."
                            logger.warning(warning)
                            result.message += warning
                            result.data['state_unchanged'] = True

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

                if result.success:
                    self.metrics["actions"]["successful"] += 1
                    logger.info(f"✅ Sequence Succeeded: {result.message}")
                    
                    if action.completed_subtasks:
                        for tid in action.completed_subtasks:
                            self.memory.mark_completed(tid, f"Completed via sequence '{action_names}'")

                    has_done = any(a.name == "done" for a in action.actions)
                    has_extract = any(a.name == "extract" for a in action.actions)
                    has_save = any(a.name == "save_info" for a in action.actions)

                    # FIRST: Always capture save_info data (even if done is also present)
                    if has_save and result.data:
                        # Enrich data with LLM reasoning
                        if action and action.reasoning:
                            result.data['llm_reasoning'] = action.reasoning
                        
                        self.memory.extracted_data.update(result.data)
                        self.memory.extracted_items.append(result.data)  # Accumulate data
                        logger.info(f"💾 Data saved: {result.data.get('structured_info', {}).get('key', 'unknown')}")
                    
                    # THEN: Handle task completion
                    if has_done:
                         self.memory.mark_completed(current_subtask.id, action.reasoning)
                         logger.info(f"✅ Subtask '{current_subtask.description}' marked complete (Done).")
                    elif has_extract or has_save:
                        # Only mark complete if we have data and no explicit done
                        if result.data:
                            self.memory.mark_completed(current_subtask.id, "Data extracted")
                            logger.info(f"✅ Subtask '{current_subtask.description}' marked complete (Data extracted).")
                else:
                    self.metrics["actions"]["failed"] += 1
                    self.metrics["errors"]["action_errors"] += 1
                    self.metrics["errors"]["total"] += 1
                    logger.warning(f"⚠️ Sequence Failed at {result.action}: {result.message}")
                    
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
                    
                    if self._is_stuck():
                        logger.warning("🔄 Stuck detected. Marking subtask failed.")
                        self.memory.mark_failed(current_subtask.id, "Stuck executing actions")
                        self.next_mode = "vision"
                
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
            return BrowserResult(success=False, task_summary=f"Critical failure: {str(e)}", error=str(e))
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

            await self.browser.close()

    def _needs_image_analysis(self, subtask_desc: str) -> bool:
        return False # Disabled per user request to rely on LLM logic only

    def _is_stuck(self) -> bool:
        """Check if stuck based on repetitive identical actions"""
        if len(self.memory.history) < 3: return False
        
        # Check last 3 actions
        recent = self.memory.history[-3:]
        
        # Extract action signatures (name + critical params)
        sigs = []
        for h in recent:
            actions = h['action'].get('actions', [])
            if not actions: continue
            
            # Create a signature for the FIRST action in the step
            a = actions[0]
            name = a['name']
            
            # For clicks/types, include the target
            params = str(a.get('keywords', '')) + str(a.get('xpath', '')) + str(a.get('text', '')) + str(a.get('selector', ''))
            sig = f"{name}:{params}"
            sigs.append(sig)
            
        # If we have 3 identical actions (and they aren't 'wait' or 'scroll')
        if len(sigs) == 3 and len(set(sigs)) == 1:
            action_name = sigs[0].split(':')[0]
            if action_name not in ['scroll', 'wait']:
                logger.warning(f"🔄 Stuck detection: Repeated action {action_name} 3 times")
                return True
                
        # Improved alternating loop detection:
        # ONLY flag if there's a TRUE alternating pattern like A-B-A-B-A-B
        # AND scroll is NOT one of the actions (scrolling to find things is valid!)
        if len(self.memory.history) >= 6:
            recent_6 = self.memory.history[-6:]
            sigs_6 = []
            for h in recent_6:
                actions = h['action'].get('actions', [])
                if actions: sigs_6.append(actions[0]['name'])
            
            unique_actions = set(sigs_6)
            
            # NEVER flag scroll-heavy patterns as stuck - scrolling to find elements is valid
            if 'scroll' in unique_actions:
                return False
            
            # Only flag TRUE alternating: A-B-A-B-A-B (3 occurrences of each)
            # Not just "2 unique actions in 6 steps"
            if len(unique_actions) == 2 and 'done' not in sigs_6 and 'navigate' not in sigs_6:
                # Check if it's a TRUE alternating pattern (A-B-A-B-A-B)
                is_true_alternating = all(sigs_6[i] != sigs_6[i+1] for i in range(5))
                if is_true_alternating:
                    logger.warning(f"🔄 Stuck detection: TRUE alternating loop detected {unique_actions}")
                    return True

        return False

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
        success_count = sum(1 for t in self.memory.plan if t.status == 'completed')
        total_tasks = len(self.memory.plan)
        summary_lines = [f"{t.id}. {t.description}: {t.status}" for t in self.memory.plan]
        summary = f"Completed {success_count}/{total_tasks} subtasks.\n" + "\n".join(summary_lines)
        if self.memory.observations:
            summary += "\n\nObservations:\n" + "\n".join([f"- {k}: {v}" for k,v in self.memory.observations.items()])

        if self.memory.extracted_items:
            summary += "\n\nExtracted Data Highlights:\n"
            for item in self.memory.extracted_items:
                if 'structured_info' in item:
                   s = item['structured_info']
                   summary += f"- {s['key']}: {s['value']} (Source: {s.get('source', 'unknown')})\n"
                else:
                   reasoning = item.get('llm_reasoning', 'No reasoning captured')
                   url = item.get('url', 'Unknown URL')
                   summary += f"- Source: {url}\n  Finding: {reasoning}\n"

        result = BrowserResult(
            success=(success_count == total_tasks),
            task_summary=summary,
            actions_taken=self.memory.history,
            extracted_data={"merged": self.memory.extracted_data, "items": self.memory.extracted_items},
            metrics={'total_time': time.time() - self.start_time if self.start_time else 0}
        )
        logger.info(f"📊 Final Result Building: Collected {len(self.memory.extracted_items)} items from URLs: {[d.get('url') for d in self.memory.extracted_items]}")
        
        # Verbose Debug for User Verification
        logger.info("🕵️ DEBUG: Extracted Item Details:")
        for idx, item in enumerate(self.memory.extracted_items):
             logger.info(f"  Item {idx+1}: {item.get('url')}")
             logger.info(f"    Reasoning: {item.get('llm_reasoning', 'N/A')}")
             
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
        
        # LLM Calls
        logger.info("")
        logger.info("LLM Calls:")
        logger.info(f"  Total: {self.metrics['llm_calls']['total']}")
        logger.info(f"  Planning: {self.metrics['llm_calls']['planning']}")
        logger.info(f"  Vision: {self.metrics['llm_calls']['vision']}")
        if self.metrics['llm_calls']['failures'] > 0:
            logger.info(f"  Failures: {self.metrics['llm_calls']['failures']}")
        
        # Vision
        if self.metrics['vision']['screenshots_taken'] > 0 or self.metrics['vision']['vision_analyses'] > 0:
            logger.info("")
            logger.info("Vision:")
            logger.info(f"  Screenshots: {self.metrics['vision']['screenshots_taken']}")
            logger.info(f"  Analyses: {self.metrics['vision']['vision_analyses']}")
        
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

