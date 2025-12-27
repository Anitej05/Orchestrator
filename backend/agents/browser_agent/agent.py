"""
Browser Agent - Main Agent Orchestrator

Stateful, SOTA browser automation with memory, planning, and vision.
"""

import time
import uuid
import base64
import logging
import asyncio
from typing import Dict, Any, List, Optional
import httpx

from .browser import Browser
from .dom import DOMExtractor
from .actions import ActionExecutor
from .llm import LLMClient
from .vision import VisionClient

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
        
        # Initialize File Managers
        self.download_manager = None
        self.screenshot_manager = None
        if AgentFileManager:
            try:
                self.download_manager = AgentFileManager(
                    agent_id="browser_agent_downloads",
                    storage_dir="storage/browser_downloads",
                    default_ttl_hours=72,
                    auto_cleanup=True
                )
                self.screenshot_manager = AgentFileManager(
                    agent_id="browser_agent_screenshots",
                    storage_dir="storage/browser_screenshots",
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

    async def _handle_download(self, download):
        """Handle file download event"""
        try:
            filename = download.suggested_filename
            logger.info(f"📥 Starting download: {filename}")
            
            storage_dir = Path(self.download_manager.storage_dir) if self.download_manager else Path("storage/browser_downloads")
            storage_dir.mkdir(parents=True, exist_ok=True)
            target_path = storage_dir / filename
            
            await download.save_as(str(target_path))
            
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
                import asyncio
                await asyncio.sleep(0.3)
                
                page_content = await self.dom.get_page_content(active_page)
                
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
                        import time as time_module
                        ss_start = time_module.time()
                        screenshot_bytes = await active_page.screenshot(type='jpeg', quality=70, timeout=15000)
                        ss_elapsed = time_module.time() - ss_start
                        
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

                # Special Case: Image Analysis explicitly requested
                if self._needs_image_analysis(current_subtask.description) and screenshot_b64:
                    self.current_action_description = "Analyzing image..."
                    logger.info("🖼️ Analysis subtask detected. Analyzing image...")
                    analysis = await self.vision.analyze_image(
                        screenshot_b64, self.task, page_content.get('url', '')
                    )
                    if analysis:
                        self.memory.add_observation(f"Image Analysis ({current_subtask.description})", analysis)
                        logger.info(f"✅ Image analyzed. Result stored in memory.")
                        self.memory.mark_completed(current_subtask.id, "Image analyzed")
                        if self.next_mode != "vision":
                            self.next_mode = "text"
                        continue 

                # Action Planning
                self.current_action_description = "Planning action..."
                action_prompt_context = self.memory.to_prompt_context()
                
                # INJECT STUCK WARNING
                if is_stuck:
                    # Check recent history for context
                    recent_actions = [h['action']['actions'][0]['name'] for h in self.memory.history[-3:] if h['action'].get('actions')]
                    
                    warning_msg = "\n\n⚠️ SYSTEM WARNING: YOU ARE STUCK! ⚠️\nYou have repeated the same failed action multiple times.\n"
                    
                    if 'go_back' in recent_actions:
                         warning_msg += "Your 'go_back' action seems to be failing or looping. STOP using 'go_back'.\nInstead, use the 'navigate' action to explicitly go to the previous URL (e.g. the search results page)."
                    elif 'click' in recent_actions:
                         warning_msg += "Your 'click' action is not working or you are clicking the wrong thing repeatedly.\nTRY A DIFFERENT STRATEGY: Scroll to find a different element, use KeyDown/KeyUp to navigate, or try 'navigate'."
                    else:
                         warning_msg += "DO NOT try the exact same action again. Use a different strategy."
                         
                    logger.warning(f"🚫 Injecting STUCK WARNING into prompt: {warning_msg}")
                    action_prompt_context += warning_msg
                
                action = None
                if use_vision:
                    logger.info("🎨 Using VISION for action planning")
                    vision_task_context = f"Main Task: {self.task}\nCurrent Subtask: {current_subtask.description}\n{action_prompt_context}"
                    action = await self.vision.plan_action_with_vision(
                        vision_task_context, screenshot_b64, page_content, self.memory.history, step
                    )
                
                if not action:
                    if use_vision: logger.info("⚠️ Vision failed, falling back to TEXT")
                    logger.info("📝 Using TEXT LLM for action planning")
                    text_task_context = f"Main Task: {self.task}\nCurrent Subtask: {current_subtask.description}\n{action_prompt_context}"
                    action = await self.llm.plan_action(
                        text_task_context, page_content, self.memory.history, step
                    )

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
                result = await self.executor.execute(active_page, action)
                
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
                    'timestamp': time.time()
                })

                if result.success:
                    logger.info(f"✅ Sequence Succeeded: {result.message}")
                    
                    if action.completed_subtasks:
                        for tid in action.completed_subtasks:
                            self.memory.mark_completed(tid, f"Completed via sequence '{action_names}'")

                    has_done = any(a.name == "done" for a in action.actions)
                    has_extract = any(a.name == "extract" for a in action.actions)
                    has_save = any(a.name == "save_info" for a in action.actions)

                    if has_done:
                         self.memory.mark_completed(current_subtask.id, action.reasoning)
                         logger.info(f"✅ Subtask '{current_subtask.description}' marked complete (Done).")
                    elif has_extract or has_save:
                        if result.data:
                            # Enrich data with LLM reasoning (often contains the observed price)
                            if action and action.reasoning:
                                result.data['llm_reasoning'] = action.reasoning
                                
                            self.memory.extracted_data.update(result.data)
                            self.memory.extracted_items.append(result.data)  # Accumulate data
                            self.memory.mark_completed(current_subtask.id, "Data extracted")
                            logger.info(f"✅ Subtask '{current_subtask.description}' marked complete (Data extracted).")
                else:
                    logger.warning(f"⚠️ Sequence Failed at {result.action}: {result.message}")
                    if self._is_stuck():
                        logger.warning("🔄 Stuck detected. Marking subtask failed.")
                        self.memory.mark_failed(current_subtask.id, "Stuck executing actions")
                        self.next_mode = "vision"
                
            self.is_running = False
            return self._build_final_result()

        except Exception as e:
            self.is_running = False
            logger.error(f"❌ Critical Agent Failure: {e}", exc_info=True)
            return BrowserResult(success=False, task_summary=f"Critical failure: {str(e)}", error=str(e))
        finally:
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
                
        # Fallback: check for alternating loop (A-B-A-B) in last 6 steps
        if len(self.memory.history) >= 6:
            recent_6 = self.memory.history[-6:]
            sigs_6 = []
            for h in recent_6:
                actions = h['action'].get('actions', [])
                if actions: sigs_6.append(actions[0]['name'])
            
            # If only 2 unique action types in last 6 steps (e.g. scroll, click, scroll, click...)
            # AND none are 'done'
            if len(set(sigs_6)) <= 2 and 'done' not in sigs_6:
                 logger.warning(f"🔄 Stuck detection: Alternating loop detected {set(sigs_6)}")
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
