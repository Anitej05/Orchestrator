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
    from agents.agent_file_manager import AgentFileManager, FileType, FileStatus
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
    
    def __init__(self, task: str, max_steps: int = 20, headless: bool = False, thread_id: Optional[str] = None, backend_url: Optional[str] = "http://localhost:8000"):
        self.task = task
        self.max_steps = max_steps
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
            for step in range(1, self.max_steps + 1):
                logger.info(f"\n{'='*50}\n📍 Step {step}/{self.max_steps}\n{'='*50}")
                
                # Get Active Subtask
                current_subtask = self.memory.get_active_subtask()
                if not current_subtask:
                    logger.info("🎉 No more subtasks! Task Complete.")
                    break
                logger.info(f"🎯 Current Goal: {current_subtask.description}")

                # Context Gathering
                self.current_action_description = "Observing page..."
                page_content = await self.dom.get_page_content(self.browser.page)
                
                # Capture screenshot for Logic (Vision/Analysis) - In Memory Only
                screenshot_bytes = await self.browser.page.screenshot()
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode() if screenshot_bytes else None
                
                logger.info(f"🌐 URL: {page_content.get('url')} | 📄 Title: {page_content.get('title')}")

                # Decision: Text vs Vision based on LLM prediction
                use_vision = (
                    self.vision.available and 
                    screenshot_b64 and 
                    (self.next_mode == "vision" or self._is_stuck())
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
                
                action = None
                if use_vision:
                    logger.info("🎨 Using VISION for action planning")
                    vision_task_context = f"Main Task: {self.task}\nCurrent Subtask: {current_subtask.description}\n{action_prompt_context}"
                    action = await self.vision.plan_action_with_vision(
                        vision_task_context, screenshot_b64, page_content, self.memory.history, step, self.max_steps
                    )
                
                if not action:
                    if use_vision: logger.info("⚠️ Vision failed, falling back to TEXT")
                    logger.info("📝 Using TEXT LLM for action planning")
                    text_task_context = f"Main Task: {self.task}\nCurrent Subtask: {current_subtask.description}\n{action_prompt_context}"
                    action = await self.llm.plan_action(
                        text_task_context, page_content, self.memory.history, step, self.max_steps
                    )

                action_names = [a.name for a in action.actions]
                logger.info(f"💭 Action Sequence: {action_names} | 💡 {action.reasoning[:100]}...")
                
                self.current_action_description = f"{action.reasoning[:60]}..."
                
                logger.info(f"🔮 Next Mode Prediction: {action.next_mode}")
                self.next_mode = action.next_mode

                # Execute Action Sequence
                result = await self.executor.execute(self.browser.page, action)

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

                    if has_done:
                         self.memory.mark_completed(current_subtask.id, action.reasoning)
                         logger.info(f"✅ Subtask '{current_subtask.description}' marked complete (Done).")
                    elif has_extract:
                        if result.data:
                            self.memory.extracted_data.update(result.data)
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
        keywords = ['describe', 'analyze', 'what is', 'doodle', 'image analysis', 'check image']
        return any(kw in subtask_desc.lower() for kw in keywords)

    def _is_stuck(self) -> bool:
        """Check if stuck based on simplified action sequence hash"""
        if len(self.memory.history) < 4: return False
        recent_hashes = []
        for h in self.memory.history[-4:]:
            actions = h['action'].get('actions', [])
            action_sig = "-".join([f"{a['name']}" for a in actions])
            recent_hashes.append(action_sig)
        return len(set(recent_hashes)) == 1 and "done" not in recent_hashes[0]

    def _build_final_result(self) -> BrowserResult:
        success_count = sum(1 for t in self.memory.plan if t.status == 'completed')
        total_tasks = len(self.memory.plan)
        summary_lines = [f"{t.id}. {t.description}: {t.status}" for t in self.memory.plan]
        summary = f"Completed {success_count}/{total_tasks} subtasks.\n" + "\n".join(summary_lines)
        if self.memory.observations:
            summary += "\n\nObservations:\n" + "\n".join([f"- {k}: {v}" for k,v in self.memory.observations.items()])

        return BrowserResult(
            success=(success_count == total_tasks),
            task_summary=summary,
            actions_taken=self.memory.history,
            extracted_data=self.memory.extracted_data,
            metrics={'total_time': time.time() - self.start_time if self.start_time else 0}
        )
