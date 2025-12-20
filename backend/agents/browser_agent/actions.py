"""
Browser Agent - Action Execution

Execute browser actions reliably. THE MOST CRITICAL COMPONENT.
"""

import logging
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from playwright.async_api import Page

from .schemas import ActionPlan, ActionResult, AtomicAction
from .dom import DOMExtractor

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Execute browser actions reliably"""
    
    def __init__(self, screenshot_manager=None, thread_id: str = None):
        self.dom = DOMExtractor()
        self.screenshot_manager = screenshot_manager
        self.thread_id = thread_id
    
    async def execute(self, page: Page, plan: ActionPlan) -> ActionResult:
        """Execute a sequence of actions and return result"""
        results_log = []
        final_data = {}
        
        logger.info(f"⚡ Executing Sequence: {[a.name for a in plan.actions]} | {plan.reasoning[:50]}...")
        
        for action in plan.actions:
            try:
                result = await self._execute_single(page, action)
                results_log.append(f"{action.name}: {result.message}")
                
                if result.data:
                    final_data.update(result.data)
                
                if not result.success:
                    logger.warning(f"⚠️ Action '{action.name}' failed: {result.message}. Stopping sequence.")
                    return ActionResult(
                        success=False, 
                        action=action.name, 
                        message=f"Sequence stopped at {action.name}: {result.message}",
                        data=final_data
                    )
            except Exception as e:
                logger.error(f"Critical execution error on {action.name}: {e}")
                return ActionResult(success=False, action=action.name, message=str(e), data=final_data)

        return ActionResult(
            success=True, 
            action="sequence", 
            message="; ".join(results_log),
            data=final_data
        )

    async def _execute_single(self, page: Page, action: AtomicAction) -> ActionResult:
        if action.name == "navigate":
            return await self._navigate(page, action.params)
        elif action.name == "click":
            return await self._click(page, action.params)
        elif action.name == "type":
            return await self._type(page, action.params)
        elif action.name == "scroll":
            return await self._scroll(page, action.params)
        elif action.name == "hover":
            return await self._hover(page, action.params)
        elif action.name == "press":
            return await self._press(page, action.params)
        elif action.name == "wait":
            return await self._wait(page, action.params)
        elif action.name == "go_back":
            try:
                await page.go_back()
                return ActionResult(success=True, action="go_back", message="Went back")
            except Exception as e:
                return ActionResult(success=False, action="go_back", message=str(e))
        elif action.name == "go_forward":
            try:
                await page.go_forward()
                return ActionResult(success=True, action="go_forward", message="Went forward")
            except Exception as e:
                return ActionResult(success=False, action="go_forward", message=str(e))
        elif action.name == "extract":
            return await self._extract(page, action.params)
        elif action.name == "done":
            return ActionResult(success=True, action="done", message="Task complete")
        elif action.name == "screenshot":
            return await self._screenshot(page, action.params)
        else:
            return ActionResult(success=False, action=action.name, message=f"Unknown action: {action.name}")

    async def _hover(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Hover over element"""
        selector = params.get('selector', '')
        try:
            if selector:
                element = await self.dom.find_element(page, selector)
                if element:
                    await element.hover()
                    await page.wait_for_timeout(500)
                    return ActionResult(success=True, action="hover", message=f"Hovered over {selector}")
            return ActionResult(success=False, action="hover", message="Element not found")
        except Exception as e:
            return ActionResult(success=False, action="hover", message=str(e))

    async def _press(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Press keyboard key"""
        key = params.get('key', '')
        try:
            if key:
                await page.keyboard.press(key)
                await page.wait_for_timeout(300)
                return ActionResult(success=True, action="press", message=f"Pressed {key}")
            return ActionResult(success=False, action="press", message="No key provided")
        except Exception as e:
            return ActionResult(success=False, action="press", message=str(e))

    async def _wait(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Wait for seconds"""
        seconds = params.get('seconds', 1)
        try:
            await page.wait_for_timeout(seconds * 1000)
            return ActionResult(success=True, action="wait", message=f"Waited {seconds}s")
        except Exception as e:
            return ActionResult(success=False, action="wait", message=str(e))

    async def _navigate(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Navigate to URL"""
        url = params.get('url', '')
        if not url:
            return ActionResult(success=False, action="navigate", message="No URL provided")
        
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await page.wait_for_timeout(1000)
            
            logger.info(f"✅ Navigated to: {url}")
            return ActionResult(success=True, action="navigate", message=f"Navigated to {url}")
        except Exception as e:
            return ActionResult(success=False, action="navigate", message=str(e))
    
    async def _click(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Click on element"""
        selector = params.get('selector', '')
        text = params.get('text', '')
        x = params.get('x')
        y = params.get('y')
        
        try:
            clicked = False
            
            if selector and not clicked:
                try:
                    element = await self.dom.find_element(page, selector)
                    if element:
                        await element.click(timeout=5000)
                        clicked = True
                        logger.info(f"✅ Clicked selector: {selector}")
                except Exception as e:
                    logger.debug(f"Selector click failed: {e}")
            
            if text and not clicked:
                try:
                    element = page.get_by_text(text, exact=False).first
                    if await element.count() > 0:
                        await element.click(timeout=5000)
                        clicked = True
                        logger.info(f"✅ Clicked text: {text}")
                except Exception as e:
                    logger.debug(f"Text click failed: {e}")
            
            if x is not None and y is not None and not clicked:
                try:
                    await page.mouse.click(x, y)
                    clicked = True
                    logger.info(f"✅ Clicked at ({x}, {y})")
                except Exception as e:
                    logger.debug(f"Coordinate click failed: {e}")
            
            if clicked:
                await page.wait_for_timeout(500)
                return ActionResult(success=True, action="click", message="Click successful")
            else:
                return ActionResult(success=False, action="click", message="Could not find element to click")
        except Exception as e:
            return ActionResult(success=False, action="click", message=str(e))
    
    async def _type(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Type text"""
        text = params.get('text', '')
        selector = params.get('selector', '')
        submit = params.get('submit', True) # Default to True for backward compat/efficiency
        
        if not text:
            return ActionResult(success=False, action="type", message="No text provided")
        
        try:
            target = None
            if selector:
                target = await self.dom.find_element(page, selector)
            
            if not target:
                # Find best input
                input_selectors = ['input[type="search"]', 'input[name="q"]', 'input:visible', 'textarea:visible']
                for sel in input_selectors:
                    try:
                        el = page.locator(sel).first
                        if await el.count() > 0 and await el.is_visible():
                            target = el
                            break
                    except: continue
            
            if target:
                await target.click()
                await target.fill(text)
                logger.info(f"✅ Typed '{text}'")
                
                if submit:
                    await page.keyboard.press('Enter')
                    await page.wait_for_timeout(1000)
                    logger.info("✅ Pressed Enter")
                
                return ActionResult(success=True, action="type", message=f"Typed: {text}")
            
            return ActionResult(success=False, action="type", message="No input found")
        except Exception as e:
            return ActionResult(success=False, action="type", message=str(e))
    
    async def _scroll(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        direction = params.get('direction', 'down')
        amount = params.get('amount', 500)
        try:
            val = amount if direction == 'down' else -amount
            await page.evaluate(f'window.scrollBy(0, {val})')
            await page.wait_for_timeout(300)
            return ActionResult(success=True, action="scroll", message=f"Scrolled {direction}")
        except Exception as e:
            return ActionResult(success=False, action="scroll", message=str(e))
    
    async def _extract(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        try:
            content = await self.dom.get_page_content(page)
            data = {
                'url': content['url'],
                'title': content['title'],
                'text_content': content['body_text'],
                'element_count': content['element_count']
            }
            return ActionResult(success=True, action="extract", message="Extracted data", data=data)
        except Exception as e:
            return ActionResult(success=False, action="extract", message=str(e))

    async def _screenshot(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Save screenshot to disk"""
        label = params.get('label', 'screenshot')
        try:
            # Generate unique filename
            timestamp = int(time.time())
            file_id = str(uuid.uuid4())[:8]
            filename = f"{label}_{timestamp}_{file_id}.png"
            
            # Storage path
            storage_dir = Path("storage/browser_screenshots")
            storage_dir.mkdir(parents=True, exist_ok=True)
            file_path = storage_dir / filename
            
            # Capture and save
            await page.screenshot(path=str(file_path))
            logger.info(f"📸 Screenshot saved: {file_path}")
            
            # Register with file manager if available
            if self.screenshot_manager:
                try:
                    await self.screenshot_manager.register_file(
                        content=None,
                        filename=filename,
                        file_type="screenshot",
                        file_path=str(file_path),
                        thread_id=self.thread_id,
                        custom_metadata={"label": label}
                    )
                except Exception as reg_err:
                    logger.warning(f"Failed to register screenshot: {reg_err}")
            
            return ActionResult(
                success=True, 
                action="screenshot", 
                message=f"Screenshot saved: {filename}",
                screenshot_id=file_id,
                data={"screenshot_path": str(file_path), "screenshot_id": file_id}
            )
        except Exception as e:
            return ActionResult(success=False, action="screenshot", message=str(e))
