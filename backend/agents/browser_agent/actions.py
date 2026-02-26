"""
Browser Agent - Action Execution

Execute browser actions reliably. THE MOST CRITICAL COMPONENT.
"""

import logging
import uuid
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from playwright.async_api import Page, Error as PError

from .agent_schemas import ActionPlan, ActionResult, AtomicAction
from .dom import DOMExtractor
from .config import CONFIG
from .persistent_memory import get_persistent_memory
from backend.agents.utils.agent_file_manager import FileType

from backend.services.inference_service import inference_service, InferencePriority, ProviderType
from langchain_core.messages import HumanMessage, SystemMessage
import base64

logger = logging.getLogger(__name__)

# CMS Integration for query_page_content
from backend.services.content_management_service import (
    ContentManagementService, 
    ProcessingTaskType,
    ProcessingStrategy
)
content_service = ContentManagementService()


class ActionExecutor:
    """Execute browser actions reliably"""
    
    def __init__(self, screenshot_manager=None, thread_id: str = None):
        self.dom = DOMExtractor()
        self.screenshot_manager = screenshot_manager
        self.thread_id = thread_id
        self._cached_elements = []
        self._cached_page_text = ""  # Store page text the LLM saw for verification

    def set_cached_elements(self, elements: List[Dict]):
        """Update cached elements for index-based interaction"""
        self._cached_elements = elements
    
    def set_cached_page_text(self, page_text: str):
        """Store the page text that LLM saw - used to verify save_info values"""
        self._cached_page_text = page_text
    
    async def execute(self, page: Page, plan: ActionPlan) -> ActionResult:
        """Execute a sequence of actions with smart retry for failures"""
        results_log = []
        final_data = {}
        
        # Context for variable interpolation between actions
        action_context = {
            'last_run_js_output': None
        }
        
        logger.info(f"⚡ Executing Sequence: {[a.name for a in plan.actions]} | {plan.reasoning[:50]}...")
        
        for action in plan.actions:
            try:
                # Interpolate variables in action params (e.g., {{last_run_js_output}})
                interpolated_params = self._interpolate_params(action.params, action_context)
                interpolated_action = AtomicAction(name=action.name, params=interpolated_params)
                
                result = await self._execute_single(page, interpolated_action)
                results_log.append(f"{action.name}: {result.message}")
                
                # CRITICAL: If navigation occurred, STOP the sequence immediately
                # The page reference is now stale - let the agent loop get a fresh page
                if result.success and action.name in ('navigate', 'click') and 'context changed' in result.message.lower():
                    logger.info(f"🛑 Navigation occurred - stopping sequence to get fresh page")
                    return ActionResult(
                        success=True,
                        action="sequence",
                        message="; ".join(results_log) + " (stopped: navigation occurred)",
                        data=final_data
                    )
                
                if result.data:
                    # ACCUMULATE structured_info from multiple save_info actions
                    if 'structured_info' in result.data:
                        if 'all_saved_items' not in final_data:
                            final_data['all_saved_items'] = []
                        final_data['all_saved_items'].append(result.data['structured_info'])
                    # Also keep the latest for backward compatibility
                    final_data.update(result.data)
                    
                    # Store run_js output for subsequent save_info
                    if action.name == "run_js" and 'result' in result.data:
                        action_context['last_run_js_output'] = result.data['result']
                
                # SMART RETRY for click failures
                if not result.success and action.name == "click":
                    logger.info("🔄 Click failed, attempting smart retry strategies...")
                    
                    # Strategy 1: Wait for dynamic content to settle (DON'T scroll — it closes dropdowns!)
                    await page.wait_for_timeout(2000)
                    result = await self._execute_single(page, action)
                    if result.success:
                        results_log[-1] = f"{action.name}: {result.message} (after retry)"
                        logger.info(f"✅ Retry after wait succeeded!")
                    
                    # Strategy 2: JS click fallback for text-based clicks
                    if not result.success:
                        click_text = action.params.get('text', '')
                        if click_text:
                            try:
                                safe_text = click_text.replace("'", "\\'")
                                js_click = await page.evaluate(f"""
                                    (() => {{
                                        const els = [...document.querySelectorAll('a, button, span, li, div, option')];
                                        const el = els.find(e => e.innerText && e.innerText.trim().includes('{safe_text}'));
                                        if (el) {{ el.click(); return true; }}
                                        return false;
                                    }})()
                                """)
                                if js_click:
                                    result = ActionResult(success=True, action="click", message=f"Clicked via JS fallback: {click_text[:30]}")
                                    results_log[-1] = f"{action.name}: {result.message} (JS fallback)"
                                    logger.info(f"✅ JS click fallback succeeded: {click_text[:30]}")
                            except Exception as js_err:
                                logger.warning(f"JS click fallback failed: {js_err}")
                
                if not result.success:
                    logger.warning(f"⚠️ Action '{action.name}' failed: {result.message}. Stopping sequence.")
                    
                    # ENHANCEMENT: Append strategy suggestions to break loops
                    failure_msg = result.message
                    if "Timeout" in failure_msg or "Nothing clicked" in failure_msg:
                        failure_msg += " SUGGESTION: The element might be hidden or text doesn't match. 1. Try 'run_js' to click/find directly. 2. Verify element visibility with #N index."
                    
                    return ActionResult(
                        success=False, 
                        action=action.name, 
                        message=f"Sequence stopped at {action.name}: {failure_msg}",
                        data=final_data
                    )
            except Exception as e:
                error_str = str(e).lower()
                # Navigation-related "errors" are actually SUCCESS - page context changed!
                if any(nav_signal in error_str for nav_signal in [
                    'target page, context or browser has been closed',
                    'execution context was destroyed',
                    'frame was detached',
                    'navigation',
                    'target closed'
                ]):
                    logger.info(f"✅ Navigation detected during {action.name} - context changed (this is success!)")
                    return ActionResult(
                        success=True, 
                        action=action.name, 
                        message=f"Navigation triggered by {action.name}",
                        data=final_data
                    )
                logger.error(f"Critical execution error on {action.name}: {e}")
                return ActionResult(success=False, action=action.name, message=str(e), data=final_data)

        return ActionResult(
            success=True, 
            action="sequence", 
            message="; ".join(results_log),
            data=final_data
        )
    
    def _interpolate_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Interpolate variables like {{last_run_js_output}} and {{last_run_js_output.property}} in action parameters"""
        import re
        
        if not params:
            return params
        
        result = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Pattern 1: Full {{last_run_js_output}} replacement
                if '{{last_run_js_output}}' in value and context.get('last_run_js_output') is not None:
                    result[key] = value.replace('{{last_run_js_output}}', str(context['last_run_js_output']))
                    logger.info(f"📝 Interpolated {{{{last_run_js_output}}}} -> {result[key][:100]}...")
                
                # Pattern 2: Property access like {{last_run_js_output.topRated}} or {{last_run_js_output.items[0]}}
                elif '{{last_run_js_output.' in value and context.get('last_run_js_output') is not None:
                    # Find all property access patterns
                    pattern = r'\{\{last_run_js_output\.([^}]+)\}\}'
                    
                    def replace_property(match):
                        prop_path = match.group(1)  # e.g., "topRated" or "items[0].name"
                        js_output = context['last_run_js_output']
                        
                        try:
                            # Navigate the property path
                            current = js_output
                            
                            # Handle both dot notation and bracket notation
                            # Split by . but preserve [n] indices
                            parts = re.split(r'\.(?![^\[]*\])', prop_path)
                            
                            for part in parts:
                                # Check for array index like items[0]
                                bracket_match = re.match(r'(\w+)\[(\d+)\]', part)
                                if bracket_match:
                                    prop_name = bracket_match.group(1)
                                    index = int(bracket_match.group(2))
                                    if isinstance(current, dict) and prop_name in current:
                                        current = current[prop_name]
                                    if isinstance(current, list) and len(current) > index:
                                        current = current[index]
                                else:
                                    if isinstance(current, dict) and part in current:
                                        current = current[part]
                                    else:
                                        return f"{{{{error: property '{prop_path}' not found}}}}"
                            
                            return str(current)
                        except Exception as e:
                            logger.warning(f"Property access failed for '{prop_path}': {e}")
                            return f"{{{{error: {e}}}}}"
                    
                    result[key] = re.sub(pattern, replace_property, value)
                    logger.info(f"📝 Interpolated property access -> {result[key][:100]}...")
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    async def _execute_single(self, page: Page, action: AtomicAction) -> ActionResult:
        if action.name == "navigate":
            return await self._navigate(page, action.params)
        elif action.name == "click":
            return await self._click(page, action.params)
        elif action.name == "click_coordinate":
            return await self._click_coordinate(page, action.params.get('x'), action.params.get('y'))
        elif action.name == "type":
            return await self._type(page, action.params)
        elif action.name == "scroll":
            return await self._scroll(page, action.params)
        elif action.name == "hover":
            return await self._hover(page, action.params)
        elif action.name == "press":
            return await self._press(page, action.params)
        elif action.name == "select":
            return await self._select_option(page, action.params)
        elif action.name == "wait":
            return await self._wait(page, action.params)
        elif action.name == "search_text":
            return await self._search_text(page, action.params)
        elif action.name == "scan_page":
            return await self._scan_page(page, action.params)
        elif action.name == "go_back":
            try:
                # Check if we CAN go back (has history)
                # Wrap in try/except because evaluate might fail if page is dead
                try:
                    can_go_back = await page.evaluate("() => window.history.length > 1")
                except Exception:
                    # If we can't check history, assume we can't go back or page is dead
                    return ActionResult(success=False, action="go_back", message="Could not check history (page might be closed).")

                if not can_go_back:
                    return ActionResult(
                        success=False,
                        action="go_back",
                        message="No browser history to go back to. Use 'navigate' to go to a specific URL instead."
                    )
                
                # Capture URL before (if possible)
                start_url = page.url
                
                # Go back with wait
                await page.go_back(wait_until='domcontentloaded', timeout=15000)
                
                # Verify navigation
                current_url = page.url
                if start_url == current_url:
                     return ActionResult(
                         success=False, 
                         action="go_back", 
                         message="Go back failed: URL did not change. Try using 'navigate' to the specific URL instead."
                     )
                
                return ActionResult(success=True, action="go_back", message="Went back successfully")
            except Exception as e:
                # Handle "Target closed" specifically as a soft failure
                msg = str(e)
                if "Target page, context or browser has been closed" in msg:
                    return ActionResult(success=False, action="go_back", message="Browser context closed during navigation. Agent will recover.")
                return ActionResult(success=False, action="go_back", message=msg)

        elif action.name == "go_forward":
            try:
                await page.go_forward()
                return ActionResult(success=True, action="go_forward", message="Went forward")
            except Exception as e:
                msg = str(e)
                if "Target page, context or browser has been closed" in msg:
                    return ActionResult(success=False, action="go_forward", message="Browser context closed during navigation. Agent will recover.")
                return ActionResult(success=False, action="go_forward", message=msg)
        elif action.name == "extract":
            return await self._extract(page, action.params)
        elif action.name == "done":
            # Support done with inline data: {"name": "done", "data": {"ram": "12GB", ...}}
            done_data = action.params.get('data', {})
            if done_data:
                return ActionResult(
                    success=True, action="done",
                    message="Task complete (with data)",
                    data={'structured_info': {'key': 'final_result', 'value': done_data, 'verified': True, 'source': 'done_action'}}
                )
            return ActionResult(success=True, action="done", message="Task complete")
        elif action.name == "save_screenshot":
            return await self._save_screenshot(page, action.params)
        elif action.name == "save_info":
            return await self._save_info(page, action.params)
        elif action.name == "skip_subtask":
            return await self._skip_subtask(page, action.params)
        elif action.name == "upload_file":
            return await self._upload_file(page, action.params)
        elif action.name == "download_file":
            return await self._download_file(page, action.params)
        elif action.name == "run_js":
            return await self._run_javascript(page, action.params)
        elif action.name == "remove_element":
            return await self._remove_element(page, action.params)
        elif action.name == "press_keys":
            return await self._press_keys(page, action.params)
        elif action.name == "save_credential":
            return await self._save_credential(page, action.params)
        elif action.name == "get_credential":
            return await self._get_credential(page, action.params)
        elif action.name == "search_text":
            return await self._search_text(page, action.params)

    async def _query_page_content(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """
        Query large page content offloaded to CMS (RAG-over-Page).
        Expected params: {'query': 'search term'}
        """
        query = params.get('query', '')
        
        # We need the active content ID. Since ActionExecutor is stateless regarding agent memory,
        # we rely on the agent to inject this ID into the params or we need to access the shared memory.
        # IMPROVEMENT: The planner should inject 'active_content_id' into params if available.
        # Fallback: Check if we have a way to access agent state.
        
        # For this implementation, we will assume the AGENT injects 'content_id' into params
        # when it sees 'query_page_content' action being planned and it knows it has an active large page.
        content_id = params.get('content_id')
        
        if not query:
             return ActionResult(success=False, action="query_page_content", message="No query provided")
             
        if not content_id:
             return ActionResult(
                 success=False, 
                 action="query_page_content", 
                 message="No active content ID found. Use this tool only when a 'LARGE PAGE' has been detected and offloaded."
             )
             
        try:
            logger.info(f"🔍 CMS Query: '{query}' on Content ID: {content_id}")
            
            # Execute Map-Reduce Search via CMS
            result = await content_service.process_large_content(
                content_id=content_id,
                task_type=ProcessingTaskType.SEARCH,
                query=query,
                strategy=ProcessingStrategy.MAP_REDUCE # Parallel search across chunks
            )
            
            final_answer = result.final_output
            logger.info(f"✅ CMS Query Result ({len(final_answer)} chars)")
            
            return ActionResult(
                success=True, 
                action="query_page_content", 
                message=f"Query successful",
                data={'result': final_answer, 'structured_info': {'key': f"query_{query}", 'value': final_answer}}
            )
            
        except Exception as e:
            logger.error(f"CMS Query Failed: {e}")
            return ActionResult(success=False, action="query_page_content", message=f"CMS Query failed: {e}")




    async def _hover(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Hover over element - supports XPath, CSS selector, role+name, or coordinates"""
        selector = params.get('selector', '')
        xpath = params.get('xpath', '')
        role = params.get('role', '')
        name = params.get('name', '')
        x = params.get('x')
        y = params.get('y')
        
        try:
            hovered = False
            
            # 1. Coordinates
            if x is not None and y is not None:
                await page.mouse.move(x, y)
                hovered = True
                logger.info(f"✅ Hovered via coordinates: ({x}, {y})")
            
            # 2. XPath
            if not hovered and xpath:
                try:
                    locator = page.locator(f"xpath={xpath}").first
                    if await locator.count() > 0:
                        await locator.hover()
                        hovered = True
                        logger.info(f"✅ Hovered via XPath: {xpath[:40]}")
                except Exception as e:
                    logger.warning(f"XPath hover failed: {e}")
            
            # 3. Role+Name
            if not hovered and role and name:
                try:
                    role_map = {'link': 'link', 'button': 'button', 'menuitem': 'menuitem'}
                    pw_role = role_map.get(role.lower(), role.lower())
                    locator = page.get_by_role(pw_role, name=name).first
                    if await locator.count() > 0:
                        await locator.hover()
                        hovered = True
                        logger.info(f"✅ Hovered via role+name: [{role}] {name}")
                except Exception as e:
                    logger.warning(f"Role+name hover failed: {e}")
            
            # 4. CSS Selector
            if not hovered and selector:
                element = await self.dom.find_element(page, selector)
                if element:
                    await element.hover()
                    hovered = True
                    logger.info(f"✅ Hovered via selector: {selector}")
            
            if hovered:
                await page.wait_for_timeout(500)
                return ActionResult(success=True, action="hover", message=f"Hovered successfully")
            
            return ActionResult(success=False, action="hover", message="Element not found for hover")
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

    async def _click_coordinate(self, page: Page, x: int, y: int) -> ActionResult:
        """Click a specific absolute (x,y) screen coordinate."""
        try:
            if x is None or y is None:
                return ActionResult(success=False, action="click_coordinate", message="Requires both 'x' and 'y' integer parameters")
                
            x, y = int(x), int(y)
            await page.mouse.click(x, y)
            
            # Wait a moment for dynamic interactions to process
            await page.wait_for_timeout(500)
            
            return ActionResult(success=True, action="click_coordinate", message=f"Clicked absolute coordinate ({x}, {y})")
        except Exception as e:
            logger.error(f"click_coordinate Error: {str(e)}")
            return ActionResult(success=False, action="click_coordinate", message=f"Failed to click coordinate ({x}, {y}): {str(e)}")

    async def _wait(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Wait for seconds"""
        seconds = params.get('seconds', 1)
        try:
            await page.wait_for_timeout(seconds * 1000)
            return ActionResult(success=True, action="wait", message=f"Waited {seconds}s")
        except PError as e:
            msg = str(e)
            if "Target closed" in msg or "Session closed" in msg:
                 logger.warning(f"⚠️ Wait interrupted by target close: {msg}")
                 # Return success so we don't abort the sequence immediately.
                 # The next action will likely fail if page is truly gone, 
                 # but we avoid a crash on 'wait'.
                 return ActionResult(success=True, action="wait", message=f"Waited {seconds}s (Target Closed)")
            return ActionResult(success=False, action="wait", message=msg)
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
            
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await page.wait_for_timeout(2000)  # Wait for SPAs
            
            logger.info(f"✅ Navigated to: {url}")
            return ActionResult(success=True, action="navigate", message=f"Navigated to {url}")
        except Exception as e:
            msg = str(e)
            
            # "Target page, context or browser has been closed" often means navigation 
            # happened successfully but triggered a page context change (redirect, new tab)
            # The URL DID change - we should treat this as success
            if "closed" in msg.lower() or "destroyed" in msg.lower() or "navigation" in msg.lower():
                logger.info(f"⚠️ Page context changed during navigation to {url} - treating as success")
                
                # Try to get fresh page reference from browser context
                try:
                    if hasattr(self, 'browser') and self.browser:
                        fresh_page = self.browser.get_active_page()
                        if fresh_page:
                            logger.info(f"🔄 Got fresh page after navigation: {fresh_page.url[:60] if fresh_page.url else 'new page'}")
                except Exception:
                    pass  # Browser reference not available here
                
                # CRITICAL: Wait for the redirect/new tab to fully settle
                # Without this delay, the main loop tries to access the page
                # before the new page context is ready, causing cascading failures
                await asyncio.sleep(1.5)
                
                return ActionResult(
                    success=True, 
                    action="navigate", 
                    message=f"Navigated to {url} (page context changed)"
                )
            
            is_timeout = "Timeout" in msg or "timeout" in msg
            return ActionResult(
                success=False, 
                action="navigate", 
                message=msg,
                timeout_occurred=is_timeout,
                timeout_context={"action": "navigate", "url": url} if is_timeout else None
            )
    
    async def _select_option(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Select an option from a dropdown by label, value, or index"""
        selector = params.get('selector', '')
        xpath = params.get('xpath', '')
        label = params.get('label', '')  # Visible text of option
        value = params.get('value', '')  # Value attribute
        index = params.get('index')  # 0-based index
        
        try:
            # Find the select element
            if xpath:
                select = page.locator(f"xpath={xpath}").first
            elif selector:
                select = page.locator(selector).first
            else:
                return ActionResult(success=False, action="select", message="No selector or xpath provided")
            
            # Select the option
            if label:
                await select.select_option(label=label)
                return ActionResult(success=True, action="select", message=f"Selected option: {label}")
            elif value:
                await select.select_option(value=value)
                return ActionResult(success=True, action="select", message=f"Selected value: {value}")
            elif index is not None:
                await select.select_option(index=index)
                return ActionResult(success=True, action="select", message=f"Selected index: {index}")
            else:
                return ActionResult(success=False, action="select", message="No label, value, or index provided")
                
        except Exception as e:
            return ActionResult(success=False, action="select", message=f"Select failed: {str(e)}")
            
    async def _robust_click(self, locator, timeout: int = 3000) -> bool:
        """
        Attempts to click a locator robustly.
        Playwright's actionability checks often fail on modern custom components (like Amazon dropdowns) 
        where the logical element is covered by an invisible wrapper div or span.
        1. Try standard click with force=True to bypass actionability overlays.
        2. If that fails, inject JS to execute .click() directly on the DOM node.
        """
        try:
            # force=True bypasses Playwright's actionability checks
            # This is critical for custom dropdowns where options are often covered by invisible wrappers
            await locator.click(timeout=timeout, force=True)
            return True
        except Exception:
            # Standard click blocked natively
            # Fallback to direct JS click on the element as a last resort
            try:
                element = await locator.element_handle()
                if element:
                    await element.evaluate("el => el.click()")
                    return True
            except Exception:
                pass
        return False
    
    async def _click(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Click on element - supports index, XPath, CSS selector, role+name (a11y), text, or coordinates"""
        # DEBUG PARAM LOGGING
        logger.info(f"DEBUG_CLICK params: {params}")
        
        index = params.get('index')  # Element index from DOM list (#N)
        xpath = params.get('xpath', '')
        selector = params.get('selector', '')
        text = params.get('text', '')
        role = params.get('role', '')
        name = params.get('name', '')
        x = params.get('x')
        y = params.get('y')
        
        # Track what we tried for better error messages
        attempts = []
        
        try:
            clicked = False
            
            # 0. Index-based click (highest priority - clicks specific element from DOM list)
            if not clicked and index is not None:
                try:
                    # Get elements from DOM - index is 1-based from LLM
                    elements = getattr(self, '_cached_elements', [])
                    elem_idx = int(index) - 1  # Convert to 0-based
                    if 0 <= elem_idx < len(elements):
                        el = elements[elem_idx]
                        el_x = el.get('x')
                        el_y = el.get('y')
                        if el_x is not None and el_y is not None:
                            await page.mouse.click(el_x, el_y)
                            clicked = True
                            logger.info(f"✅ Clicked via index #{index}: ({el_x}, {el_y}) - {el.get('name', '')[:30]}")
                        else:
                            # Try xpath if coordinates not available
                            el_xpath = el.get('xpath', '')
                            if el_xpath:
                                locator = page.locator(f"xpath={el_xpath}").first
                                if await locator.count() > 0:
                                    if await self._robust_click(locator, timeout=5000):
                                        clicked = True
                                        logger.info(f"✅ Clicked via index #{index} xpath (robust): {el_xpath[:50]}")
                            if not clicked:
                                attempts.append(f"index({index}):no coords/xpath")
                    else:
                        attempts.append(f"index({index}):out of range (max {len(elements)})")
                        logger.warning(f"Index {index} out of range. Available: 1-{len(elements)}")
                except Exception as idx_err:
                    attempts.append(f"index:{str(idx_err)[:30]}")
                    logger.warning(f"Index-based click failed: {idx_err}")
            
            # 1. Coordinate click (highest priority for vision mode)
            if not clicked and x is not None and y is not None:
                # Coordinates from DOM are PAGE-relative (include scroll).
                # We need to scroll the element into view and click at viewport-relative position.
                try:
                    # Get current scroll position
                    scroll_y = await page.evaluate("() => window.scrollY")
                    viewport_height = await page.evaluate("() => window.innerHeight")
                    
                    # Calculate if element is visible in current viewport
                    viewport_y = y - scroll_y  # Convert to viewport-relative
                    
                    # If element is outside viewport, scroll to it first
                    if viewport_y < 0 or viewport_y > viewport_height:
                        # Scroll so element is centered in viewport
                        target_scroll = max(0, y - viewport_height // 2)
                        await page.evaluate(f"window.scrollTo(0, {target_scroll})")
                        await page.wait_for_timeout(300)  # Let scroll settle
                        
                        # Recalculate viewport Y after scroll
                        new_scroll_y = await page.evaluate("() => window.scrollY")
                        viewport_y = y - new_scroll_y
                        logger.info(f"📜 Scrolled to bring element into view (scroll: {new_scroll_y})")
                    
                    # Validate the viewport-relative coordinates are now valid
                    if 0 <= viewport_y <= viewport_height + 50 and 0 <= x <= 1400:
                        await page.mouse.click(x, viewport_y)
                        clicked = True
                        logger.info(f"✅ Clicked via coordinates: page({x}, {y}) → viewport({x}, {viewport_y})")
                    else:
                        attempts.append(f"coords({x},{y}):still outside after scroll")
                        logger.warning(f"⚠️ Coordinates ({x}, {y}) still outside viewport after scroll")
                except Exception as coord_err:
                    attempts.append(f"coords({x},{y}):{str(coord_err)[:30]}")
                    logger.warning(f"Coordinate click failed: ({x}, {y}) - {coord_err}")
            
            # 2. XPath click (most reliable for text mode)
            if not clicked and xpath:
                try:
                    locator = page.locator(f"xpath={xpath}").first
                    count = await locator.count()
                    if count > 0:
                        if await self._robust_click(locator, timeout=5000):
                            clicked = True
                            logger.info(f"✅ Clicked via XPath (robust): {xpath[:50]}")
                        else:
                            attempts.append(f"xpath(robust_failed)")
                    else:
                        attempts.append(f"xpath(not found)")
                        logger.warning(f"XPath not found: {xpath[:50]}")
                except Exception as xpath_err:
                    attempts.append(f"xpath:{str(xpath_err)[:30]}")
                    logger.warning(f"XPath robust click failed: {xpath[:50]}")
            
            # 3. Role + Name click (from accessibility tree) - VERY ROBUST
            if not clicked and role and name:
                try:
                    # Map common role names to Playwright role types
                    role_map = {
                        'link': 'link', 'button': 'button', 'textbox': 'textbox',
                        'heading': 'heading', 'checkbox': 'checkbox', 'radio': 'radio',
                        'combobox': 'combobox', 'listbox': 'listbox', 'option': 'option',
                        'menuitem': 'menuitem', 'tab': 'tab', 'searchbox': 'searchbox',
                        'search': 'searchbox', 'input': 'textbox'  # Common aliases
                    }
                    pw_role = role_map.get(role.lower(), role.lower())
                    
                    # Try exact name first, then partial
                    for name_to_try in [name, name[:20]]:
                        if clicked: break
                        if not name_to_try: continue
                        
                        locator = page.get_by_role(pw_role, name=name_to_try).first
                        if await locator.count() > 0:
                            if await self._robust_click(locator, timeout=3000):
                                clicked = True
                                logger.info(f"✅ Clicked via role+name (robust): [{role}] {name_to_try}")
                            else:
                                attempts.append(f"role_robust_fail")
                                    
                    if not clicked:
                        attempts.append(f"role({role}):{name[:20]}(not found)")
                        
                except Exception as role_err:
                    attempts.append(f"role:{str(role_err)[:30]}")
                    logger.warning(f"Role+name click failed: [{role}] {name} - {role_err}")
            
            # 4. Text click (before CSS selector as it's more specific)
            if not clicked and text:
                try:
                    # First try exact match, then loose match
                    for exact_match in [True, False]:
                        if clicked: break
                        
                        locator = page.get_by_text(text, exact=exact_match).first
                        count = await locator.count()
                        if count > 0:
                            if await self._robust_click(locator, timeout=3000):
                                clicked = True
                                logger.info(f"✅ Clicked via text (exact={exact_match}, robust): {text[:30]}")
                            else:
                                attempts.append(f"text_robust_fail")
                        
                    if not clicked:
                        attempts.append(f"text(not found)")
                except Exception as text_err:
                    attempts.append(f"text:{str(text_err)[:30]}")
                    logger.warning(f"Text click failed: {text[:30]} - {text_err}")
            
            # 5. CSS Selector click (last resort for specific selectors)
            if not clicked and selector:
                try:
                    element = page.locator(selector).first
                    if await element.count() > 0:
                        if await self._robust_click(element, timeout=5000):
                            clicked = True
                            logger.info(f"✅ Clicked via CSS selector (robust): {selector[:30]}")
                        else:
                            attempts.append(f"selector_robust_fail")
                    else:
                        attempts.append(f"selector(not found)")
                except Exception as sel_err:
                    attempts.append(f"selector:{str(sel_err)[:30]}")
                    logger.warning(f"CSS selector click failed: {selector[:30]}")

            if clicked:
                try:
                    await page.wait_for_timeout(500)
                except Exception:
                    pass  # Context may have changed during click - that's OK
                return ActionResult(success=True, action="click", message="Click successful")
            else:
                error_detail = f"Tried: {', '.join(attempts)}" if attempts else "No valid params provided"
                return ActionResult(success=False, action="click", message=f"Nothing clicked. {error_detail}")
                 
        except Exception as e:
            msg = str(e)
            is_timeout = "Timeout" in msg or "timeout" in msg
            return ActionResult(
                success=False, 
                action="click", 
                message=str(e),
                timeout_occurred=is_timeout,
                timeout_context={"action": "click", "xpath": xpath, "selector": selector, "text": text} if is_timeout else None
            )
    

    async def _remove_element(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Remove an element from the DOM"""
        index = params.get('index')
        xpath = params.get('xpath', '')
        selector = params.get('selector', '')
        
        try:
            removed = False
            
            # 1. Index-based removal
            if index is not None:
                elements = getattr(self, '_cached_elements', [])
                try:
                    elem_idx = int(index) - 1
                    if 0 <= elem_idx < len(elements):
                        el = elements[elem_idx]
                        el_xpath = el.get('xpath', '')
                        if el_xpath:
                            locator = page.locator(f"xpath={el_xpath}").first
                            if await locator.count() > 0:
                                await locator.evaluate("el => el.remove()")
                                removed = True
                                logger.info(f"🗑️ Removed element via index #{index}: {el_xpath[:50]}")
                except Exception as idx_err:
                    logger.warning(f"Index removal failed: {idx_err}")
            
            # 2. XPath removal
            if not removed and xpath:
                locator = page.locator(f"xpath={xpath}").first
                if await locator.count() > 0:
                    await locator.evaluate("el => el.remove()")
                    removed = True
                    logger.info(f"🗑️ Removed element via XPath: {xpath[:50]}")
            
            # 3. Selector removal
            if not removed and selector:
                locator = page.locator(selector).first
                if await locator.count() > 0:
                    await locator.evaluate("el => el.remove()")
                    removed = True
                    logger.info(f"🗑️ Removed element via Selector: {selector[:30]}")

            if removed:
                return ActionResult(success=True, action="remove_element", message="Element removed from DOM")
            else:
                return ActionResult(success=False, action="remove_element", message="Element not found to remove")
        
        except Exception as e:
            return ActionResult(success=False, action="remove_element", message=f"Failed to remove element: {e}")

    async def _type(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Type text into input element"""
        text = params.get('text', '')
        selector = params.get('selector', '')
        xpath = params.get('xpath', '')
        role = params.get('role', '')
        name = params.get('name', '')
        submit = params.get('submit', True)  # Default to True for backward compat/efficiency
        
        if not text:
            return ActionResult(success=False, action="type", message="No text provided")
        
        try:
            target = None
            
            # 1. Try XPath first
            if xpath:
                try:
                    locator = page.locator(f"xpath={xpath}").first
                    if await locator.count() > 0:
                        target = locator
                        logger.info(f"Found input via XPath: {xpath[:40]}")
                except Exception as e:
                    logger.warning(f"XPath input not found: {xpath[:40]} - {e}")
            
            # 2. Try role+name
            if not target and role and name:
                try:
                    role_map = {'textbox': 'textbox', 'searchbox': 'searchbox', 'input': 'textbox', 'search': 'searchbox'}
                    pw_role = role_map.get(role.lower(), role.lower())
                    locator = page.get_by_role(pw_role, name=name).first
                    if await locator.count() > 0:
                        target = locator
                        logger.info(f"Found input via role+name: [{role}] {name}")
                except Exception as e:
                    logger.warning(f"Role+name input not found: [{role}] {name} - {e}")
            
            # 3. Try CSS selector
            if not target and selector:
                target = await self.dom.find_element(page, selector)
            
            # 4. Find best visible input (fallback)
            if not target:
                input_selectors = [
                    'input[type="search"]',
                    'input[name="q"]',
                    'input[aria-label*="Search"]',
                    'input[placeholder*="Search"]',
                    'textarea:visible',
                    'input:visible'
                ]
                for sel in input_selectors:
                    try:
                        el = page.locator(sel).first
                        if await el.count() > 0 and await el.is_visible():
                            target = el
                            logger.info(f"Found input via fallback selector: {sel}")
                            break
                    except Exception:
                        continue
            
            if target:
                await target.click()
                await target.fill(text)
                logger.info(f"✅ Typed '{text[:30]}...'")
                
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
        max_retries = 3
        
        try:
            # Get initial position and page metrics
            scroll_info = await page.evaluate('''
                () => ({
                    scrollY: window.scrollY,
                    scrollX: window.scrollX,
                    innerHeight: window.innerHeight,
                    scrollHeight: document.documentElement.scrollHeight,
                    maxScrollY: document.documentElement.scrollHeight - window.innerHeight
                })
            ''')
            
            start_y = scroll_info['scrollY']
            max_scroll_y = scroll_info['maxScrollY']
            
            # Check if already at limit BEFORE attempting (save time)
            # Returns success=False so the agent's no-effect detection catches scroll loops
            if direction == 'down' and start_y >= max_scroll_y - 1:
                return ActionResult(
                    success=False,  # Signal to agent: scrolling further is pointless
                    action="scroll", 
                    message=f"CANNOT scroll down — already at bottom of page (position: {int(start_y)}/{int(max_scroll_y)}). Use run_js to extract data from the current viewport instead.",
                    data={"scroll_position": int(start_y), "max_scroll": int(max_scroll_y), "at_bottom": True, "at_limit": True}
                )
            elif direction == 'up' and start_y <= 1:
                return ActionResult(
                    success=False,  # Signal to agent: scrolling further is pointless
                    action="scroll", 
                    message=f"CANNOT scroll up — already at top of page (position: 0/{int(max_scroll_y)}). The content you see IS the top. Use run_js to extract data directly.",
                    data={"scroll_position": 0, "max_scroll": int(max_scroll_y), "at_top": True, "at_limit": True}
                )
            
            # Try scrolling with retry and decreasing amounts
            amounts_to_try = [amount, amount // 2, 100, 50]
            
            for retry, try_amount in enumerate(amounts_to_try):
                if retry >= max_retries:
                    break
                    
                # Perform scroll
                val = try_amount if direction == 'down' else -try_amount
                await page.evaluate(f'window.scrollBy(0, {val})')
                await page.wait_for_timeout(300)
                
                # Verify scroll
                end_y = await page.evaluate('window.scrollY')
                delta = abs(end_y - start_y)
                
                if delta > 0:
                    logger.info(f"✅ Scrolled {direction} by {delta}px (requested {amount}, used {try_amount})")
                    return ActionResult(
                        success=True, 
                        action="scroll", 
                        message=f"Scrolled {direction} by {delta}px (position: {int(end_y)}/{int(max_scroll_y)})",
                        data={"scroll_position": int(end_y), "max_scroll": int(max_scroll_y), "scrolled_by": delta}
                    )
                
                # If first attempt failed, log and retry with smaller amount
                if retry < len(amounts_to_try) - 1:
                    logger.warning(f"⚠️ Scroll attempt {retry+1} failed with amount {try_amount}, retrying with {amounts_to_try[retry+1]}")
            
            # All retries exhausted
            at_limit = (direction == 'down' and start_y >= max_scroll_y - 10) or (direction == 'up' and start_y <= 10)
            if at_limit:
                return ActionResult(
                    success=True,  # At limit is not a failure
                    action="scroll", 
                    message=f"At {direction} limit of page (position: {int(start_y)}/{int(max_scroll_y)})",
                    data={"scroll_position": int(start_y), "max_scroll": int(max_scroll_y), "at_limit": True}
                )
            else:
                return ActionResult(
                    success=False, 
                    action="scroll", 
                    message=f"Could not scroll {direction} after {max_retries} attempts (position: {int(start_y)}/{int(max_scroll_y)})",
                    data={"scroll_position": int(start_y), "max_scroll": int(max_scroll_y)}
                )
                
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

    async def _search_text(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Find text on page and return surrounding context to the LLM"""
        query = params.get('query', '')
        if not query:
            return ActionResult(success=False, action="search_text", message="No query provided")
            
        try:
            # Inject JS to find text nodes and return their parent's textContext
            js_code = f"""
                (() => {{
                    const query = '{query.lower()}';
                    const results = [];
                    // Simple text search in DOM
                    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
                    let node;
                    while (node = walker.nextNode()) {{
                        if (node.nodeValue.toLowerCase().includes(query)) {{
                            // Get parent element's full text for context
                            const parent = node.parentElement;
                            if (parent && parent.innerText) {{
                                const text = parent.innerText.replace(/\\s+/g, ' ').trim();
                                if (text && !results.includes(text)) {{
                                    results.push(text);
                                }}
                            }}
                        }}
                    }}
                    return results.slice(0, 5); // Return top 5 matches context
                }})()
            """
            matches = await page.evaluate(js_code)
            
            if matches:
                 # Extract the context surrounding the query
                 context_texts = [f"Match {i+1}: {t[:250]}" for i, t in enumerate(matches)]
                 
                 # Also try to scroll to the first match
                 try:
                     scroll_js = f"""
                        (() => {{
                            const query = '{query.lower()}';
                            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
                            let node;
                            while (node = walker.nextNode()) {{
                                if (node.nodeValue.toLowerCase().includes(query)) {{
                                    if (node.parentElement) {{
                                        node.parentElement.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                                        return true;
                                    }}
                                }}
                            }}
                            return false;
                        }})()
                     """
                     await page.evaluate(scroll_js)
                     await page.wait_for_timeout(1000) # Wait for smooth scroll
                 except Exception as e:
                     logger.warning(f"Could not scroll to searched text: {e}")
                     
                 # Join outputs for message
                 joined_matches = "\\n".join(context_texts)
                 
                 return ActionResult(
                     success=True, 
                     action="search_text", 
                     message=f"Found {len(matches)} matches for '{query}'",
                     data={"matches": joined_matches, 'structured_info': {'key': f'search_{query}', 'value': joined_matches, 'verified': True}}
                 )
            else:
                 return ActionResult(success=False, action="search_text", message=f"Text '{query}' not found on page")
                 
        except Exception as e:
             return ActionResult(success=False, action="search_text", message=f"Search failed: {e}")

    async def _screenshot(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Save screenshot to disk with robust handling"""
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
            
            # Capture and save with timeout and logging
            ss_start = time.time()
            logger.info(f"📸 _screenshot action: Capturing from {page.url[:50]}...")
            
            await page.screenshot(path=str(file_path), type='jpeg', quality=80, timeout=15000)
            
            ss_elapsed = time.time() - ss_start
            logger.info(f"📸 Screenshot saved in {ss_elapsed:.2f}s: {file_path}")
            
            # Register with file manager if available
            if self.screenshot_manager:
                try:
                    # Read the screenshot bytes from the saved file
                    with open(file_path, 'rb') as f:
                        screenshot_bytes = f.read()
                    
                    await self.screenshot_manager.register_file(
                        content=screenshot_bytes,
                        filename=filename,
                        file_type="screenshot",
                        thread_id=self.thread_id,
                        custom_metadata={"label": label}
                    )
                    logger.info(f"📁 Screenshot registered with file manager: {filename}")
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

    async def _save_info(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Save specific structured information found on the page.
        
        Validates that the value actually exists on the page to prevent hallucination.
        Returns a verified flag indicating whether the value was found.
        """
        key = params.get('key', 'unknown_info')
        value = params.get('value', '')
        source = params.get('source')
        
        if not source:
            try:
                source = page.url
            except Exception:
                source = "unknown"
        
        # VALIDATION: Check if value exists in PAGE CONTENT that LLM saw (not live page)
        # This is more reliable because we verify against exactly what the LLM was given
        verified = False
        validation_context = ""
        
        if value and len(str(value).strip()) >= 2:
            try:
                # Use cached page text (what LLM saw) - more reliable than live page
                if self._cached_page_text:
                    page_text = self._cached_page_text
                    validation_context = "cached_context"
                else:
                    # Fallback to live page if cache not available
                    page_text = await page.evaluate("document.body.innerText || ''")
                    validation_context = "live_page"
                
                value_str = str(value).strip().lower()
                page_text_lower = page_text.lower()
                
                # Check for exact match first
                if value_str in page_text_lower:
                    verified = True
                    logger.info(f"✅ save_info value verified (exact): {key}='{str(value)[:50]}'")
                else:
                    # TOKEN-BASED VERIFICATION for combined values
                    # Extract significant tokens (prices, model names, ratings, numbers)
                    import re
                    
                    # Extract significant tokens from value
                    tokens = []
                    
                    # Prices: $642.99, ₹56,490, etc
                    prices = re.findall(r'[\$₹€£]?\s*[\d,]+(?:\.\d{2})?', str(value))
                    tokens.extend([p.strip() for p in prices if len(p.strip()) >= 3])
                    
                    # Model identifiers: RTX 3050, i7-13620H, DDR5, etc
                    models = re.findall(r'\b(?:RTX|GTX|RX|i[3579]|Ryzen|DDR[45]|SSD|FHD|QHD|Hz)\s*[\w-]*\b', str(value), re.IGNORECASE)
                    tokens.extend(models)
                    
                    # Ratings: 4.3 stars, (1210 ratings), etc
                    ratings = re.findall(r'\d+\.?\d*\s*(?:star|rating|review)', str(value), re.IGNORECASE)
                    tokens.extend(ratings)
                    
                    # Brand names
                    brands = re.findall(r'\b(?:MSI|Asus|Acer|Dell|HP|Lenovo|ASUS)\b', str(value), re.IGNORECASE)
                    tokens.extend(brands)
                    
                    if tokens:
                        # Count how many tokens are found on page
                        found_count = 0
                        for token in tokens:
                            if token.lower() in page_text_lower:
                                found_count += 1
                        
                        match_ratio = found_count / len(tokens)
                        
                        if match_ratio >= 0.5:  # 50%+ tokens found
                            verified = True
                            validation_context = f"token_match:{found_count}/{len(tokens)}"
                            logger.info(f"✅ save_info value verified (tokens: {found_count}/{len(tokens)}): {key}='{str(value)[:50]}'")
                        else:
                            logger.warning(f"⚠️ save_info token match low ({found_count}/{len(tokens)}): {key}='{str(value)[:50]}'")
                    else:
                        # No significant tokens - try digit match as fallback
                        value_digits = re.sub(r'[^\d.]', '', str(value))
                        if value_digits and len(value_digits) >= 2 and value_digits in page_text:
                            verified = True
                            validation_context = "digit_match"
                            logger.info(f"✅ save_info value verified (digits): {key}='{str(value)[:50]}'")
                    
                    if not verified:
                        logger.warning(f"⚠️ save_info value NOT verified: {key}='{str(value)[:50]}'")
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                validation_context = f"validation_error: {str(e)[:50]}"
        else:
            # Empty or very short values - mark as verified if intentionally empty
            verified = value == "" or value is None
        
        return ActionResult(
            success=True,
            action="save_info",
            message=f"Saved info: {key}='{str(value)[:50]}...' [{'VERIFIED' if verified else 'UNVERIFIED'}]",
            data={
                "structured_info": {
                    "key": key,
                    "value": value,
                    "source": source,
                    "verified": verified,
                    "validation_context": validation_context,
                    "timestamp": time.time()
                }
            }
        )
    
    async def _skip_subtask(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Explicitly skip the current subtask due to issues"""
        reason = params.get('reason', 'No reason provided')
        logger.warning(f"⏩ SKIPPING SUBTASK: {reason}")
        return ActionResult(
             success=True,
             action="skip_subtask",
             message=f"Created skipping request: {reason}",
             data={"skipped": True, "reason": reason}
        )

    async def _upload_file(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Upload a file to a file input element
        
        Params:
            file_path: Path to the file (can be filename in uploads dir or absolute path)
            selector: CSS selector for the file input (default: input[type='file'])
            xpath: XPath for the file input (alternative to selector)
            index: Index of file input on page (1-based)
        """
        file_path_str = params.get('file_path') or params.get('filename')
        selector = params.get('selector')
        xpath = params.get('xpath')
        index = params.get('index')
        
        if not file_path_str:
            return ActionResult(
                success=False,
                action="upload_file",
                message="No file_path provided. Specify the file to upload."
            )
        
        # Resolve file path using config
        resolved_path = CONFIG.get_upload_path(file_path_str)
        if not resolved_path:
            available = CONFIG.list_available_uploads()
            return ActionResult(
                success=False,
                action="upload_file",
                message=f"File not found: {file_path_str}. Available files in uploads folder: {available[:10]}"
            )
        
        try:
            # Find the file input element
            file_input = None
            
            if xpath:
                file_input = page.locator(f"xpath={xpath}").first
            elif selector:
                file_input = page.locator(selector).first
            elif index:
                # Find by index (1-based)
                all_inputs = page.locator("input[type='file']")
                count = await all_inputs.count()
                if 0 < index <= count:
                    file_input = all_inputs.nth(index - 1)
                else:
                    return ActionResult(
                        success=False,
                        action="upload_file",
                        message=f"File input index {index} out of range. Found {count} file inputs."
                    )
            else:
                # Default: find first file input
                file_input = page.locator("input[type='file']").first
            
            if not file_input or await file_input.count() == 0:
                return ActionResult(
                    success=False,
                    action="upload_file",
                    message="No file input element found on page."
                )
            
            # Upload the file
            await file_input.set_input_files(str(resolved_path))
            
            # Wait for any upload processing
            await page.wait_for_timeout(1000)
            
            logger.info(f"📤 Uploaded file: {resolved_path.name}")
            return ActionResult(
                success=True,
                action="upload_file",
                message=f"Uploaded file: {resolved_path.name}",
                data={"uploaded_file": str(resolved_path), "filename": resolved_path.name}
            )
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return ActionResult(
                success=False,
                action="upload_file",
        )

    async def _save_screenshot(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Save a screenshot of the current page to disk
        
        Params:
            filename: Optional custom filename (e.g. "checkout_page.jpg")
            full_page: Whether to capture full scrollable page (default: False)
        """
        custom_filename = params.get('filename')
        full_page = params.get('full_page', False)
        
        try:
            # Generate filename if not provided
            if not custom_filename:
                timestamp = int(time.time())
                custom_filename = f"screenshot_{timestamp}.jpg"
            
            # Ensure extension
            if not custom_filename.endswith(('.jpg', '.jpeg', '.png')):
                custom_filename += ".jpg"
                
            save_path = CONFIG.get_screenshot_path(custom_filename)
            
            # Determine format
            is_jpeg = custom_filename.endswith(('.jpg', '.jpeg'))
            
            # Take screenshot - quality only valid for JPEG, not PNG
            screenshot_options = {
                'path': str(save_path),
                'full_page': full_page,
                'type': 'jpeg' if is_jpeg else 'png',
            }
            if is_jpeg:
                screenshot_options['quality'] = 80
                
            await page.screenshot(**screenshot_options)
            
            # Register with File Manager if available
            if self.screenshot_manager:
                try:
                    # Read content to register
                    if save_path.exists():
                        with open(save_path, 'rb') as f:
                            file_content = f.read()
                            
                        await self.screenshot_manager.register_file(
                            content=file_content,
                            filename=custom_filename,
                            file_type=FileType.SCREENSHOT,
                            thread_id=self.thread_id,
                            custom_metadata={"action": "save_screenshot"}
                        )
                except Exception as e:
                    logger.warning(f"Failed to register screenshot: {e}")
            
            file_size = save_path.stat().st_size
            logger.info(f"📸 Saved screenshot: {custom_filename} ({file_size} bytes)")
            
            return ActionResult(
                success=True,
                action="save_screenshot",
                message=f"Screenshot saved: {custom_filename}",
                data={
                    "screenshot_path": str(save_path),
                    "filename": custom_filename,
                    "size_bytes": file_size
                }
            )
            
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return ActionResult(
                success=False,
                action="save_screenshot",
                message=f"Failed to save screenshot: {str(e)}"
            )

    async def _download_file(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Trigger a download by clicking an element and capture the file
        
        Params:
            xpath: XPath of download link/button
            selector: CSS selector of download link/button
            text: Text of download link/button
            url: Direct URL to download (alternative to clicking)
            filename: Optional custom filename to save as
            wait_timeout: Timeout in seconds to wait for download (default: 30)
        """
        xpath = params.get('xpath')
        selector = params.get('selector')
        text = params.get('text')
        url = params.get('url')
        custom_filename = params.get('filename')
        wait_timeout = params.get('wait_timeout', 30) * 1000  # Convert to ms
        
        try:
            
            if url:
                # Direct URL download
                async with page.expect_download(timeout=wait_timeout) as download_info:
                    await page.goto(url)
                download = await download_info.value
                
            else:
                # Find and click the download trigger
                element = None
                
                if xpath:
                    element = page.locator(f"xpath={xpath}").first
                elif selector:
                    element = page.locator(selector).first
                elif text:
                    # Try link first, then button
                    element = page.locator(f"a:has-text('{text}')").first
                    if await element.count() == 0:
                        element = page.locator(f"button:has-text('{text}')").first
                
                if not element or await element.count() == 0:
                    return ActionResult(
                        success=False,
                        action="download_file",
                        message="Could not find download element. Provide xpath, selector, or text."
                    )
                
                # Click and wait for download
                async with page.expect_download(timeout=wait_timeout) as download_info:
                    await element.click()
                download = await download_info.value
            
            # Determine save path
            suggested_name = download.suggested_filename
            final_filename = custom_filename or suggested_name
            save_path = CONFIG.get_download_path(final_filename)
            
            # Save the file
            await download.save_as(str(save_path))
            
            # Verify download completed
            if save_path.exists():
                file_size = save_path.stat().st_size
                logger.info(f"📥 Downloaded: {final_filename} ({file_size} bytes)")
                return ActionResult(
                    success=True,
                    action="download_file",
                    message=f"Downloaded: {final_filename} ({file_size} bytes)",
                    data={
                        "download_path": str(save_path),
                        "filename": final_filename,
                        "size_bytes": file_size,
                        "suggested_filename": suggested_name
                    }
                )
            else:
                return ActionResult(
                    success=False,
                    action="download_file",
                    message=f"Download may have failed - file not found at {save_path}"
                )
                
        except asyncio.TimeoutError:
            return ActionResult(
                success=False,
                action="download_file",
                message=f"Download timed out after {wait_timeout/1000} seconds. The element may not trigger a download."
            )
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return ActionResult(
                success=False,
                action="download_file",
                message=f"Download failed: {str(e)}"
            )

    async def _run_javascript(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Execute JavaScript code on the page
        
        Params:
            code: JavaScript code to execute
            return_value: If true, return the result of the script (default: true)
            timeout: Timeout in ms (default: 30000)
            
        Examples:
            - Extract React state: {"code": "return window.__REACT_DEVTOOLS_GLOBAL_HOOK__"}
            - Get localStorage: {"code": "return JSON.stringify(localStorage)"}
            - Scroll to element: {"code": "document.querySelector('#target').scrollIntoView()"}
            - Click hidden button: {"code": "document.querySelector('button.hidden').click()"}
            - Get page data: {"code": "return document.body.dataset"}
        """
        code = params.get('code') or params.get('script')
        params.get('return_value', True)
        params.get('timeout', 30000)
        
        if not code:
            return ActionResult(
                success=False,
                action="run_js",
                message="No JavaScript code provided. Use 'code' parameter."
            )
        
        try:
            # SIMPLIFIED WRAPPER: Just wrap in an IIFE that can handle both sync and async code
            # Don't use nested functions which break variable scope
            wrapped_code = f"""(() => {{
                {code}
            }})()"""
            
            result = await page.evaluate(wrapped_code)
            
            # Store result for {{last_run_js_output}} interpolation
            self.last_js_result = result
            
            # Format result for display
            if result is None:
                result_str = "Script executed (no return value)"
            elif isinstance(result, (dict, list)):
                import json
                result_str = json.dumps(result, indent=2, default=str)[:500]
            else:
                result_str = str(result)[:500]
            
            logger.info(f"🔧 JavaScript executed successfully: {result_str[:100]}...")
            return ActionResult(
                success=True,
                action="run_js",
                message=f"JavaScript executed: {result_str[:200]}",
                data={
                    "result": result, 
                    "code_preview": code[:100],
                    "auto_extracted": isinstance(result, (dict, list)) and result is not None
                }
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"JavaScript execution failed: {error_msg}")
            return ActionResult(
                success=False,
                action="run_js",
                message=f"JavaScript error: {error_msg[:200]}"
            )

    async def _press_keys(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Press keyboard keys or key combinations
        
        Params:
            keys: Key or key combination to press
                - Single key: "Enter", "Escape", "Tab", "Backspace", "Delete"
                - Arrow keys: "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"
                - Modifiers: "Control+a", "Control+c", "Control+v", "Alt+F4"
                - Function keys: "F1", "F5", "F12"
                - Multiple keys: ["Tab", "Tab", "Enter"] (press in sequence)
            delay: Delay between key presses in ms (default: 50)
            
        Examples:
            - Close modal: {"keys": "Escape"}
            - Submit form: {"keys": "Enter"}
            - Select all: {"keys": "Control+a"}
            - Copy: {"keys": "Control+c"}
            - Navigate: {"keys": ["Tab", "Tab", "Enter"]}
            - Refresh: {"keys": "F5"}
            - Find on page: {"keys": "Control+f"}
        """
        keys = params.get('keys') or params.get('key')
        delay = params.get('delay', 50)
        
        if not keys:
            return ActionResult(
                success=False,
                action="press_keys",
                message="No keys specified. Use 'keys' parameter."
            )
        
        try:
            # Handle list of keys (press in sequence)
            if isinstance(keys, list):
                for key in keys:
                    await page.keyboard.press(key)
                    await page.wait_for_timeout(delay)
                pressed = ", ".join(keys)
            else:
                # Single key or combination
                await page.keyboard.press(keys)
                pressed = keys
            
            logger.info(f"⌨️ Pressed: {pressed}")
            return ActionResult(
                success=True,
                action="press_keys",
                message=f"Pressed: {pressed}",
                data={"keys_pressed": keys}
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Key press failed: {error_msg}")
            return ActionResult(
                success=False,
                action="press_keys",
                message=f"Key press failed: {error_msg}"
            )

    # ============== PERSISTENT MEMORY ACTIONS ==============
    
    async def _save_credential(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Save login credentials for a site to persistent memory.
        
        Params:
            site: Site domain (e.g. "amazon.in", "github.com")
            username: Username or email
            password: Password
            notes: Optional notes about this credential
        """
        site = params.get('site', '')
        username = params.get('username', '')
        password = params.get('password', '')
        notes = params.get('notes', '')
        
        if not site:
            try:
                site = page.url.split('/')[2]
            except Exception:
                return ActionResult(
                    success=False,
                    action="save_credential",
                    message="No site specified. Provide 'site' parameter or ensure you're on a page."
                )
        
        if not username or not password:
            return ActionResult(
                success=False,
                action="save_credential",
                message="Both 'username' and 'password' are required."
            )
        
        try:
            persistent = get_persistent_memory()
            persistent.save_credential(site, username, password, notes)
            logger.info(f"🔐 Saved credentials for {site}")
            
            return ActionResult(
                success=True,
                action="save_credential",
                message=f"Credentials saved for {site}. Will be available in future sessions.",
                data={"site": site, "username": username}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="save_credential",
                message=f"Failed to save credentials: {str(e)}"
            )
    
    async def _get_credential(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Retrieve stored credentials for a site.
        
        Params:
            site: Site domain to get credentials for (optional, uses current page if not specified)
        """
        site = params.get('site', '')
        
        if not site:
            try:
                site = page.url.split('/')[2]
            except Exception:
                return ActionResult(
                    success=False,
                    action="get_credential",
                    message="No site specified. Provide 'site' parameter or ensure you're on a page."
                )
        
        try:
            persistent = get_persistent_memory()
            cred = persistent.get_credential(site)
            
            if cred:
                logger.info(f"🔐 Retrieved credentials for {site}")
                return ActionResult(
                    success=True,
                    action="get_credential",
                    message=f"Found credentials for {site}: username={cred['username']}",
                    data={
                        "site": site,
                        "username": cred['username'],
                        "password": cred['password'],  # Available for auto-fill
                        "notes": cred.get('notes', '')
                    }
                )
            else:
                return ActionResult(
                    success=False,
                    action="get_credential",
                    message=f"No stored credentials for {site}. Ask user to provide login info."
                )
        except Exception as e:
            return ActionResult(
                success=False,
                action="get_credential",
                message=f"Failed to retrieve credentials: {str(e)}"
            )
    
    async def _save_learning(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Save a learning/fact to persistent memory for future sessions.
        
        Params:
            category: Category of learning (e.g., "site_navigation", "user_preference", "element_pattern")
            key: Key/name for this learning
            value: The actual learning/fact
            confidence: Optional confidence score (0-1, default 1.0)
        """
        category = params.get('category', 'general')
        key = params.get('key', '')
        value = params.get('value', '')
        confidence = params.get('confidence', 1.0)
        
        if not key or not value:
            return ActionResult(
                success=False,
                action="save_learning",
                message="Both 'key' and 'value' are required for save_learning."
            )
        
        try:
            source = page.url if page else None
            persistent = get_persistent_memory()
            persistent.add_learning(category, key, value, source, confidence)
            
            logger.info(f"📝 Saved learning: [{category}] {key}")
            return ActionResult(
                success=True,
                action="save_learning",
                message=f"Learning saved: [{category}] {key} = {value[:50]}...",
                data={
                    "category": category,
                    "key": key,
                    "value": value,
                    "persistent": True
                }
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="save_learning",
                message=f"Failed to save learning: {str(e)}"
            )

    async def _search_text(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """Search for text on the page and scroll it into view (Simulates Ctrl+F)"""
        query = params.get('query')
        if not query:
            return ActionResult(success=False, action="search_text", message="query parameter is required")
        
        try:
            # Step 1: Try Playwright's locator to find the element and scroll to it
            # This is robust because it handles Shadow DOM and custom components well
            locator = page.get_by_text(query, exact=False).first
            count = await locator.count()
            
            if count > 0:
                try:
                    # Scroll it into view
                    await locator.scroll_into_view_if_needed(timeout=2000)
                    
                    # Highlight it temporarily to simulate Ctrl+F visual feedback
                    try:
                        element = await locator.element_handle()
                        if element:
                            await element.evaluate('''el => {
                                const originalBg = el.style.backgroundColor;
                                el.style.transition = 'background-color 0.5s ease';
                                el.style.backgroundColor = 'yellow';
                                setTimeout(() => { el.style.backgroundColor = originalBg; }, 2000);
                            }''')
                    except Exception:
                        pass # Ignore highlight failures
                    
                    logger.info(f"✅ Found and scrolled to text via locator: '{query[:30]}'")
                    return ActionResult(success=True, action="search_text", message=f"Found '{query}' and scrolled it into view")
                except Exception as loc_err:
                    logger.debug(f"Locator scroll failed, falling back to window.find: {loc_err}")
            
            # Step 2: Fallback to native window.find()
            # This is the literal Ctrl+F browser API, but it might miss ShadowDOM elements
            result = await page.evaluate(f'''(query) => {{
                // Reset search to top if we've reached the bottom
                if (window.scrollY + window.innerHeight >= document.body.scrollHeight) {{
                    window.scrollTo(0, 0);
                }}
                return window.find(query);
            }}''', query)
            
            if result:
                logger.info(f"✅ Found and scrolled to text via window.find: '{query[:30]}'")
                return ActionResult(success=True, action="search_text", message=f"Found '{query}' and scrolled it into view")
            
            return ActionResult(success=False, action="search_text", message=f"Text '{query}' not found on page")
            
        except Exception as e:
            logger.warning(f"search_text action failed: {e}")
            return ActionResult(success=False, action="search_text", message=f"Search failed: {str(e)}")

    async def _scan_page(self, page: Page, params: Dict[str, Any]) -> ActionResult:
        """
        Scan Mode Hack: Automates manual scrolling by taking consecutive viewport screenshots 
        and asking the Vision LLM to find the target information in a single batched inference call.
        """
        query = params.get('query')
        if not query:
            return ActionResult(success=False, action="scan_page", message="query parameter is required")
        
        try:
            # 1. Scroll to top to begin scan
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)
            
            viewport_height = await page.evaluate("window.innerHeight")
            total_height = await page.evaluate("document.body.scrollHeight")
            
            max_scrolls = 6  # Limit to 6 viewports to avoid token bloat/infinite scroll
            screenshots_b64 = []
            scroll_positions = []
            
            logger.info(f"📸 Starting batched page scan for: '{query}'")
            
            # 2. Loop through viewports and capture screenshots
            for step in range(max_scrolls):
                current_y = await page.evaluate("window.scrollY")
                scroll_positions.append(current_y)
                
                # Take screenshot
                screenshot_bytes = await page.screenshot(type="jpeg", quality=60) # lower quality for batching
                b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                screenshots_b64.append(f"data:image/jpeg;base64,{b64}")
                
                # Check if we've reached the bottom
                if current_y + viewport_height >= total_height - 10:
                    break
                    
                # Scroll down one viewport
                await page.evaluate(f"window.scrollBy(0, {viewport_height})")
                await page.wait_for_timeout(400) # Give it time to render
                
            # 3. Construct the Vision prompt
            prompt = (
                f"You are a visual search assistant. I am scanning a long webpage for: '{query}'.\n"
                f"I have attached {len(screenshots_b64)} consecutive screenshots of the page scrolling from top to bottom.\n\n"
                f"Here are the Y-coordinates for each screenshot image (in order from 1 to {len(screenshots_b64)}):\n"
            )
            for idx, y_pos in enumerate(scroll_positions):
                prompt += f"Image {idx+1}: Y={y_pos}px\n"
                
            prompt += (
                "\nWhich image contains the best match for the query? "
                "Respond ONLY with a JSON object containing the winning Y-coordinate and a brief reason. "
                "If none of the images contain the target, return -1.\n\n"
                'Example format: {"found": true, "y_coordinate": 850, "reason": "Pricing table is clearly visible here"}\n'
                'Or: {"found": false, "y_coordinate": -1, "reason": "No pricing details found in any screenshot"}'
            )
            
            messages = [
                SystemMessage(content="You are a strict JSON-only visual search agent."),
                HumanMessage(content=prompt)
            ]
            
            logger.info(f"🧠 Sending {len(screenshots_b64)} images to Vision LLM for analysis...")
            
            # Call inference service directly (bypassing conversational memory)
            response = await inference_service.generate(
                messages=messages,
                provider=ProviderType.OPENAI,
                model_name="kimi-k2.5:cloud", # Assuming the multimodal model from llm.py
                priority=InferencePriority.QUALITY,
                temperature=0.1,
                json_mode=True,
                images=screenshots_b64,
                fallback_enabled=False,
                use_cache=False
            )
            
            if not response:
                return ActionResult(success=False, action="scan_page", message="Vision LLM returned no answer.")
                
            # 4. Parse response and teleport to coordinate
            try:
                import json
                # Strip markdown code blocks if any
                clean_json = response.strip()
                if clean_json.startswith('```json'):
                    clean_json = clean_json[7:-3]
                elif clean_json.startswith('```'):
                    clean_json = clean_json[3:-3]
                    
                data = json.loads(clean_json.strip())
                
                if data.get('found', False) and data.get('y_coordinate', -1) != -1:
                    target_y = data['y_coordinate']
                    await page.evaluate(f"window.scrollTo(0, {target_y})")
                    await page.wait_for_timeout(500)
                    reason = data.get('reason', '')
                    logger.info(f"✅ Scan successful! Teleported to Y={target_y}. Reason: {reason}")
                    return ActionResult(success=True, action="scan_page", message=f"Successfully found visual target. Scrolled to Y={target_y}. Context: {reason}")
                else:
                    return ActionResult(success=False, action="scan_page", message=f"Target not found in any of the {len(screenshots_b64)} viewports scanned. Reason: {data.get('reason')}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Vision response: {response}")
                return ActionResult(success=False, action="scan_page", message=f"Failed to parse Vision response: {e}")
            
        except Exception as e:
            logger.warning(f"Feature scan_page failed: {e}")
            return ActionResult(success=False, action="scan_page", message=f"Scan failed: {str(e)}")
