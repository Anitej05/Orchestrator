"""
Browser Agent - Action Execution

Execute browser actions reliably. THE MOST CRITICAL COMPONENT.
"""

import logging
import re
import uuid
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from playwright.async_api import Page, TimeoutError as PTimeoutError, Error as PError

from .schemas import ActionPlan, ActionResult, AtomicAction
from .dom import DOMExtractor
from .config import CONFIG

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Execute browser actions reliably"""
    
    def __init__(self, screenshot_manager=None, thread_id: str = None):
        self.dom = DOMExtractor()
        self.screenshot_manager = screenshot_manager
        self.thread_id = thread_id
        self._cached_elements = []

    def set_cached_elements(self, elements: List[Dict]):
        """Update cached elements for index-based interaction"""
        self._cached_elements = elements
    
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
                
                if result.data:
                    final_data.update(result.data)
                    # Store run_js output for subsequent save_info
                    if action.name == "run_js" and 'result' in result.data:
                        action_context['last_run_js_output'] = result.data['result']
                
                # SMART RETRY for click failures
                if not result.success and action.name == "click":
                    logger.info("🔄 Click failed, attempting smart retry strategies...")
                    
                    # Strategy 1: Scroll to potential element location
                    await page.evaluate("window.scrollBy(0, 300)")
                    await page.wait_for_timeout(500)
                    
                    # Strategy 2: Try smart text-based fallback
                    retry_result = await self._smart_click_fallback(page, action.params)
                    if retry_result and retry_result.success:
                        result = retry_result
                        results_log[-1] = f"{action.name}: {result.message} (via smart retry)"
                        logger.info(f"✅ Smart retry succeeded!")
                    else:
                        # Strategy 3: Wait longer for dynamic content and retry original
                        await page.wait_for_timeout(1500)
                        result = await self._execute_single(page, action)
                        if result.success:
                            results_log[-1] = f"{action.name}: {result.message} (after wait)"
                            logger.info(f"✅ Retry after wait succeeded!")
                
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
                logger.error(f"Critical execution error on {action.name}: {e}")
                return ActionResult(success=False, action=action.name, message=str(e), data=final_data)

        return ActionResult(
            success=True, 
            action="sequence", 
            message="; ".join(results_log),
            data=final_data
        )
    
    def _interpolate_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Interpolate variables like {{last_run_js_output}} in action parameters"""
        if not params:
            return params
        
        result = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Replace {{last_run_js_output}} with actual value
                if '{{last_run_js_output}}' in value and context.get('last_run_js_output'):
                    result[key] = value.replace('{{last_run_js_output}}', str(context['last_run_js_output']))
                    logger.info(f"📝 Interpolated {{{{last_run_js_output}}}} -> {result[key][:100]}...")
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
        elif action.name == "go_back":
            try:
                # Capture URL before
                start_url = page.url
                
                # Check if we CAN go back (has history)
                can_go_back = await page.evaluate("() => window.history.length > 1")
                if not can_go_back:
                    return ActionResult(
                        success=False,
                        action="go_back",
                        message="No browser history to go back to. Use 'navigate' to go to a specific URL instead."
                    )
                
                # Go back with wait
                await page.go_back(wait_until='domcontentloaded', timeout=15000)
                await page.wait_for_timeout(1000) # Ensure state settles
                
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
        elif action.name == "press_keys":
            return await self._press_keys(page, action.params)
        else:
            return ActionResult(success=False, action=action.name, message=f"Unknown action: {action.name}")

    async def _smart_click_fallback(self, page: Page, params: Dict[str, Any]) -> Optional[ActionResult]:
        """Smart fallback when click fails - extract keywords and find matching elements.
        
        This is called when the primary click fails. It:
        1. Extracts keywords from the failed xpath/text/name
        2. Searches for visible links containing those keywords
        3. Clicks the first matching visible element
        """
        
        # Extract search keywords from various params
        keywords = []
        
        xpath = params.get('xpath', '')
        text = params.get('text', '')
        name = params.get('name', '')
        
        # Extract from xpath like //a[contains(text(),"Galaxy S25 Ultra 5G AI Smartp
        if xpath:
            # Look for text patterns in xpath
            text_match = re.search(r'contains\([^,]+,\s*["\']([^"\']+)', xpath)
            if text_match:
                keywords.append(text_match.group(1))
            # Also try aria-label patterns
            aria_match = re.search(r'aria-label[*]?=\s*["\']([^"\']+)', xpath)
            if aria_match:
                keywords.append(aria_match.group(1))
        
        if text:
            keywords.append(text)
        if name:
            keywords.append(name)
        
        if not keywords:
            logger.warning("🔍 No keywords found for smart fallback")
            return None
        
        logger.info(f"🔍 Smart click fallback - searching for keywords: {keywords}")
        
        # Try to find and click elements containing these keywords
        
        # Strategy 0: Check MEMORY (cached elements) for text match
        if hasattr(self, '_cached_elements') and self._cached_elements:
            for keyword in keywords:
                keyword_clean = keyword.strip().lower()
                if len(keyword_clean) < 3: continue
                
                for el in self._cached_elements:
                    # Check text content, name, or aria-label
                    el_text = (el.get('text') or el.get('name') or '').lower()
                    el_label = (el.get('attributes') or {}).get('aria-label', '').lower()
                    
                    if keyword_clean in el_text or keyword_clean in el_label:
                        xpath = el.get('xpath')
                        if xpath:
                            try:
                                logger.info(f"🧠 Memory Hit! Found '{keyword}' in cached element: {el.get('name')}")
                                locator = page.locator(f"xpath={xpath}").first
                                await locator.click(timeout=5000)
                                return ActionResult(success=True, action="click", message=f"Smart fallback (Memory): clicked '{el.get('name')}'")
                            except Exception as e:
                                logger.warning(f"Memory click failed: {e}")

        # Continue with page scanning strategies
        for keyword in keywords:
            try:
                # Clean keyword - remove trailing truncation
                keyword = keyword.strip().rstrip('.')
                if len(keyword) < 5:
                    continue  # Too short to be useful
                
                # Strategy 1: Get by text (partial match)
                try:
                    locator = page.get_by_text(keyword[:30], exact=False).first
                    if await locator.count() > 0:
                        # Verify it's visible
                        if await locator.is_visible():
                            await locator.scroll_into_view_if_needed()
                            await page.wait_for_timeout(200)
                            await locator.click(timeout=3000)
                            logger.info(f"✅ Smart fallback clicked via text: {keyword[:30]}...")
                            return ActionResult(success=True, action="click", message=f"Smart fallback: clicked '{keyword[:30]}...'")
                except Exception as e:
                    logger.debug(f"Smart text click failed: {e}")
                
                # Strategy 2: Find links containing keyword in href
                try:
                    # Use product ID patterns common in e-commerce
                    links = await page.query_selector_all(f'a[href*="{keyword[:20]}"]')
                    if links:
                        for link in links[:3]:  # Try first 3 matches
                            if await link.is_visible():
                                await link.scroll_into_view_if_needed()
                                await page.wait_for_timeout(200)
                                await link.click()
                                logger.info(f"✅ Smart fallback clicked via href match")
                                return ActionResult(success=True, action="click", message="Smart fallback: clicked via href")
                except Exception as e:
                    logger.debug(f"Smart href click failed: {e}")
                
                # Strategy 3: Find any clickable element with matching content
                try:
                    # Broader search for the first significant keyword
                    first_word = keyword.split()[0] if ' ' in keyword else keyword[:15]
                    locator = page.locator(f'a:has-text("{first_word}")').first
                    if await locator.count() > 0 and await locator.is_visible():
                        await locator.scroll_into_view_if_needed()
                        await page.wait_for_timeout(200)
                        await locator.click(timeout=3000)
                        logger.info(f"✅ Smart fallback clicked via keyword: {first_word}")
                        return ActionResult(success=True, action="click", message=f"Smart fallback: clicked '{first_word}'")
                except Exception as e:
                    logger.debug(f"Smart keyword click failed: {e}")
                    
            except Exception as e:
                logger.debug(f"Smart fallback strategy failed for '{keyword}': {e}")
                continue
        
        logger.warning("🔍 Smart click fallback exhausted all strategies")
        return None

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
            
            await page.goto(url, wait_until='domcontentloaded', timeout=600000)
            await page.wait_for_timeout(3000) # Increased wait for SPAs
            
            
            logger.info(f"✅ Navigated to: {url}")
            return ActionResult(success=True, action="navigate", message=f"Navigated to {url}")
        except Exception as e:
            msg = str(e)
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
                                    await locator.click(timeout=5000)
                                    clicked = True
                                    logger.info(f"✅ Clicked via index #{index} xpath: {el_xpath[:50]}")
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
                try:
                    await page.mouse.click(x, y)
                    clicked = True
                    logger.info(f"✅ Clicked via coordinates: ({x}, {y})")
                except Exception as coord_err:
                    attempts.append(f"coords({x},{y}):{str(coord_err)[:30]}")
                    logger.warning(f"Coordinate click failed: ({x}, {y}) - {coord_err}")
            
            # 2. XPath click (most reliable for text mode)
            if not clicked and xpath:
                try:
                    locator = page.locator(f"xpath={xpath}").first
                    count = await locator.count()
                    if count > 0:
                        await locator.click(timeout=5000)
                        clicked = True
                        logger.info(f"✅ Clicked via XPath: {xpath[:50]}")
                    else:
                        attempts.append(f"xpath(not found)")
                        logger.warning(f"XPath not found: {xpath[:50]}")
                except Exception as xpath_err:
                    attempts.append(f"xpath:{str(xpath_err)[:30]}")
                    # Try JS click fallback
                    try:
                        element = await page.query_selector(f"xpath={xpath}")
                        if element:
                            await element.evaluate("el => el.click()")
                            clicked = True
                            logger.info(f"✅ Clicked via XPath JS fallback: {xpath[:50]}")
                        else:
                            logger.warning(f"XPath element not found for JS click: {xpath[:50]}")
                    except Exception as js_err:
                        attempts.append(f"xpath_js:{str(js_err)[:20]}")
                        logger.warning(f"XPath JS click also failed: {xpath[:50]}")
            
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
                    locator = page.get_by_role(pw_role, name=name).first
                    count = await locator.count()
                    if count > 0:
                        await locator.click(timeout=5000)
                        clicked = True
                        logger.info(f"✅ Clicked via role+name: [{role}] {name}")
                    else:
                        attempts.append(f"role({role}):{name[:20]}(not found)")
                        # Try partial name match
                        locator = page.get_by_role(pw_role, name=name[:20]).first
                        if await locator.count() > 0:
                            await locator.click(timeout=5000)
                            clicked = True
                            logger.info(f"✅ Clicked via role+partial name: [{role}] {name[:20]}...")
                except Exception as role_err:
                    attempts.append(f"role:{str(role_err)[:30]}")
                    logger.warning(f"Role+name click failed: [{role}] {name} - {role_err}")
            
            # 4. Text click (before CSS selector as it's more specific)
            if not clicked and text:
                try:
                    locator = page.get_by_text(text, exact=False).first
                    count = await locator.count()
                    if count > 0:
                        await locator.click(timeout=5000)
                        clicked = True
                        logger.info(f"✅ Clicked via text: {text[:30]}")
                    else:
                        attempts.append(f"text(not found)")
                except Exception as text_err:
                    attempts.append(f"text:{str(text_err)[:30]}")
                    logger.warning(f"Text click failed: {text[:30]} - {text_err}")
            
            # 5. CSS Selector click (last resort for specific selectors)
            if not clicked and selector:
                try:
                    element = page.locator(selector).first
                    if await element.count() > 0:
                        await element.click(timeout=5000)
                        clicked = True
                        logger.info(f"✅ Clicked via CSS selector: {selector[:30]}")
                    else:
                        attempts.append(f"selector(not found)")
                except Exception as sel_err:
                    attempts.append(f"selector:{str(sel_err)[:30]}")
                    # Try JS click as final fallback
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            await element.evaluate("el => el.click()")
                            clicked = True
                            logger.info(f"✅ Clicked via CSS JS fallback: {selector[:30]}")
                    except Exception:
                        logger.warning(f"CSS selector click failed: {selector[:30]}")

            if clicked:
                await page.wait_for_timeout(500)
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
            if direction == 'down' and start_y >= max_scroll_y - 1:
                return ActionResult(
                    success=True,  # Not a failure, just at bottom
                    action="scroll", 
                    message=f"Already at bottom of page (position: {int(start_y)}/{int(max_scroll_y)})",
                    data={"scroll_position": int(start_y), "max_scroll": int(max_scroll_y), "at_bottom": True}
                )
            elif direction == 'up' and start_y <= 1:
                return ActionResult(
                    success=True,  # Not a failure, just at top
                    action="scroll", 
                    message=f"Already at top of page (position: 0/{int(max_scroll_y)})",
                    data={"scroll_position": 0, "max_scroll": int(max_scroll_y), "at_top": True}
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
        """Save specific structured information found on the page"""
        key = params.get('key', 'unknown_info')
        value = params.get('value', '')
        source = params.get('source')
        
        if not source:
             try:
                source = page.url
             except:
                source = "unknown"

        return ActionResult(
            success=True,
            action="save_info",
            message=f"Saved info: {key}='{str(value)[:50]}...'",
            data={
                "structured_info": {
                    "key": key,
                    "value": value,
                    "source": source
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
            
            # Take screenshot
            await page.screenshot(
                path=str(save_path),
                full_page=full_page,
                type='jpeg' if custom_filename.endswith(('.jpg', '.jpeg')) else 'png',
                quality=80
            )
            
            # Register with File Manager if available
            if self.screenshot_manager:
                try:
                    await self.screenshot_manager.register_file(
                        content=None,
                        filename=custom_filename,
                        file_type="screenshot",
                        file_path=str(save_path),
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
            download_path = None
            
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
        return_value = params.get('return_value', True)
        timeout = params.get('timeout', 30000)
        
        if not code:
            return ActionResult(
                success=False,
                action="run_js",
                message="No JavaScript code provided. Use 'code' parameter."
            )
        
        try:
            # Wrap code to return value if not already returning
            if return_value and not code.strip().startswith('return'):
                # Check if it's an expression that should return a value
                if not any(code.strip().startswith(kw) for kw in ['if', 'for', 'while', 'try', 'const', 'let', 'var', 'function']):
                    code = f"return {code}"
            
            # Execute with async function wrapper for async code support
            wrapped_code = f"""
                (async () => {{
                    {code}
                }})()
            """
            
            result = await page.evaluate(wrapped_code)
            
            # Format result for display
            if result is None:
                result_str = "Script executed (no return value)"
            elif isinstance(result, (dict, list)):
                import json
                result_str = json.dumps(result, indent=2, default=str)[:1000]
            else:
                result_str = str(result)[:1000]
            
            logger.info(f"🔧 JavaScript executed successfully")
            return ActionResult(
                success=True,
                action="run_js",
                message=f"JavaScript executed: {result_str[:200]}",
                data={"result": result, "code_preview": code[:100]}
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
