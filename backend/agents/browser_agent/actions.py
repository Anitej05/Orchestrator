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
        """Execute a sequence of actions with smart retry for failures"""
        results_log = []
        final_data = {}
        
        logger.info(f"⚡ Executing Sequence: {[a.name for a in plan.actions]} | {plan.reasoning[:50]}...")
        
        for action in plan.actions:
            try:
                result = await self._execute_single(page, action)
                results_log.append(f"{action.name}: {result.message}")
                
                if result.data:
                    final_data.update(result.data)
                
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
        elif action.name == "select":
            return await self._select_option(page, action.params)
        elif action.name == "wait":
            return await self._wait(page, action.params)
        elif action.name == "go_back":
            try:
                # Capture URL before
                start_url = page.url
                
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
        elif action.name == "screenshot":
            return await self._screenshot(page, action.params)
        elif action.name == "save_info":
            return await self._save_info(page, action.params)
        elif action.name == "skip_subtask":
            return await self._skip_subtask(page, action.params)
        else:
            return ActionResult(success=False, action=action.name, message=f"Unknown action: {action.name}")

    async def _smart_click_fallback(self, page: Page, params: Dict[str, Any]) -> Optional[ActionResult]:
        """Smart fallback when click fails - extract keywords and find matching elements.
        
        This is called when the primary click fails. It:
        1. Extracts keywords from the failed xpath/text/name
        2. Searches for visible links containing those keywords
        3. Clicks the first matching visible element
        """
        import re
        
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
        """Click on element - supports XPath, CSS selector, role+name (a11y), text, or coordinates"""
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
            
            # 1. Coordinate click (highest priority for vision mode)
            if x is not None and y is not None:
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
                    except:
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
                    except:
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
            import time as time_module
            ss_start = time_module.time()
            logger.info(f"📸 _screenshot action: Capturing from {page.url[:50]}...")
            
            await page.screenshot(path=str(file_path), type='jpeg', quality=80, timeout=15000)
            
            ss_elapsed = time_module.time() - ss_start
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
