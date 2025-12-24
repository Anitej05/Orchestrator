"""
Browser Agent - DOM Extraction (SOTA)

Extract page content with:
- Accessibility Tree (semantic structure)
- Robust XPath generation (reliable selectors)
- Coordinates (vision fallback)
"""

import logging
from typing import List, Dict, Any, Optional
from playwright.async_api import Page

logger = logging.getLogger(__name__)


class DOMExtractor:
    """Extract DOM content with a11y tree and robust XPath"""
    
    async def get_page_content(self, page: Page) -> Dict[str, Any]:
        """Get comprehensive page content for LLM"""
        try:
            url = page.url
            title = await page.title()
            
            # Get visible text (with null check)
            body_text = await page.evaluate('''() => {
                if (!document.body) return '';
                return document.body.innerText.substring(0, 200000);
            }''')
            
            # Get scroll position info
            scroll_info = await page.evaluate('''() => ({
                scrollY: window.scrollY,
                scrollX: window.scrollX,
                innerHeight: window.innerHeight,
                scrollHeight: document.documentElement.scrollHeight,
                maxScrollY: Math.max(0, document.documentElement.scrollHeight - window.innerHeight),
                scrollPercent: document.documentElement.scrollHeight > window.innerHeight 
                    ? Math.round((window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100)
                    : 100
            })''')
            
            # Get interactive elements with robust XPath
            elements = await self._get_interactive_elements(page)
            
            # DEBUG: Log extracted elements
            logger.info(f"🔍 DOM Extracted {len(elements)} elements")
            if elements:
                for i, el in enumerate(elements[:5]):
                    logger.info(f"  [{i+1}] {el.get('role', '?')}: '{el.get('name', '')[:30]}' → {el.get('xpath', 'NO XPATH')}")
            
            # Get accessibility tree (semantic structure)
            a11y_tree = await self._get_accessibility_tree(page)
            
            return {
                'url': url,
                'title': title,
                'body_text': body_text,
                'elements': elements[:500],
                'element_count': len(elements),
                'a11y_tree': a11y_tree,  # Semantic structure for LLM
                'scroll_position': scroll_info.get('scrollY', 0),
                'max_scroll': scroll_info.get('maxScrollY', 0),
                'scroll_percent': scroll_info.get('scrollPercent', 100),
                'viewport_height': scroll_info.get('innerHeight', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            return {'url': page.url, 'title': '', 'body_text': '', 'elements': [], 'element_count': 0, 'a11y_tree': ''}

    async def _get_accessibility_tree(self, page: Page) -> str:
        """Get accessibility tree as formatted string for LLM context"""
        try:
            snapshot = await page.accessibility.snapshot()
            if not snapshot:
                return "(No accessibility tree available)"
            
            # Format tree recursively (limit depth to avoid explosion)
            # Track index for correlation with elements list
            element_idx = [0]  # Use list to allow modification in nested function
            
            def format_node(node: Dict, depth: int = 0, max_depth: int = 5) -> List[str]:
                if depth > max_depth:
                    return []
                
                lines = []
                role = node.get('role', 'unknown')
                name = node.get('name', '')
                
                # Extract state
                state = []
                if node.get('checked'): 
                    state.append('[checked]' if node.get('checked') is True else '[mixed]')
                if node.get('expanded'): state.append('[expanded]')
                if node.get('disabled'): state.append('[disabled]')
                if node.get('selected'): state.append('[selected]')
                
                state_str = " ".join(state)
                
                # Skip empty/hidden nodes
                if role in ['none', 'generic'] and not name:
                    pass
                elif name or role in ['button', 'link', 'textbox', 'searchbox', 'combobox', 'heading', 'menuitem', 'tab', 'checkbox', 'radio']:
                    indent = "  " * depth
                    # Include index for interactive elements to help with correlation
                    if role in ['button', 'link', 'textbox', 'searchbox', 'combobox', 'menuitem', 'checkbox', 'radio']:
                        element_idx[0] += 1
                        lines.append(f"{indent}#{element_idx[0]} {state_str} [{role}] \"{name}\"".strip())
                    else:
                        lines.append(f"{indent}{state_str} [{role}] {name}".strip())
                
                # Recurse into children
                for child in node.get('children', []):
                    lines.extend(format_node(child, depth + 1, max_depth))
                
                return lines
            
            tree_lines = format_node(snapshot)
            return "\n".join(tree_lines)  # Full a11y tree
            
        except Exception as e:
            logger.warning(f"Failed to get a11y tree: {e}")
            return "(Accessibility tree unavailable)"
    
    async def _get_interactive_elements(self, page: Page) -> List[Dict[str, Any]]:
        """Get interactive elements with robust Shadow DOM traversal"""
        try:
            elements = await page.evaluate('''() => {
                const results = [];
                const viewportCenterY = window.scrollY + (window.innerHeight / 2);
                const processedNodes = new Set();

                // Helper: Check if element is visible and in viewport
                function isVisible(el) {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    
                    // 1. Basic Visibility Check
                    if (rect.width === 0 || rect.height === 0 || 
                        style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') {
                        return false;
                    }
                    
                    // 2. Strict Viewport Intersection Check
                    // Must overlap with the visible window area
                    const inViewport = (
                        rect.bottom > 0 &&
                        rect.right > 0 &&
                        rect.top < window.innerHeight &&
                        rect.left < window.innerWidth
                    );
                    
                    return inViewport;
                }

                // Helper: Check if element is interactive
                function isInteractive(el, style) {
                    const tag = el.tagName.toLowerCase();
                    // Native interactive elements
                    if (['a', 'button', 'select', 'textarea', 'input', 'details', 'summary'].includes(tag)) return true;
                    // ARIA interactive roles
                    const role = el.getAttribute('role');
                    if (['button', 'link', 'menuitem', 'option', 'switch', 'checkbox', 'radio', 'tab', 'treeitem', 'gridcell'].includes(role)) return true;
                    // Event indicators (heuristics)
                    if (el.getAttribute('onclick') || el.getAttribute('data-action') || el.getAttribute('data-testid')) return true;
                    if (el.tabIndex >= 0) return true;
                    if (style.cursor === 'pointer') return true;
                    return false;
                }

                // Helper: Generate robust XPath (Shadow DOM aware)
                function getXPath(el) {
                    // 1. ID (best)
                    if (el.id) return `//*[@id="${el.id}"]`;
                    
                    // 2. Test ID
                    const testId = el.getAttribute('data-testid');
                    if (testId) return `//*[@data-testid="${testId}"]`;
                    
                    // 3. ARIA Label
                    const ariaLabel = el.getAttribute('aria-label');
                    if (ariaLabel) return `//${el.tagName.toLowerCase()}[@aria-label="${ariaLabel}"]`;
                    
                    // 4. Text Content (Filtered)
                    const text = el.innerText ? el.innerText.trim().substring(0, 50) : '';
                    if (text && text.length > 2) {
                        const escaped = text.replace(/"/g, "'");
                        return `//${el.tagName.toLowerCase()}[contains(text(), "${escaped}")]`;
                    }
                    
                    // 5. Placeholder
                    if (el.placeholder) return `//input[@placeholder="${el.placeholder}"]`;
                    
                    return ''; // XPath less critical if we have coordinates
                }

                // Helper: Recursive DOM Walker
                function walk(root, depth=0) {
                    if (depth > 20) return; // Prevent infinite recursion

                    const children = root.children || [];
                    for (let el of children) {
                        if (processedNodes.has(el)) continue;
                        processedNodes.add(el);

                        // 1. Analyze accessibility & style
                        const style = window.getComputedStyle(el);
                        if (!isVisible(el)) {
                            // If element is NOT visible/in-viewport, we usually skip it.
                            // BUT: its children might be visible (e.g. if parent is large container).
                            // Optimization: Check if the element *intersects* viewport at all. 
                            // Our isVisible() already checks intersection. If false, it's completely off-screen.
                            // So we can safely skip recursion for performance!
                            continue; 
                        } else {
                            if (isInteractive(el, style)) {
                                const rect = el.getBoundingClientRect();
                                const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim();
                                
                                // Only keep if valid
                                if (text || el.tagName === 'INPUT' || style.cursor === 'pointer') {
                                    
                                    // Extract State
                                    let state = '';
                                    if (el.checked || el.getAttribute('aria-checked') === 'true') state = '[checked]';
                                    if (el.expanded || el.getAttribute('aria-expanded') === 'true') state = '[expanded]';
                                    if (el.disabled || el.getAttribute('aria-disabled') === 'true') state = '[disabled]';
                                    
                                    // Add to results
                                    const absoluteY = rect.y + window.scrollY;
                                    results.push({
                                        role: el.getAttribute('role') || el.tagName.toLowerCase(),
                                        name: (state + ' ' + text).trim() || '(no-text)',
                                        xpath: getXPath(el),
                                        x: Math.round(rect.x + rect.width / 2),
                                        y: Math.round(absoluteY + rect.height / 2),
                                        dist: Math.abs(absoluteY - viewportCenterY),
                                        attributes: {
                                            type: el.type,
                                            placeholder: el.placeholder,
                                            required: el.required
                                        }
                                    });
                                }
                            }
                        }

                        // 2. Recurse into Shadow DOM
                        if (el.shadowRoot) {
                            walk(el.shadowRoot, depth + 1);
                        }
                        
                        // 3. Recurse into children (with null safety)
                        if (el.children && el.children.length > 0) {
                            walk(el, depth + 1);
                        }
                    }
                }

                // Start parsing from Body (with null safety for page navigation)
                if (document.body) {
                    walk(document.body);
                }
                
                // Sort by distance from center (User Experience heuristic)
                results.sort((a, b) => a.dist - b.dist);
                
                // Deduplicate by location (approximate) to avoid overlay clutter
                const uniqueResults = [];
                const seenLocs = new Set();
                
                for (let res of results) {
                    const locKey = `${res.x},${res.y}`;
                    if (!seenLocs.has(locKey)) {
                        seenLocs.add(locKey);
                        uniqueResults.push(res);
                    }
                }

                return uniqueResults;
            }''')
            return elements
        except Exception as e:
            logger.error(f"Failed to get elements: {e}")
            return []
    
    async def find_element(self, page: Page, selector: str):
        """Find element by selector (CSS or XPath) with fallbacks"""
        try:
            # Try XPath first (starts with //)
            if selector.startswith('//'):
                element = page.locator(f"xpath={selector}").first
                if await element.count() > 0:
                    return element
            
            # Try CSS selector
            element = page.locator(selector).first
            if await element.count() > 0:
                return element
            
            # Try as text
            element = page.locator(f'text="{selector}"').first
            if await element.count() > 0:
                return element
            
            # Try partial text match
            element = page.get_by_text(selector, exact=False).first
            if await element.count() > 0:
                return element
                
            return None
        except Exception as e:
            logger.error(f"Failed to find element '{selector}': {e}")
            return None
    
    async def extract_text(self, page: Page) -> str:
        """Extract all visible text"""
        try:
            return await page.evaluate('() => document.body.innerText')
        except:
            return ""

