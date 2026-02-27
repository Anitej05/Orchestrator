"""
Browser Agent - DOM Extraction (SOTA)

Extract page content with:
- Accessibility Tree (semantic structure)
- Robust XPath generation (reliable selectors)
- Coordinates (vision fallback)
"""

import logging
import asyncio
from typing import List, Dict, Any, Set
from playwright.async_api import Page

logger = logging.getLogger(__name__)


class DOMExtractor:
    """Extract DOM content with a11y tree, robust XPath, and iFrame support"""
    
    # Configuration
    MAX_IFRAME_DEPTH = 3  # Maximum depth for nested iframes
    MAX_IFRAMES = 3       # Reduced from 10 - skip most ad/tracking iframes
    MAX_ELEMENTS = 500    # Matches 32k prompt budget — viewport-priority scoring ensures important elements first
    
    async def get_page_content(self, page: Page) -> Dict[str, Any]:
        """Get comprehensive page content for LLM, including iFrame contents"""
        try:
            url = page.url
            title = await page.title()
            
            # PHASE 1: Detect JS click listeners via CDP (catches React/Vue/Angular handlers)
            js_click_backend_ids = await self._detect_js_click_listeners(page)
            
            # Get visible text (FULL TEXT - No 200k limit for CMS RAG)
            body_text = await page.evaluate('''() => {
                if (!document.body) return '';
                return document.body.innerText; 
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
            
            # Get all frames (main + iframes)
            all_frames = await self._get_accessible_frames(page)
            
            # Get interactive elements from ALL frames
            all_elements = []
            frame_info = []
            
            for frame_data in all_frames:
                frame = frame_data['frame']
                frame_id = frame_data['id']
                frame_name = frame_data['name']
                is_main = frame_data['is_main']
                
                try:
                    elements = await self._get_interactive_elements_from_frame(
                        frame, 
                        frame_id=frame_id,
                        frame_name=frame_name,
                        js_click_backend_ids=js_click_backend_ids
                    )
                    all_elements.extend(elements)
                    
                    if not is_main:
                        frame_info.append({
                            'id': frame_id,
                            'name': frame_name,
                            'url': frame.url,
                            'element_count': len(elements)
                        })
                except Exception as frame_err:
                    logger.warning(f"Failed to extract from frame {frame_id}: {frame_err}")
            
            # Viewport-priority scoring: sort elements by importance, then cap
            viewport_height = scroll_info.get('innerHeight', 1000)
            scroll_y = scroll_info.get('scrollY', 0)
            all_elements = self._prioritize_elements(all_elements, viewport_height, scroll_y)
            
            if len(all_elements) > self.MAX_ELEMENTS:
                logger.info(f"🚫 Element limit reached ({len(all_elements)} → {self.MAX_ELEMENTS} after viewport-priority scoring)")
                all_elements = all_elements[:self.MAX_ELEMENTS]
            
            # PHASE 2: Build unified selector_map — ONLY interactive elements NOT excluded by parent
            # This single map feeds BOTH text representation AND visual highlights AND click actions
            # Bounding box propagation: children inside <a>, <button>, etc. are excluded
            selector_map = []
            excluded_count = 0
            for el in all_elements:
                if el.get('interactive', False):
                    if el.get('excludedByParent', False):
                        excluded_count += 1
                    else:
                        selector_map.append(el)
            
            if excluded_count > 0:
                logger.info(f"📦 BBox propagation: excluded {excluded_count} children inside propagating parents")
            
            # DEBUG: Log extracted elements
            logger.info(f"🔍 DOM Extracted {len(all_elements)} elements, {len(selector_map)} interactive (selector_map) from {len(all_frames)} frames")
            if all_elements:
                for i, el in enumerate(all_elements[:5]):
                    frame_tag = f"[{el.get('frame_id', 'main')}] " if el.get('frame_id') != 'main' else ""
                    logger.info(f"  [{i+1}] {frame_tag}{el.get('role', '?')}: '{el.get('name', '')[:30]}' → {el.get('xpath', 'NO XPATH')}")
            
            # Get accessibility tree (semantic structure) - safe-guarded
            try:
                # Limit timeout to prevent pipe hangs/crashes on massive pages
                # Use unified tree builder directly with correct context dict
                tree_context = {
                    'elements': all_elements,
                    'viewport_height': scroll_info.get('innerHeight', 1000),
                    'max_scroll': scroll_info.get('maxScrollY', 0)
                }
                a11y_tree = await asyncio.wait_for(self.build_unified_page_tree(page, tree_context, mode='text'), timeout=5.0)
            except Exception as e:
                logger.warning(f"A11y tree extraction failed or timed out: {e}")
                a11y_tree = ""
            
            # Detect overlays, modals, popups
            overlay_info = await self._detect_overlays(page)
            
            # Discover working selectors for LLM (NEW)
            from .selector_discovery import get_selector_discovery
            discovery = get_selector_discovery()
            try:
                selector_hints = await discovery.discover_patterns(page)
            except Exception as disc_err:
                logger.warning(f"Selector discovery failed: {disc_err}")
                selector_hints = {}
            
            # Create page observation summary for memory
            observation_summary = self._create_observation_summary(all_elements, overlay_info, title)
            
            return {
                'url': url,
                'title': title,
                'body_text': self._clean_text(body_text),
                'elements': all_elements,  # ALL elements (interactive + content)
                'selector_map': selector_map,  # ONLY interactive elements — feeds text list AND visual highlights
                'element_count': len(all_elements),
                'a11y_tree': a11y_tree,  # Semantic structure for LLM
                'scroll_position': scroll_info.get('scrollY', 0),
                'max_scroll': scroll_info.get('maxScrollY', 0),
                'scroll_percent': scroll_info.get('scrollPercent', 100),
                'viewport_height': scroll_info.get('innerHeight', 0),
                'viewport_width': scroll_info.get('innerWidth', 1280),
                'frames': frame_info,  # Info about iframes found
                'overlays': overlay_info,  # Detected modals/popups
                'observation_summary': observation_summary,  # Key observations for memory
                'selector_hints': selector_hints  # Discovered selector patterns
            }
        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            return {'url': page.url, 'title': '', 'body_text': '', 'elements': [], 'selector_map': [], 'element_count': 0, 'a11y_tree': '', 'frames': []}
    
    async def _detect_js_click_listeners(self, page: Page) -> Set[str]:
        """
        Detect elements with JavaScript click event listeners using CDP.
        
        This catches framework-attached handlers (React onClick, Vue @click, Angular (click))
        that are invisible to regular DOM attribute checks.
        
        Uses DevTools' getEventListeners() API via CDP, then marks elements with a
        data attribute so the main JS extraction can detect them.
        
        Returns:
            Set of marker IDs (strings) for elements with JS click listeners.
            Empty set if detection fails (graceful degradation).
        """
        try:
            cdp = await page.context.new_cdp_session(page)
            try:
                # Use Runtime.evaluate with includeCommandLineAPI to access getEventListeners()
                result = await cdp.send('Runtime.evaluate', {
                    'expression': '''
                    (() => {
                        if (typeof getEventListeners !== 'function') return [];
                        
                        const markers = [];
                        let counter = 0;
                        const allElements = document.querySelectorAll('*');
                        
                        for (const el of allElements) {
                            try {
                                const listeners = getEventListeners(el);
                                // Check for click-related event listeners
                                if (listeners.click || listeners.mousedown || listeners.mouseup || 
                                    listeners.pointerdown || listeners.pointerup) {
                                    // Mark with data attribute so JS extraction can find it
                                    const markerId = 'cdp_' + counter++;
                                    el.setAttribute('data-bu-click', markerId);
                                    markers.push(markerId);
                                }
                            } catch (e) {
                                // Ignore cross-origin elements
                            }
                        }
                        
                        return markers;
                    })()
                    ''',
                    'includeCommandLineAPI': True,
                    'returnByValue': True,
                })
                
                markers = result.get('result', {}).get('value', [])
                if markers:
                    logger.info(f"🎯 CDP detected {len(markers)} elements with JS click listeners")
                return set(markers) if markers else set()
                
            finally:
                await cdp.detach()
                
        except Exception as e:
            logger.debug(f"CDP click listener detection failed (non-critical): {e}")
            return set()
    
    async def _get_accessible_frames(self, page: Page, max_depth: int = None) -> List[Dict[str, Any]]:
        """
        Get all accessible frames (main + iframes) for element extraction.
        
        Returns list of dicts with:
            - frame: The Frame object
            - id: Unique identifier for the frame
            - name: Frame name/title
            - is_main: Whether this is the main frame
            - depth: Nesting depth (0 = main)
        """
        if max_depth is None:
            max_depth = self.MAX_IFRAME_DEPTH
            
        frames = []
        processed_urls = set()  # Avoid duplicates
        
        # Add main frame
        frames.append({
            'frame': page.main_frame,
            'id': 'main',
            'name': 'Main Frame',
            'is_main': True,
            'depth': 0
        })
        processed_urls.add(page.main_frame.url)
        
        # Get all child frames
        try:
            all_page_frames = page.frames
            iframe_count = 0
            
            for frame in all_page_frames:
                # Skip main frame (already added)
                if frame == page.main_frame:
                    continue
                    
                # Skip if we've hit the limit
                if iframe_count >= self.MAX_IFRAMES:
                    logger.debug(f"Reached max iframe limit ({self.MAX_IFRAMES})")
                    break
                
                # Skip duplicate URLs (some frameworks create multiple frame handles)
                if frame.url in processed_urls:
                    continue
                processed_urls.add(frame.url)
                
                # Skip about:blank frames (usually placeholders)
                if frame.url == 'about:blank':
                    continue
                
                # Skip ad/tracking iframes by URL pattern
                ad_patterns = ['doubleclick', 'googlesyndication', 'googleadservices', 
                               'facebook.com/tr', 'analytics', 'tracking', 'adserver', 
                               'adsystem', 'advertising', 'criteo', 'outbrain', 'taboola',
                               'amazon-adsystem', 'pubmatic', 'rubiconproject']
                frame_url_lower = frame.url.lower()
                if any(pattern in frame_url_lower for pattern in ad_patterns):
                    logger.debug(f"Skipping ad/tracking iframe: {frame.url[:60]}")
                    continue
                
                # Check if frame is detached
                try:
                    # Quick check to see if frame is accessible
                    await frame.evaluate('() => true')
                except Exception:
                    logger.debug(f"Skipping detached frame: {frame.url[:50]}")
                    continue
                
                iframe_count += 1
                frame_id = f"iframe_{iframe_count}"
                frame_name = frame.name or f"IFrame {iframe_count}"
                
                frames.append({
                    'frame': frame,
                    'id': frame_id,
                    'name': frame_name,
                    'is_main': False,
                    'depth': 1  # Playwright flattens frame hierarchy
                })
                
                logger.debug(f"📝 Found iframe: {frame_id} - {frame.url[:60]}")
                
        except Exception as e:
            logger.warning(f"Error getting child frames: {e}")
        
        return frames


    def _clean_text(self, text: str) -> str:
        """Remove Private Use Area characters (icons) and clean text"""
        if not text:
            return ""
        # Remove Private Use Area characters (E000-F8FF) commonly used for icons
        return "".join(c for c in text if not (0xE000 <= ord(c) <= 0xF8FF)).strip()
    
    def _prioritize_elements(self, elements: list, viewport_height: int, scroll_y: int) -> list:
        """Score and sort elements by viewport visibility and interactivity.
        
        Ensures viewport-visible elements always appear first in the list,
        with off-screen elements getting lower priority. This replaces the
        old flat cap that could miss important in-viewport elements.
        
        Scoring:
            +100: Fully within current viewport
            +50:  Partially visible in viewport
            +30:  Interactive element (button, link, input, etc.)
            +20:  Has meaningful text content
            -20:  Completely off-screen (also tagged with offscreen=True)
        """
        viewport_top = scroll_y
        viewport_bottom = scroll_y + viewport_height
        
        INTERACTIVE_ROLES = {
            'button', 'link', 'textbox', 'input', 'checkbox', 'radio',
            'combobox', 'menuitem', 'option', 'switch', 'tab', 'select',
            'searchbox', 'spinbutton', 'slider', 'a'
        }
        
        for el in elements:
            score = 0
            y = el.get('y', 0)
            h = el.get('height', 0)
            el_top = y
            el_bottom = y + h
            
            # Viewport visibility scoring
            if viewport_top <= el_top and el_bottom <= viewport_bottom:
                score += 100  # Fully visible
            elif el_top < viewport_bottom and el_bottom > viewport_top:
                score += 50   # Partially visible
            else:
                score -= 20   # Off-screen
                el['offscreen'] = True
            
            # Interactivity bonus
            role = el.get('role', '').lower()
            tag = el.get('tag', '').lower()
            if role in INTERACTIVE_ROLES or tag in INTERACTIVE_ROLES:
                score += 30
            
            # Has meaningful content
            if el.get('name', '').strip():
                score += 20
            
            el['_priority_score'] = score
        
        # Sort by priority score descending (viewport elements first)
        elements.sort(key=lambda e: e.get('_priority_score', 0), reverse=True)
        return elements
    
    async def build_unified_page_tree(
        self,
        page: Any,
        page_content: Dict[str, Any],
        mode: str = 'text',
        selector_hints: Dict[str, Any] = None
    ) -> str:
        """
        Build a flat, token-efficient DOM representation combining:
        - Interactive elements with clickable indices `[N]`
        - Semantic content summary block
        """
        try:
            # Use selector_map (interactive-only) for the [N] indexed elements
            # This ensures text indices match visual highlight indices exactly
            elements = page_content.get('selector_map', page_content.get('elements', []))
            
            # --- STRUCTURE 1: INTERACTIVE ELEMENTS ---
            lines = ["[Interactive Elements]"]
            
            # Pre-process semantic hints
            semantic_map = {}
            if selector_hints:
                content_sels = selector_hints.get('contentSelectors', {})
                for item in content_sels.get('titles', []):
                    semantic_map[item['selector'].replace('.', '')] = "TITLE"
                for item in content_sels.get('prices', []):
                    semantic_map[item['selector'].replace('.', '')] = "PRICE"
                    
            for idx, el in enumerate(elements):
                role = el.get('role', 'element')
                name = self._clean_text(el.get('name', ''))[:50]
                
                # Check semantic tags
                sem_tag = ""
                el_classes = el.get('attributes', {}).get('class', '').split()
                for cls in el_classes:
                    if cls in semantic_map:
                        sem_tag = f" [{semantic_map[cls]}]"
                        break
                        
                # Format attributes
                attrs = ""
                if role in ('textbox', 'searchbox', 'input') and el.get('attributes', {}).get('placeholder'):
                    attrs += f" placeholder=\"{el['attributes']['placeholder'][:30]}\""
                if el.get('checked'):
                    attrs += " checked=true"
                if el.get('expanded') is True:
                    attrs += " expanded=true"
                
                # Spatial Tags (keep our advantage)
                y = el.get('y', 0)
                if y < 150: attrs += " [TOP]"
                elif y > 2000 and y > (page_content.get('max_scroll', 0) - 500): attrs += " [BOTTOM]"
                
                if el.get('sticky'): attrs += " [STICKY]"
                if el.get('modal'): attrs += " [MODAL]"
                
                if name:
                    lines.append(f"[{idx+1}] {role} \"{name}\"{attrs}{sem_tag}")
                else:
                    lines.append(f"[{idx+1}] {role}{attrs}{sem_tag}")
                    
            # --- STRUCTURE 2: CONTENT SUMMARY ---
            lines.append("\n[Page Content Summary]")
            
            title = page_content.get('title', '')
            if title:
                lines.append(f"Title: {title}")
                
            # --- STRUCTURE 3: SMART ANCHORS (TABLE OF CONTENTS) ---
            anchors = []
            for el in elements:
                href = el.get('attributes', {}).get('href', '')
                # Only include valid anchors that have a name/text
                if href and href.startswith('#') and len(href) > 1:
                    name = self._clean_text(el.get('name', '')).split(']')[-1].strip() # Strip spatial tags from name
                    if name and name.lower() not in ['top', 'back to top', 'skip to content']:
                        anchors.append(f"'{name}' ({href})")
            
            if anchors:
                # Deduplicate while preserving order
                seen = set()
                unique_anchors = [x for x in anchors if not (x in seen or seen.add(x))]
                lines.append("Table of Contents / Jump Links Available (Click these instead of scrolling!):")
                lines.append("  " + " | ".join(unique_anchors[:15])) # Limit to 15 to save tokens
                
            scroll = page_content.get('scroll_info', {})
            if scroll:
                pct = scroll.get('scroll_percent', 0)
                y = scroll.get('y', 0)
                max_y = scroll.get('max_scroll', 0)
                lines.append(f"Scroll: {pct:.0f}% ({y}px / {max_y}px)")
                
            # Visible text preview
            body_text = page_content.get('body_text', '')
            if body_text:
                import re
                clean_text = re.sub(r'\s+', ' ', body_text).strip()
                if clean_text:
                    lines.append("Visible Text (Preview):")
                    # Send up to 800 chars of actual content text
                    lines.append(f"\"{clean_text[:800]}...\"")
            
            # Truncate lines if too long (MAX_ELEMENTS is already applied during JS extraction)
            MAX_LINES = 300
            if len(lines) > MAX_LINES:
                lines = lines[:MAX_LINES] + [f"... ({len(elements)} total elements, use scroll to see more)"]
                
            return "\n".join(lines)
            
        except Exception as e:
            logger.warning(f"Failed to build unified tree: {e}")
            return self._format_elements_fallback(elements, mode)
    
    def _get_group_icon(self, role: str) -> str:
        """Get icon for group/container roles."""
        icons = {
            'navigation': '🧭',
            'banner': '🔝',
            'main': '📄',
            'complementary': '📎',
            'contentinfo': '📋',
            'form': '📝',
            'search': '🔍',
            'dialog': '💬',
            'menu': '☰',
            'menubar': '☰',
            'toolbar': '🔧',
            'list': '📑',
            'article': '📰',
            'section': '§',
            'region': '▢',
        }
        return icons.get(role, '📦')
    
    def _format_elements_fallback(self, elements: List[Dict], mode: str) -> str:
        """Fallback format when a11y tree is unavailable."""
        lines = ["[Interactive Elements]"]
        
        for idx, el in enumerate(elements[:200]):
            role = el.get('role', 'element')
            name = el.get('name', '')[:50]
            if name:
                lines.append(f"[{idx+1}] {role} \"{name}\"")
            else:
                lines.append(f"[{idx+1}] {role}")
                
        lines.append("\n[Page Content Summary]")
        lines.append("A11y tree unavailable. Elements extracted via JS only.")
        return "\n".join(lines)
    
    async def _get_interactive_elements_from_frame(
        self, 
        frame, 
        frame_id: str = 'main',
        frame_name: str = 'Main Frame',
        js_click_backend_ids: Set[str] = None
    ) -> List[Dict[str, Any]]:
        """Get interactive elements from a specific frame with robust Shadow DOM traversal"""
        try:
            # Pass frame_id to JS so it can be added to each element
            elements = await frame.evaluate('''(frameId) => {
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
                    
                    // 2. Viewport Intersection Check with Buffer
                    // REVERTED to 50px (original safe value) to fix "Closed Pipe" crash
                    const buffer = 50;
                    const inViewport = (
                        rect.bottom > -buffer &&
                        rect.right > -buffer &&
                        rect.top < window.innerHeight + buffer &&
                        rect.left < window.innerWidth + buffer
                    );
                    
                    return inViewport;
                }

                // Helper: Check if element is interactive (COMPREHENSIVE Heuristic)
                // Goal: NEVER miss a clickable element - prefer false positives over false negatives
                function isInteractive(el, style, rect) {
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role');
                    
                    // 1. Native interactive elements - ALWAYS interactive
                    const nativeInteractive = ['a', 'button', 'select', 'textarea', 'input', 'details', 'summary', 'option', 'label', 'area', 'audio', 'video', 'embed', 'object', 'iframe'];
                    if (nativeInteractive.includes(tag)) {
                        return true;
                    }
                    
                    // 1b. CDP-detected JS click listeners (React onClick, Vue @click, Angular (click))
                    // These are marked by the _detect_js_click_listeners() method before this call
                    if (el.hasAttribute('data-bu-click')) {
                        return true;
                    }
                    
                    // 2. ARIA interactive roles - ALWAYS interactive
                    const interactiveRoles = ['button', 'link', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'switch', 'checkbox', 'radio', 'tab', 'treeitem', 'option', 'slider', 'spinbutton', 'combobox', 'searchbox', 'textbox', 'listbox', 'menu', 'tree', 'grid', 'gridcell', 'row', 'dialog', 'alertdialog', 'progressbar', 'scrollbar', 'tooltip', 'application'];
                    if (interactiveRoles.includes(role)) {
                        return true;
                    }
                    
                    // 3. Contenteditable elements - editable = interactive
                    if (el.isContentEditable || el.getAttribute('contenteditable') === 'true') {
                        return true;
                    }
                    
                    // 4. HTML event handler attributes - DEFINITELY interactive
                    const eventAttrs = ['onclick', 'onmousedown', 'onmouseup', 'ontouchstart', 'ontouchend', 'ondblclick', 'onkeydown', 'onkeyup', 'onkeypress', 'onfocus', 'onblur'];
                    for (const attr of eventAttrs) {
                        if (el.hasAttribute(attr)) return true;
                    }
                    
                    // 5. JavaScript-attached onclick handler (catches React, Vue, etc.)
                    // This is the KEY check for modern frameworks
                    if (typeof el.onclick === 'function') {
                        return true;
                    }
                    
                    // 6. Framework-specific click attributes (Angular, Vue, Alpine, HTMX, etc.)
                    const frameworkAttrs = ['ng-click', '@click', 'v-on:click', 'data-action', 'data-onclick', 'x-on:click', 'hx-get', 'hx-post', 'hx-trigger', 'wire:click'];
                    for (const attr of frameworkAttrs) {
                        if (el.hasAttribute(attr)) return true;
                    }
                    
                    // 7. Data attributes that often indicate interactivity
                    const interactiveDataAttrs = ['data-toggle', 'data-dismiss', 'data-target', 'data-slide', 'data-bs-toggle', 'data-bs-dismiss', 'data-fancybox', 'data-lightbox', 'data-modal', 'data-popup'];
                    for (const attr of interactiveDataAttrs) {
                        if (el.hasAttribute(attr)) return true;
                    }
                    
                    // 8. Focusable elements (positive tabindex)
                    if (el.tabIndex > 0) {
                        return true;
                    }
                    
                    // 9. Elements with tabindex=0 - explicitly made focusable = likely interactive
                    if (el.tabIndex === 0) {
                        return true;
                    }
                    
                    // 10. Cursor pointer check - RELAXED but with size constraint
                    // Accept cursor:pointer for elements with reasonable clickable size (min 15x15 pixels)
                    if (style.cursor === 'pointer') {
                        const hasReasonableSize = rect && rect.width >= 15 && rect.height >= 15;
                        if (hasReasonableSize) {
                            return true;
                        }
                    }
                    
                    // 11. Elements with aria-* attributes suggesting interactivity
                    const interactiveAria = ['aria-expanded', 'aria-pressed', 'aria-haspopup', 'aria-controls', 'aria-owns', 'aria-activedescendant', 'aria-selected', 'aria-checked'];
                    for (const attr of interactiveAria) {
                        if (el.hasAttribute(attr)) return true;
                    }
                    
                    // 12. Draggable elements
                    if (el.draggable === true || el.getAttribute('draggable') === 'true') {
                        return true;
                    }
                    
                    // 13. SVG interactive elements (if they have title or are focusable)
                    if (['svg', 'path', 'g', 'circle', 'rect'].includes(tag)) {
                        // SVG is interactive if parent is link/button (handled by parent aggregation)
                        // or if it has explicit interaction attributes
                        if (el.tabIndex >= 0 || el.hasAttribute('onclick') || typeof el.onclick === 'function') {
                            return true;
                        }
                        // For SVG, rely on parent aggregation (checked separately)
                        return false;
                    }
                    
                    // 14. List items in menu/listbox context
                    if (tag === 'li') {
                        const parent = el.parentElement;
                        if (parent) {
                            const parentRole = parent.getAttribute('role');
                            if (['menu', 'listbox', 'tree', 'tablist'].includes(parentRole)) {
                                return true;
                            }
                        }
                    }
                    
                    return false;
                }
                
                // Helper: Find the nearest interactive parent element
                // Used to aggregate child elements into their clickable container
                function getInteractiveParent(el, maxLevels = 3) {
                    let current = el.parentElement;
                    let level = 0;
                    
                    while (current && level < maxLevels) {
                        // Check if this parent is natively interactive
                        const tag = current.tagName.toLowerCase();
                        if (['a', 'button', 'select', 'textarea', 'input', 'details', 'summary', 'label'].includes(tag)) {
                            return current;
                        }
                        
                        // Check for interactive role
                        const role = current.getAttribute('role');
                        if (['button', 'link', 'menuitem', 'switch', 'checkbox', 'radio', 'tab', 'option'].includes(role)) {
                            return current;
                        }
                        
                        // Check for explicit click handler
                        if (current.hasAttribute('onclick') || current.hasAttribute('ng-click') || current.hasAttribute('@click') || current.hasAttribute('data-action')) {
                            return current;
                        }
                        
                        current = current.parentElement;
                        level++;
                    }
                    
                    return null; // No interactive parent found
                }
                
                // BOUNDING BOX PROPAGATION FILTER (Browser-Use inspired)
                // If an interactive child is >=95% contained within a "propagating" parent
                // (a, button, div[role=button], etc.), the child gets excluded.
                // This collapses product cards: <a> contains <img>, <span>, etc.
                // Only the parent <a> enters the selector_map.
                function isExcludedByParent(el) {
                    // EXCEPTION RULES — NEVER exclude these:
                    const tag = el.tagName.toLowerCase();
                    // 1. Form elements need individual interaction
                    if (['input', 'select', 'textarea', 'label'].includes(tag)) return false;
                    // 2. Elements with explicit onclick or CDP-detected click
                    if (el.hasAttribute('onclick') || el.hasAttribute('data-bu-click')) return false;
                    // 3. Elements with aria-label (independently interactive)
                    if (el.getAttribute('aria-label')) return false;
                    // 4. Elements with interactive ARIA roles
                    const role = el.getAttribute('role');
                    if (['button', 'link', 'checkbox', 'radio', 'tab', 'menuitem', 'option', 'switch', 'combobox'].includes(role)) return false;
                    
                    // PROPAGATING PARENT TAGS: these "own" their children's clicks
                    const propagatingTags = ['a', 'button'];
                    const propagatingRoles = ['button', 'link', 'combobox'];
                    
                    // Walk up looking for a propagating parent
                    let parent = el.parentElement;
                    let levels = 0;
                    while (parent && levels < 5) {
                        const parentTag = parent.tagName.toLowerCase();
                        const parentRole = parent.getAttribute('role');
                        
                        const isPropagating = propagatingTags.includes(parentTag) ||
                            (parentRole && propagatingRoles.includes(parentRole));
                        
                        if (isPropagating) {
                            // Check bounding box containment (>=95%)
                            const childRect = el.getBoundingClientRect();
                            const parentRect = parent.getBoundingClientRect();
                            
                            if (childRect.width > 0 && childRect.height > 0) {
                                const xOverlap = Math.max(0, 
                                    Math.min(childRect.right, parentRect.right) - 
                                    Math.max(childRect.left, parentRect.left));
                                const yOverlap = Math.max(0, 
                                    Math.min(childRect.bottom, parentRect.bottom) - 
                                    Math.max(childRect.top, parentRect.top));
                                const intersection = xOverlap * yOverlap;
                                const childArea = childRect.width * childRect.height;
                                const containment = intersection / childArea;
                                
                                if (containment >= 0.95) {
                                    return true; // Child is contained → exclude from selector_map
                                }
                            }
                        }
                        
                        parent = parent.parentElement;
                        levels++;
                    }
                    
                    return false; // No propagating parent found → keep
                }

                // Helper: Generate robust XPath (Shadow DOM aware)
                function getXPath(el) {
                    if (el.id) return `//*[@id="${el.id}"]`;
                    
                    const testId = el.getAttribute('data-testid') || el.getAttribute('data-test-id') || el.getAttribute('data-qa');
                    if (testId) return `//*[@data-testid="${testId}"]`;
                    
                    const ariaLabel = el.getAttribute('aria-label');
                    if (ariaLabel) return `//${el.tagName.toLowerCase()}[@aria-label="${ariaLabel}"]`;
                    
                    // Text fallback (careful with escaping)
                    const text = el.innerText ? el.innerText.trim().substring(0, 50) : '';
                    if (text && text.length > 2) {
                        const escaped = text.replace(/"/g, "'");
                        return `//${el.tagName.toLowerCase()}[contains(text(), "${escaped}")]`;
                    }
                    
                    if (el.placeholder) return `//input[@placeholder="${el.placeholder}"]`;
                    if (el.name) return `//input[@name="${el.name}"]`;
                    
                    return ''; 
                }

                // Helper: Recursive DOM Walker with heading context
                let currentHeading = '';  // Track the most recent heading we've seen
                const markedParents = new Set(); // Track already-marked interactive parents
                
                function walk(root, depth=0) {
                    if (depth > 20) return; // Prevent infinite recursion

                    const children = root.children || [];
                    for (let el of children) {
                        if (processedNodes.has(el)) continue;
                        processedNodes.add(el);
                        
                        // Update section context
                        if (/^h[1-6]$/i.test(el.tagName)) {
                            const headingText = (el.innerText || '').trim().substring(0, 80);
                            if (headingText) currentHeading = headingText;
                        }

                        const style = window.getComputedStyle(el);
                        
                        // Recurse unless hidden
                        if (style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0') {
                            
                            if (isVisible(el)) {
                                // Extract RICH attributes
                                const rect = el.getBoundingClientRect();
                                const tag = el.tagName.toLowerCase();
                                const isClickable = isInteractive(el, style, rect);
                                
                                // REVISED LOGIC: Include ALL visible elements with text
                                // Parent aggregation ONLY marks the parent as interactive, but we STILL process children for text
                                let targetEl = el;
                                let targetRect = rect;
                                let targetTag = tag;
                                let targetIsClickable = isClickable;
                                
                                // If this element is not interactive, check if parent is
                                // But DON'T skip children - just mark their interactivity status
                                if (!isClickable) {
                                    const interactiveParent = getInteractiveParent(el);
                                    if (interactiveParent && !markedParents.has(interactiveParent)) {
                                        // Mark the parent as the click target
                                        markedParents.add(interactiveParent);
                                        // Add the parent SEPARATELY if not already processed
                                        const parentRect = interactiveParent.getBoundingClientRect();
                                        const parentTag = interactiveParent.tagName.toLowerCase();
                                        const parentText = (interactiveParent.innerText || '').trim().substring(0, 300);
                                        
                                        // Only add parent if visible and has meaningful content
                                        if (parentRect.width > 0 && parentRect.height > 0) {
                                            const absoluteY = parentRect.y + window.scrollY;
                                            results.push({
                                                role: interactiveParent.getAttribute('role') || parentTag,
                                                tag: parentTag,
                                                name: parentText || '(clickable)',
                                                text_content: parentText,
                                                xpath: getXPath(interactiveParent),
                                                x: Math.round(parentRect.x + parentRect.width / 2),
                                                y: Math.round(absoluteY + parentRect.height / 2),
                                                top_left_x: Math.round(parentRect.x),
                                                top_left_y: Math.round(absoluteY),
                                                width: Math.round(parentRect.width),
                                                height: Math.round(parentRect.height),
                                                dist: Math.abs(absoluteY - viewportCenterY),
                                                section: currentHeading,
                                                frame_id: frameId,
                                                interactive: true,
                                                is_parent_click_target: true
                                            });
                                        }
                                    }
                                    // NOTE: We DON'T continue/skip here - we still process the current element
                                }
                                
                                // Clean text extraction from current element (not parent)
                                let text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || el.title || el.alt || '').trim();
                                
                                // SAFETY: Skip huge text (base64, JSON blobs, etc) - truncate to 300 chars
                                if (text.length > 500) {
                                    // Check for base64 or JSON patterns
                                    if (text.includes('base64') || text.startsWith('data:') || text.startsWith('{') || text.startsWith('[')) {
                                        text = '[large data blob filtered]';
                                    } else {
                                        text = text.substring(0, 1000) + '...';
                                    }
                                } else {
                                    text = text.replace(/\\s+/g, ' ').substring(0, 300); // Reverted to 300 for stability
                                }
                                
                                // Capture State
                                const state = [];
                                if (el.checked || el.getAttribute('aria-checked') === 'true') state.push('[checked]');
                                if (el.expanded || el.getAttribute('aria-expanded') === 'true') state.push('[expanded]');
                                if (el.disabled || el.getAttribute('aria-disabled') === 'true') state.push('[disabled]');
                                if (el.selected || el.getAttribute('aria-selected') === 'true') state.push('[selected]');
                                if (el.getAttribute('aria-pressed') === 'true') state.push('[pressed]');
                                
                                // New: Completeness States (Form & Interaction)
                                if (el.required || el.getAttribute('aria-required') === 'true') state.push('[required]');
                                if (el.readOnly || el.getAttribute('aria-readonly') === 'true') state.push('[readonly]');
                                if (el.getAttribute('aria-invalid') === 'true') state.push('[invalid]');
                                
                                // New: Scrollability Check - DISABLED COMPLETELY
                                // User requested removal to prevent crashes
                                // let isScrollable = false;
                                // if (['div', 'ul', ...].includes(tag)) { ... }
                                // if (isScrollable) state.push('[scrollable]');
                                
                                const stateStr = state.join(' ');
                                const displayName = (stateStr + ' ' + text).trim() || '(no-text)';
                                
                                // SMART ELEMENT INCLUSION:
                                // 1. ALWAYS include: interactive elements, inputs, images, headings
                                // 2. ONLY include containers (div/span/p) IF they have meaningful text
                                // 3. Skip technical tags entirely
                                // 4. NEW: Skip viewport-spanning containers (>80% viewport area)
                                
                                const viewportArea = window.innerWidth * window.innerHeight;
                                const elArea = rect.width * rect.height;
                                const isHugeContainer = elArea > viewportArea * 0.80;
                                
                                const alwaysIncludeTags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'button', 'input', 'select', 'textarea', 'img', 'label'];
                                const containerTags = ['div', 'span', 'p', 'li', 'td', 'th', 'strong', 'em', 'b', 'i', 'article', 'section', 'figure', 'figcaption', 'main', 'aside'];
                                const skipElement = ['defs', 'clipPath', 'mask', 'style', 'script', 'noscript', 'meta', 'link', 'head', 'svg', 'path', 'g', 'circle', 'rect', 'line', 'polygon', 'polyline'].includes(tag);
                                
                                // Require minimum 3 chars of text for non-interactive containers
                                const hasMeaningfulText = text && text.length >= 3;
                                
                                const shouldInclude = !skipElement && !isHugeContainer && (
                                    targetIsClickable ||                        // Interactive = always
                                    alwaysIncludeTags.includes(tag) ||          // Headings, buttons, inputs, imgs = always
                                    (containerTags.includes(tag) && hasMeaningfulText)  // Containers ONLY with text
                                );
                                
                                if (shouldInclude) {
                                    const absoluteY = rect.y + window.scrollY;
                                    
                                    // EXTRACT DYNAMIC STATE (New)
                                    const isSticky = style.position === 'sticky' || style.position === 'fixed';
                                    const zIndex = parseInt(style.zIndex) || 0;
                                    const isModal = el.getAttribute('role') === 'dialog' || el.classList.contains('modal') || zIndex > 900;
                                    const isError = style.color === 'rgb(255, 0, 0)' || el.getAttribute('aria-invalid') === 'true';
                                    
                                    results.push({
                                        role: el.getAttribute('role') || tag,
                                        tag: tag,
                                        name: displayName,
                                        text_content: text, // Pure text for filtering
                                        xpath: getXPath(el),
                                        x: Math.round(rect.x + rect.width / 2),
                                        y: Math.round(absoluteY + rect.height / 2),
                                        top_left_x: Math.round(rect.x),
                                        top_left_y: Math.round(absoluteY),
                                        width: Math.round(rect.width),
                                        height: Math.round(rect.height),
                                        dist: Math.abs(absoluteY - viewportCenterY),
                                        section: currentHeading,
                                        frame_id: frameId,
                                        interactive: targetIsClickable,
                                        // Include new dynamic states
                                        sticky: isSticky,
                                        zIndex: zIndex,
                                        modal: isModal,
                                        error: isError,
                                        interactive: targetIsClickable,
                                        excludedByParent: targetIsClickable ? isExcludedByParent(el) : false,
                                        attributes: {
                                            type: el.type,
                                            placeholder: el.placeholder,
                                            href: el.href ? (el.href.length > 100 ? el.href.substring(0, 100) + '...' : el.href) : undefined,
                                            src: el.src ? (el.src.startsWith('data:') ? '[base64 image]' : el.src.substring(0, 100)) : undefined,
                                            alt: el.alt,
                                            title: el.title,
                                            target: el.target,
                                            value: el.value,
                                            class: el.className ? el.className.toString().substring(0, 50) : '',
                                            testId: el.getAttribute('data-testid')
                                        }
                                    });
                                }
                            }
                            
                            // Recurse into Shadow DOM & Children
                            if (el.shadowRoot) walk(el.shadowRoot, depth + 1);
                            if (el.children && el.children.length > 0) walk(el, depth + 1);
                        }
                    }
                }

                if (document.body) walk(document.body);
                
                // Sort by visual logic (top-down, left-to-right) usually better than distance for reading
                results.sort((a, b) => {
                    const diffY = a.y - b.y;
                    if (Math.abs(diffY) > 20) return diffY; // Different lines
                    return a.x - b.x; // Same line (LTR)
                });
                
                // Deduplicate with Interactive Prioritization
                // If two elements have exact same coords and text, keep the INTERACTIVE one.
                const unique = [];
                const seen = new Map(); // Key -> index in 'unique' array
                
                for (let r of results) {
                    const key = `${r.x},${r.y},${r.name}`;
                    if (!seen.has(key)) {
                        seen.set(key, unique.length);
                        unique.push(r);
                    } else {
                        // Collision! Check if we should upgrade to the new element
                        const idx = seen.get(key);
                        const existing = unique[idx];
                        
                        // PRIORITIZE INTERACTIVE ELEMENTS
                        // If new is interactive and existing is not, replace it.
                        if (r.interactive && !existing.interactive) {
                            unique[idx] = r;
                        }
                    }
                }

                return unique;
            }''', frame_id)  # Pass frame_id as argument to JS
            
            # Post-process to clean text on Python side
            for el in elements:
                el['frame_name'] = frame_name
                    
            return elements
        except Exception as e:
            logger.error(f"Failed to get elements from frame {frame_id}: {e}")
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
        except Exception:
            return ""
    
    async def _detect_overlays(self, page: Page) -> Dict[str, Any]:
        """Detect modal overlays, dialogs, popups, and ads"""
        try:
            overlay_data = await page.evaluate('''() => {
                // PRODUCTION-GRADE OVERLAY DETECTION
                // Detect TRUE blocking modals including role="dialog" and common patterns
                
                const viewportWidth = window.innerWidth;
                const viewportHeight = window.innerHeight;
                const overlays = [];
                const closeButtons = [];
                
                // HELPER: Find close buttons in element
                function findCloseButtons(container) {
                    // Query standard CSS selectors (no jQuery-only selectors!)
                    const closeBtns = container.querySelectorAll(
                        '[aria-label*="close" i], [aria-label*="dismiss" i], [aria-label*="cancel" i], ' +
                        'button[class*="close"], [class*="close-btn"], [class*="modal-close"], ' +
                        'button:has(svg), [class*="dismiss"], [data-testid*="close"]'
                    );
                    // Also find buttons by text content (replaces invalid :contains() selectors)
                    const allButtons = container.querySelectorAll('button');
                    const textMatches = [...allButtons].filter(btn => {
                        const text = btn.textContent?.trim().toLowerCase() || '';
                        return text === 'cancel' || text === 'close' || text === '×' || text === 'x' || text === '✕';
                    });
                    const combined = new Set([...closeBtns, ...textMatches]);
                    combined.forEach(btn => {
                        if (btn.offsetParent !== null) {
                            closeButtons.push({ 
                                text: btn.textContent?.trim().substring(0, 20) || btn.getAttribute('aria-label') || 'X',
                                xpath: '//' + btn.tagName.toLowerCase() + '[@class="' + btn.className + '"]'
                            });
                        }
                    });
                }
                
                // METHOD 1: Check for aria-modal="true"
                const ariaModals = document.querySelectorAll('[aria-modal="true"]');
                for (const el of ariaModals) {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    if (rect.width < 100 || rect.height < 100) continue;
                    if (rect.right < 0 || rect.bottom < 0 || rect.left > viewportWidth || rect.top > viewportHeight) continue;
                    
                    overlays.push({
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        type: 'aria-modal',
                        title: el.querySelector('h1, h2, h3, [class*="title"]')?.textContent?.trim()?.substring(0, 50),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    });
                    findCloseButtons(el);
                }
                
                // METHOD 2: Check for role="dialog" or role="alertdialog" (like eBay's modal!)
                const roleDialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"]');
                for (const el of roleDialogs) {
                    // Skip if already found as aria-modal
                    if (el.getAttribute('aria-modal') === 'true') continue;
                    
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    if (rect.width < 100 || rect.height < 100) continue;
                    if (rect.right < 0 || rect.bottom < 0 || rect.left > viewportWidth || rect.top > viewportHeight) continue;
                    
                    overlays.push({
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        type: 'role-dialog',
                        title: el.querySelector('h1, h2, h3, [class*="title"], [class*="heading"]')?.textContent?.trim()?.substring(0, 50),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    });
                    findCloseButtons(el);
                }
                
                // METHOD 3: Common modal class patterns (fallback)
                if (overlays.length === 0) {
                    const commonModals = document.querySelectorAll(
                        '.modal:not(.modal-backdrop), .popup, .dialog, [class*="modal-content"], ' +
                        '[class*="overlay-content"], [class*="lightbox"], [data-testid*="modal"]'
                    );
                    for (const el of commonModals) {
                        const style = window.getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        const zIndex = parseInt(style.zIndex) || 0;
                        
                        if (style.display === 'none' || style.visibility === 'hidden') continue;
                        if (rect.width < 200 || rect.height < 150) continue;  // Must be substantial
                        if (zIndex < 500) continue;  // Must have high z-index (not just stacking context)
                        if (rect.right < 0 || rect.bottom < 0 || rect.left > viewportWidth || rect.top > viewportHeight) continue;
                        
                        // Must cover at least 30% of viewport in BOTH dimensions
                        // This filters out sidebars, nav bars, filter panels, sticky headers
                        const coversWidth = rect.width >= viewportWidth * 0.3;
                        const coversHeight = rect.height >= viewportHeight * 0.3;
                        if (!coversWidth || !coversHeight) continue;
                        
                        // Exclude common non-modal patterns (nav, filters, headers, footers)
                        const cn = (el.className || '').toLowerCase();
                        const id = (el.id || '').toLowerCase();
                        const isNavOrFilter = /\b(nav|header|footer|sidebar|filter|menu|toolbar|banner|sticky|search)\b/.test(cn + ' ' + id);
                        if (isNavOrFilter) continue;
                        
                        overlays.push({
                            tag: el.tagName.toLowerCase(),
                            id: el.id || null,
                            className: el.className?.substring?.(0, 50),
                            type: 'class-modal',
                            title: el.querySelector('h1, h2, h3')?.textContent?.trim()?.substring(0, 50),
                            zIndex: zIndex
                        });
                        findCloseButtons(el);
                    }
                }
                
                // METHOD 2: Check for fullscreen semi-transparent backdrop with VERY high z-index
                // This catches manually-created modals without proper aria attributes
                if (overlays.length === 0) {
                    const allElements = document.querySelectorAll('div');
                    for (const el of allElements) {
                        const style = window.getComputedStyle(el);
                        const zIndex = parseInt(style.zIndex);
                        const rect = el.getBoundingClientRect();
                        const bgColor = style.backgroundColor;
                        
                        // STRICT requirements for backdrop detection:
                        // 1. Must have numeric z-index > 10000 (not NaN, not auto)
                        // 2. Must be fixed or absolute position
                        // 3. Must cover at least 80% of viewport
                        // 4. Must have semi-transparent background (rgba with alpha < 1)
                        const hasHighZIndex = !isNaN(zIndex) && zIndex > 10000;
                        const isPositioned = style.position === 'fixed' || style.position === 'absolute';
                        const coversViewport = rect.width >= viewportWidth * 0.8 && rect.height >= viewportHeight * 0.8;
                        const isSemiTransparent = bgColor.match(/rgba\\([^)]+,\\s*0\\.[0-9]/);
                        
                        if (hasHighZIndex && isPositioned && coversViewport && isSemiTransparent) {
                            overlays.push({
                                tag: 'div',
                                type: 'backdrop',
                                zIndex: zIndex
                            });
                        }
                    }
                }
                
                return {
                    hasOverlay: overlays.length > 0,
                    overlayCount: overlays.length,
                    overlays: overlays,
                    closeButtons: closeButtons.slice(0, 3)
                };
            }''')


            
            if overlay_data.get('hasOverlay'):
                logger.warning(f"🚨 OVERLAY DETECTED: {overlay_data.get('overlayCount')} overlay(s) found!")
                for ob in overlay_data.get('overlays', []):
                    logger.warning(f"   • {ob.get('tag')} id={ob.get('id')} z-index={ob.get('zIndex')}")
            
            return overlay_data
        except Exception as e:
            logger.error(f"Overlay detection failed: {e}")
            return {'hasOverlay': False, 'overlayCount': 0, 'overlays': [], 'closeButtons': []}
    
    
    def _create_observation_summary(self, elements: List[Dict], overlay_info: Dict, title: str) -> str:
        """Create a concise summary of key observations for memory context"""
        observations = []
        
        # Page title
        observations.append(f"Page: {title[:50]}")
        
        # Overlay warning
        if overlay_info.get('hasOverlay'):
            close_btns = overlay_info.get('closeButtons', [])
            if close_btns:
                observations.append(f"⚠️ OVERLAY/POPUP blocking page! Close buttons: {[b['text'] for b in close_btns[:3]]}")
            else:
                observations.append(f"⚠️ OVERLAY/POPUP blocking page (no close button found, try pressing Escape)")
        
        # Key element types found
        # FIX: Check both role AND tag name for precise counts
        buttons = [e for e in elements if e.get('role') == 'button' or e.get('tag') == 'button' or e.get('type') in ['submit', 'button']]
        links = [e for e in elements if e.get('role') == 'link' or e.get('tag') == 'a']
        inputs = [e for e in elements if e.get('tag') == 'input' or e.get('role') in ['textbox', 'searchbox', 'combobox']]
        [e for e in elements if e.get('role') == 'checkbox' or e.get('type') == 'checkbox']
        images = [e for e in elements if e.get('tag') == 'img']
        
        # Detailed input summary
        input_types = {}
        for inp in inputs:
            t = inp.get('attributes', {}).get('type', 'text')
            input_types[t] = input_types.get(t, 0) + 1
            
        input_summary = ", ".join([f"{k}:{v}" for k, v in input_types.items()]) if input_types else "0"
        
        observations.append(f"Found: {len(buttons)} buttons, {len(links)} links, {len(images)} images, {len(inputs)} inputs ({input_summary})")
        
        # Look for filter-related elements
        filter_elements = [e for e in elements if any(kw in str(e.get('name', '')).lower() for kw in ['filter', 'color', 'size', 'sort', 'price', 'brand'])]
        if filter_elements:
            filter_names = [e.get('name', '')[:30] for e in filter_elements[:5]]
            observations.append(f"Filter options: {filter_names}")
        
        return " | ".join(observations)
