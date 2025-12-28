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
    """Extract DOM content with a11y tree, robust XPath, and iFrame support"""
    
    # Configuration
    MAX_IFRAME_DEPTH = 3  # Maximum depth for nested iframes
    MAX_IFRAMES = 10      # Maximum number of iframes to process
    
    async def get_page_content(self, page: Page) -> Dict[str, Any]:
        """Get comprehensive page content for LLM, including iFrame contents"""
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
                        frame_name=frame_name
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
            
            # DEBUG: Log extracted elements
            logger.info(f"🔍 DOM Extracted {len(all_elements)} elements from {len(all_frames)} frames")
            if all_elements:
                for i, el in enumerate(all_elements[:5]):
                    frame_tag = f"[{el.get('frame_id', 'main')}] " if el.get('frame_id') != 'main' else ""
                    logger.info(f"  [{i+1}] {frame_tag}{el.get('role', '?')}: '{el.get('name', '')[:30]}' → {el.get('xpath', 'NO XPATH')}")
            
            # Get accessibility tree (semantic structure) - main frame only for now
            a11y_tree = await self._get_accessibility_tree(page)
            
            # Detect overlays, modals, popups
            overlay_info = await self._detect_overlays(page)
            
            # Create page observation summary for memory
            observation_summary = self._create_observation_summary(all_elements, overlay_info, title)
            
            return {
                'url': url,
                'title': title,
                'body_text': self._clean_text(body_text),
                'elements': all_elements,  # Elements from ALL frames
                'element_count': len(all_elements),
                'a11y_tree': a11y_tree,  # Semantic structure for LLM
                'scroll_position': scroll_info.get('scrollY', 0),
                'max_scroll': scroll_info.get('maxScrollY', 0),
                'scroll_percent': scroll_info.get('scrollPercent', 100),
                'viewport_height': scroll_info.get('innerHeight', 0),
                'frames': frame_info,  # Info about iframes found
                'overlays': overlay_info,  # Detected modals/popups
                'observation_summary': observation_summary  # Key observations for memory
            }
        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            return {'url': page.url, 'title': '', 'body_text': '', 'elements': [], 'element_count': 0, 'a11y_tree': '', 'frames': []}
    
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

    async def _get_accessibility_tree(self, page: Page) -> str:
        """Get accessibility tree formatted as hierarchical grouped structure.
        
        NOTE: This tree is for CONTEXT ONLY - it shows page structure and grouping.
        The actual clickable indices come from the INTERACTIVE ELEMENTS list.
        """
        try:
            snapshot = await page.accessibility.snapshot(interesting_only=False)  # Show ALL nodes
            if not snapshot:
                return "(No accessibility tree available)"
            
            # Roles that indicate a logical group/container
            GROUP_ROLES = {
                'article', 'listitem', 'group', 'region', 'section', 
                'dialog', 'tabpanel', 'menubar', 'menu', 'toolbar',
                'navigation', 'banner', 'main', 'complementary', 'form',
                'grid', 'row', 'gridcell', 'treegrid', 'treeitem'
            }
            
            # Roles that are interactive/clickable
            INTERACTIVE_ROLES = {
                'button', 'link', 'textbox', 'checkbox', 'radio', 
                'combobox', 'menuitem', 'option', 'switch', 'tab',
                'searchbox', 'spinbutton', 'slider', 'menuitemcheckbox',
                'menuitemradio'
            }
            
            # NOTE: No longer skipping any roles - show ALL to the LLM
            # SKIP_ROLES removed - was {'none', 'generic', 'LineBreak', 'presentation'}
            
            # Track count for summary
            interactive_count = [0]
            
            def format_node(node: Dict, depth: int = 0) -> List[str]:
                """Recursively format node with smart grouping - NO DEPTH LIMIT"""
                role = node.get('role', '')
                name = self._clean_text(node.get('name', ''))
                children = node.get('children', [])
                
                # Process ALL roles - no skipping
                # (Previously skipped noise roles, now showing everything)
                
                lines = []
                indent = "  " * min(depth, 8)  # Cap indent at 8 levels for readability
                
                # Extract state flags
                states = []
                if node.get('checked') is True: states.append('✓')
                elif node.get('checked') == 'mixed': states.append('◐')
                if node.get('expanded') is True: states.append('▼')
                elif node.get('expanded') is False: states.append('▶')
                if node.get('disabled'): states.append('⊘')
                if node.get('selected'): states.append('●')
                state_str = ''.join(states)
                
                # Determine if this is a group container
                is_group = role in GROUP_ROLES
                is_interactive = role in INTERACTIVE_ROLES
                
                # Build the node line
                if is_group and children:
                    # GROUP HEADER - shows what kind of container this is
                    group_label = name[:60] if name else role.upper()
                    lines.append(f"{indent}┌─ {role}: {group_label}")
                    
                    # Process children with increased indent
                    for child in children:
                        lines.extend(format_node(child, depth + 1))
                    
                    lines.append(f"{indent}└─")  # Close group
                    
                elif is_interactive:
                    # CLICKABLE ELEMENT - NO INDEX (use elements list for clicking)
                    interactive_count[0] += 1
                    
                    # Format based on role - descriptive only
                    if role == 'link':
                        display = f'🔗 "{name}"' if name else f'🔗 (link)'
                    elif role == 'button':
                        display = f'🔘 "{name}"' if name else f'🔘 (button)'
                    elif role in ('textbox', 'searchbox'):
                        placeholder = name or 'text input'
                        display = f'📝 [{placeholder}]'
                    elif role in ('checkbox', 'radio'):
                        display = f'{state_str} {role}: "{name}"'
                    elif role == 'combobox':
                        display = f'📋 dropdown: "{name}"'
                    else:
                        display = f'[{role}] "{name}"' if name else f'[{role}]'
                    
                    if state_str and role not in ('checkbox', 'radio'):
                        display += f' {state_str}'
                    
                    lines.append(f"{indent}{display}")
                    
                    # If interactive element has children (rare but possible)
                    for child in children:
                        lines.extend(format_node(child, depth + 1))
                    
                elif role == 'heading':
                    # HEADING - important for context
                    level = node.get('level', '')
                    if name:
                        lines.append(f"{indent}{'#' * (level or 1)} {name}")
                    for child in children:
                        lines.extend(format_node(child, depth))
                    
                elif role in ('text', 'StaticText') and name:
                    # Static text - only include if meaningful (not just whitespace)
                    if len(name.strip()) > 2:
                        # Truncate long text
                        display_text = name[:80] + "..." if len(name) > 80 else name
                        lines.append(f"{indent}\"{display_text}\"")
                    
                elif role == 'image':
                    # Image with alt text
                    if name:
                        lines.append(f"{indent}🖼️ {name[:50]}")
                    
                elif name or children:
                    # Other roles with content
                    if name:
                        lines.append(f"{indent}[{role}] {name[:60]}")
                    
                    for child in children:
                        lines.extend(format_node(child, depth + 1 if name else depth))
                
                else:
                    # Just process children without adding a line
                    for child in children:
                        lines.extend(format_node(child, depth))
                
                return lines
            
            tree_lines = format_node(snapshot)
            
            # Limit output to prevent huge prompts
            MAX_LINES = 500  # Increased from 150 - show more structure
            if len(tree_lines) > MAX_LINES:
                tree_lines = tree_lines[:MAX_LINES]
                tree_lines.append("... (truncated for brevity)")
            
            # Add summary header
            result_lines = [
                f"PAGE STRUCTURE (context only - use INTERACTIVE ELEMENTS for clicking)",
                "─" * 50,
            ]
            result_lines.extend(tree_lines)
            
            return "\n".join(result_lines)
            
        except Exception as e:
            logger.warning(f"Failed to get a11y tree: {e}")
            return "(Accessibility tree unavailable)"
    
    async def _get_interactive_elements_from_frame(
        self, 
        frame, 
        frame_id: str = 'main',
        frame_name: str = 'Main Frame'
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
                    // Capture elements slightly off-screen to handle edges better
                    const buffer = 50;
                    const inViewport = (
                        rect.bottom > -buffer &&
                        rect.right > -buffer &&
                        rect.top < window.innerHeight + buffer &&
                        rect.left < window.innerWidth + buffer
                    );
                    
                    return inViewport;
                }

                // Helper: Check if element is interactive (Enhanced Heuristic)
                function isInteractive(el, style) {
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role');
                    
                    // 1. Native interactive elements
                    if (['a', 'button', 'select', 'textarea', 'input', 'details', 'summary', 'option'].includes(tag)) return true;
                    
                    // 2. ARIA interactive roles
                    if (['button', 'link', 'menuitem', 'switch', 'checkbox', 'radio', 'tab', 'treeitem', 'gridcell', 'slider', 'spinbutton', 'combobox', 'searchbox', 'textbox'].includes(role)) return true;
                    
                    // 3. User interaction indicators
                    if (el.getAttribute('onclick') || el.getAttribute('data-action') || el.getAttribute('data-testid')) return true;
                    
                    // 4. Focusable elements
                    if (el.tabIndex >= 0) return true;
                    
                    // 5. Visual indicators (cursor: pointer)
                    // Be careful: some layout divs have pointer cursor but aren't interactive
                    if (style.cursor === 'pointer') return true;
                    
                    return false;
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
                                const isClickable = isInteractive(el, style);
                                
                                // Clean text extraction
                                let text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || el.title || el.alt || '').trim();
                                text = text.replace(/\s+/g, ' ').substring(0, 150); // Collapse whitespace
                                
                                // Capture State
                                const state = [];
                                if (el.checked || el.getAttribute('aria-checked') === 'true') state.push('[checked]');
                                if (el.expanded || el.getAttribute('aria-expanded') === 'true') state.push('[expanded]');
                                if (el.disabled || el.getAttribute('aria-disabled') === 'true') state.push('[disabled]');
                                if (el.selected || el.getAttribute('aria-selected') === 'true') state.push('[selected]');
                                if (el.getAttribute('aria-pressed') === 'true') state.push('[pressed]');
                                
                                const stateStr = state.join(' ');
                                const displayName = (stateStr + ' ' + text).trim() || '(no-text)';
                                
                                // Only meaningful elements (skip empty divs unless clickable)
                                if (text || isClickable || tag === 'input' || tag === 'img') {
                                    const absoluteY = rect.y + window.scrollY;
                                    
                                    results.push({
                                        role: el.getAttribute('role') || tag,
                                        tag: tag,
                                        name: displayName,
                                        text_content: text, // Pure text for filtering
                                        xpath: getXPath(el),
                                        x: Math.round(rect.x + rect.width / 2),
                                        y: Math.round(absoluteY + rect.height / 2),
                                        dist: Math.abs(absoluteY - viewportCenterY),
                                        section: currentHeading,
                                        frame_id: frameId,
                                        interactive: isClickable,
                                        attributes: {
                                            type: el.type,
                                            placeholder: el.placeholder,
                                            href: el.href,
                                            src: el.src,
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
                
                // Deduplicate strictly
                const unique = [];
                const seen = new Set();
                for (let r of results) {
                    const key = `${r.x},${r.y},${r.name}`;
                    if (!seen.has(key)) {
                        seen.add(key);
                        unique.push(r);
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
                // Only detect TRUE blocking modals, not dropdowns/flyouts/nav elements
                
                const viewportWidth = window.innerWidth;
                const viewportHeight = window.innerHeight;
                const overlays = [];
                const closeButtons = [];
                
                // METHOD 1: Check for aria-modal="true" (the CORRECT way to mark a blocking modal)
                const ariaModals = document.querySelectorAll('[aria-modal="true"]');
                for (const el of ariaModals) {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    
                    // Must be visible AND in viewport AND substantial size
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    if (rect.width < 100 || rect.height < 100) continue;  // Too small = not a modal
                    if (rect.right < 0 || rect.bottom < 0 || rect.left > viewportWidth || rect.top > viewportHeight) continue;
                    
                    // This is a REAL blocking modal
                    overlays.push({
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        type: 'aria-modal',
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    });
                    
                    // Find close buttons
                    const closeBtns = el.querySelectorAll('[aria-label*="close" i], [aria-label*="dismiss" i], button[class*="close"], [class*="close-btn"]');
                    closeBtns.forEach(btn => {
                        if (btn.offsetParent !== null) {
                            closeButtons.push({ text: btn.textContent?.trim().substring(0, 20) || 'X' });
                        }
                    });
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
        checkboxes = [e for e in elements if e.get('role') == 'checkbox' or e.get('type') == 'checkbox']
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
