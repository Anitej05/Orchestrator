"""
Browser Agent - DOM Extraction (SOTA)

Extract page content with:
- Accessibility Tree (semantic structure)
- Robust XPath generation (reliable selectors)
- Coordinates (vision fallback)
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from playwright.async_api import Page

logger = logging.getLogger(__name__)


class DOMExtractor:
    """Extract DOM content with a11y tree, robust XPath, and iFrame support"""
    
    # Configuration
    MAX_IFRAME_DEPTH = 3  # Maximum depth for nested iframes
    MAX_IFRAMES = 3       # Reduced from 10 - skip most ad/tracking iframes
    MAX_ELEMENTS = 250    # Increased to 250 per user request
    
    async def get_page_content(self, page: Page) -> Dict[str, Any]:
        """Get comprehensive page content for LLM, including iFrame contents"""
        try:
            url = page.url
            title = await page.title()
            
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
                        frame_name=frame_name
                    )
                    all_elements.extend(elements)
                    
                    # Enforce MAX_ELEMENTS limit to prevent prompt bloat
                    if len(all_elements) >= self.MAX_ELEMENTS:
                        all_elements = all_elements[:self.MAX_ELEMENTS]
                        logger.info(f"🚫 Element limit reached ({self.MAX_ELEMENTS}), stopping extraction")
                        break
                    
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
                'elements': all_elements,  # Elements from ALL frames
                'element_count': len(all_elements),
                'a11y_tree': a11y_tree,  # Semantic structure for LLM
                'scroll_position': scroll_info.get('scrollY', 0),
                'max_scroll': scroll_info.get('maxScrollY', 0),
                'scroll_percent': scroll_info.get('scrollPercent', 100),
                'viewport_height': scroll_info.get('innerHeight', 0),
                'frames': frame_info,  # Info about iframes found
                'overlays': overlay_info,  # Detected modals/popups
                'observation_summary': observation_summary,  # Key observations for memory
                'selector_hints': selector_hints  # NEW: Discovered selector patterns
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
    
    async def build_unified_page_tree(
        self, 
        page: Page, 
        page_content: Dict[str, Any],
        mode: str = 'text',
        selector_hints: Dict[str, Any] = None
    ) -> str:
        """
        Build a unified hierarchical page representation combining:
        - Semantic structure from accessibility tree
        - Interactive elements with clickable indices
        - Optimal selectors (XPath, CSS, or index)
        - Discovered semantic roles (Title, Price, etc.)
        
        Args:
            page: Playwright page
            elements: List of extracted interactive elements with indices
            mode: 'text' (full hierarchy) or 'vision' (simplified for visual models)
            selector_hints: Discovered CSS patterns from selector_discovery
        
        Returns:
            Formatted string for LLM prompt
        """
        try:
            elements = page_content.get('elements', [])
            
            snapshot = await page.accessibility.snapshot(interesting_only=False)
            if not snapshot:
                return self._format_elements_fallback(elements, mode)
            
            # --- PRE-PROCESS SELECTOR HINTS ---
            semantic_map = {}  # class/selector -> semantic label (e.g. "TITLE")
            if selector_hints:
                content_sels = selector_hints.get('contentSelectors', {})
                for item in content_sels.get('titles', []):
                    cls = item['selector'].replace('.', '')
                    semantic_map[cls] = "TITLE"
                for item in content_sels.get('prices', []):
                    cls = item['selector'].replace('.', '')
                    semantic_map[cls] = "PRICE"
                for item in content_sels.get('ratings', []):
                    cls = item['selector'].replace('.', '')
                    semantic_map[cls] = "RATING"
            
            # Build element lookup by name for merging with a11y tree
            element_lookup = {}
            for idx, el in enumerate(elements):
                name = el.get('name', '').strip().lower()
                # Also index by role for generic matches if name is empty
                key_name = name if name else f"__role__{el.get('role', '')}"
                
                if key_name not in element_lookup:
                    element_lookup[key_name] = []
                
                # Check for semantic tags from discovery
                semantic_tag = ""
                el_classes = el.get('attributes', {}).get('class', '').split()
                for cls in el_classes:
                    if cls in semantic_map:
                        semantic_tag = semantic_map[cls]
                        break
                
                element_lookup[key_name].append({
                    'index': idx + 1,  # 1-indexed
                    'element': el,
                    'semantic_tag': semantic_tag
                })
            
            # Roles that indicate a logical group/container
            GROUP_ROLES = {
                'navigation', 'banner', 'main', 'complementary', 'contentinfo',
                'article', 'section', 'region', 'form', 'search',
                'dialog', 'tabpanel', 'menu', 'menubar', 'toolbar',
                'list', 'listitem', 'grid', 'row'
            }
            
            # Roles that are interactive/clickable
            INTERACTIVE_ROLES = {
                'button', 'link', 'textbox', 'checkbox', 'radio', 
                'combobox', 'menuitem', 'option', 'switch', 'tab',
                'searchbox', 'spinbutton', 'slider', 'menuitemcheckbox',
                'menuitemradio', 'img'  # Images can be clickable
            }
            
            # Track which elements we've matched
            matched_indices = set()
            lines = []
            
            def get_best_selector(el: Dict) -> str:
                """Get the best selector, prioritizing Index > CSS > XPath."""
                # Note: Index is implicit in the tree format "#N"
                
                # Try to find a simple class selector if it matches
                attributes = el.get('attributes', {})
                el_id = attributes.get('id')
                if el_id:
                    return f" → #{el_id}"
                
                # Fallback to XPath
                xpath = el.get('xpath', '')
                if not xpath or xpath == 'NO XPATH':
                    return ''
                if len(xpath) > 60:
                    return '' 
                return f" → {xpath}"
            
            def format_node(node: Dict, depth: int = 0) -> None:
                """Recursively format node with unified hierarchy."""
                role = node.get('role', '')
                name = self._clean_text(node.get('name', ''))
                children = node.get('children', [])
                
                # Calculate indent (increased depth limit to 20 as requested)
                indent = "│ " * min(depth, 20)
                
                # Try to match with extracted elements
                name_lower = name.strip().lower()
                lookup_key = name_lower if name_lower else f"__role__{role}"
                
                matched_el = None
                matched_idx = None
                matched_semantic = ""
                
                if lookup_key in element_lookup:
                    for match in element_lookup[lookup_key]:
                        if match['index'] not in matched_indices:
                            matched_el = match['element']
                            matched_idx = match['index']
                            matched_semantic = match['semantic_tag']
                            matched_indices.add(matched_idx)
                            break
                
                # Determine node type
                is_group = role in GROUP_ROLES
                is_interactive = role in INTERACTIVE_ROLES or matched_el is not None
                
                # STATE indicators
                states = []
                if node.get('checked') is True: states.append('✓')
                elif node.get('checked') == 'mixed': states.append('◐')
                if node.get('expanded') is True: states.append('▼')
                elif node.get('expanded') is False: states.append('▶')
                if node.get('disabled'): states.append('⊘')
                state_str = ''.join(states)
                
                # BUILD THE LINE
                if is_group and children:
                    # === GROUP CONTAINER ===
                    group_icon = self._get_group_icon(role)
                    label = name[:50] if name else role.upper()
                    lines.append(f"{indent}├── {group_icon} {label}")
                    
                    sibling_counts = {}
                    for child in children:
                        # Smart List Compression
                        c_role = child.get('role', 'element')
                        sibling_counts[c_role] = sibling_counts.get(c_role, 0) + 1
                        if sibling_counts[c_role] > 5:
                            if sibling_counts[c_role] == 6:
                                remaining = len(children) - 5
                                lines.append(f"{indent}│   ... [{remaining} more '{c_role}' items. Use 'run_js' to extract all, or scroll to see more]")
                            continue
                        
                        format_node(child, depth + 1)
                    
                elif is_interactive:
                    # === INTERACTIVE ELEMENT ===
                    if matched_el:
                        # We have an extracted element with index
                        idx_str = f"#{matched_idx}"
                        selector = get_best_selector(matched_el) if mode == 'text' else ''
                        
                        # --- ENHANCED TAGGING ---
                        tags = []
                        if matched_semantic: tags.append(matched_semantic)
                        
                        # Spatial Tags
                        y = matched_el.get('y', 0)
                        vh = page_content.get('viewport_height', 1000)
                        if y < 150: tags.append("TOP")
                        elif y > 2000 and y > (page_content.get('max_scroll', 0) - 500): tags.append("BOTTOM")
                        
                        # State Tags
                        if matched_el.get('sticky'): tags.append("STICKY")
                        if matched_el.get('modal'): tags.append("MODAL")
                        if matched_el.get('error'): tags.append("ERROR")
                        
                        sem_tag = f" [{' '.join(tags)}]" if tags else ""
                        
                        if role == 'link':
                            line = f"{indent}├── {idx_str} 🔗 \"{name[:40]}\"{sem_tag}{selector}"
                        elif role == 'button':
                            line = f"{indent}├── {idx_str} 🔘 \"{name[:40]}\"{sem_tag}{selector}"
                        elif role in ('textbox', 'searchbox'):
                            line = f"{indent}├── {idx_str} 📝 [{name[:30] or 'input'}]{sem_tag}{selector}"
                        elif role in ('img', 'image'):
                            line = f"{indent}├── {idx_str} 🖼️ \"{name[:40]}\"{sem_tag}{selector}"
                        elif role in ('checkbox', 'radio'):
                            line = f"{indent}├── {idx_str} {state_str} {role}: \"{name[:30]}\"{sem_tag}{selector}"
                        elif role == 'combobox':
                            line = f"{indent}├── {idx_str} 📋 \"{name[:30]}\"{sem_tag}{selector}"
                        else:
                            line = f"{indent}├── {idx_str} [{role}] \"{name[:40]}\"{sem_tag}{selector}"
                        
                        lines.append(line)
                    else:
                        # Interactive but not extracted (maybe off-screen)
                        if name and mode == 'text':
                            lines.append(f"{indent}├── [{role}] \"{name[:40]}\" (not in viewport)")
                    
                    # Process children
                    for child in children:
                        format_node(child, depth + 1)
                        
                elif role == 'heading':
                    # === HEADING ===
                    level = node.get('level', 1)
                    if name:
                        lines.append(f"{indent}{'#' * level} {name[:60]}")
                    for child in children:
                        format_node(child, depth)
                        
                elif role in ('text', 'StaticText') and name and len(name.strip()) > 3:
                    # === STATIC TEXT (context) ===
                    if mode == 'text':
                        display = name[:80] + "..." if len(name) > 80 else name
                        lines.append(f'{indent}│ "{display}"')
                        
                elif role == 'image' and name:
                    # === IMAGE (may be clickable) ===
                    lines.append(f"{indent}├── 🖼️ {name[:50]}")
                    
                elif children:
                    # Just process children without adding a line for this node
                    for child in children:
                        format_node(child, depth)
            
            # Process the tree
            format_node(snapshot)
            
            # Add any elements that weren't matched to the tree
            unmatched = []
            for idx, el in enumerate(elements):
                if (idx + 1) not in matched_indices:
                    unmatched.append((idx + 1, el))
            
            if unmatched and mode == 'text':
                lines.append("")
                lines.append("── OTHER ELEMENTS ──")
                for idx, el in unmatched[:30]:  # Limit unmatched
                    role = el.get('role', 'element')
                    name = el.get('name', '')[:40]
                    selector = get_best_selector(el)
                    if name:
                        lines.append(f"  #{idx} [{role}] \"{name}\"{selector}")
                    else:
                        lines.append(f"  #{idx} [{role}]{selector}")
            
            # Truncate if too long
            MAX_LINES = 600 if mode == 'vision' else 800
            if len(lines) > MAX_LINES:
                lines = lines[:MAX_LINES]
                lines.append(f"... ({len(elements) - MAX_LINES} more elements)")
            
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
        lines = ["── INTERACTIVE ELEMENTS ──"]
        
        for idx, el in enumerate(elements[:200]):
            role = el.get('role', 'element')
            name = el.get('name', '')[:50]
            xpath = el.get('xpath', '')
            
            if mode == 'text' and xpath and len(xpath) < 60:
                line = f"#{idx+1} [{role}] \"{name}\" → {xpath}"
            else:
                line = f"#{idx+1} [{role}] \"{name}\"" if name else f"#{idx+1} [{role}]"
            
            lines.append(line)
        
        return "\n".join(lines)
    
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
                                
                                const alwaysIncludeTags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'button', 'input', 'select', 'textarea', 'img', 'label'];
                                const containerTags = ['div', 'span', 'p', 'li', 'td', 'th', 'strong', 'em', 'b', 'i', 'article', 'section', 'figure', 'figcaption', 'main', 'aside'];
                                const skipElement = ['defs', 'clipPath', 'mask', 'style', 'script', 'noscript', 'meta', 'link', 'head', 'svg', 'path', 'g', 'circle', 'rect', 'line', 'polygon', 'polyline'].includes(tag);
                                
                                // Require minimum 3 chars of text for non-interactive containers
                                const hasMeaningfulText = text && text.length >= 3;
                                
                                const shouldInclude = !skipElement && (
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
                    const closeBtns = container.querySelectorAll(
                        '[aria-label*="close" i], [aria-label*="dismiss" i], [aria-label*="cancel" i], ' +
                        'button[class*="close"], [class*="close-btn"], [class*="modal-close"], ' +
                        'button:has(svg), [class*="dismiss"], [data-testid*="close"], ' +
                        'button:contains("Cancel"), button:contains("Close"), button:contains("×")'
                    );
                    closeBtns.forEach(btn => {
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
                        if (rect.width < 150 || rect.height < 100) continue;
                        if (zIndex < 100) continue;  // Must have some z-index
                        if (rect.right < 0 || rect.bottom < 0 || rect.left > viewportWidth || rect.top > viewportHeight) continue;
                        
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
