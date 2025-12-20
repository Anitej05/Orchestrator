"""
Browser Agent - DOM Extraction

Extract page content and find elements.
"""

import logging
from typing import List, Dict, Any, Optional
from playwright.async_api import Page

logger = logging.getLogger(__name__)


class DOMExtractor:
    """Extract DOM content and elements"""
    
    async def get_page_content(self, page: Page) -> Dict[str, Any]:
        """Get comprehensive page content for LLM"""
        try:
            url = page.url
            title = await page.title()
            
            # Get visible text (limited for LLM context)
            body_text = await page.evaluate('''() => {
                return document.body.innerText.substring(0, 3000);
            }''')
            
            # Get interactive elements
            elements = await self._get_interactive_elements(page)
            
            return {
                'url': url,
                'title': title,
                'body_text': body_text,
                'elements': elements[:50],  # Limit to 50 elements
                'element_count': len(elements)
            }
        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            return {'url': page.url, 'title': '', 'body_text': '', 'elements': [], 'element_count': 0}
    
    async def _get_interactive_elements(self, page: Page) -> List[Dict[str, Any]]:
        """Get clickable and interactive elements"""
        try:
            elements = await page.evaluate('''() => {
                const results = [];
                const selectors = 'a, button, input, select, textarea, [role="button"], [onclick]';
                const els = document.querySelectorAll(selectors);
                
                els.forEach((el, idx) => {
                    if (idx >= 100) return;  // Limit
                    
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return;
                    if (rect.top < 0 || rect.top > window.innerHeight) return;
                    
                    const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '').substring(0, 50);
                    if (!text.trim()) return;
                    
                    // Build a reliable selector
                    let selector = '';
                    if (el.id) {
                        selector = '#' + el.id;
                    } else if (el.name) {
                        selector = `[name="${el.name}"]`;
                    } else if (el.className && typeof el.className === 'string') {
                        const classes = el.className.split(' ').filter(c => c && !c.includes(':'))[0];
                        if (classes) selector = '.' + classes;
                    }
                    
                    results.push({
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        text: text.trim(),
                        selector: selector,
                        x: Math.round(rect.x + rect.width / 2),
                        y: Math.round(rect.y + rect.height / 2)
                    });
                });
                
                return results;
            }''')
            return elements
        except Exception as e:
            logger.error(f"Failed to get elements: {e}")
            return []
    
    async def find_element(self, page: Page, selector: str):
        """Find element by selector with fallbacks"""
        try:
            # Try direct selector first
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
