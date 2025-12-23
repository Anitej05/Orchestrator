"""
Browser Agent - Browser Control

Simple, reliable Playwright browser control.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright, Browser as PWBrowser, Page, BrowserContext

logger = logging.getLogger(__name__)


class Browser:
    """Simple Playwright browser wrapper"""
    
    def __init__(self):
        self.playwright = None
        self.browser: Optional[PWBrowser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
    
    async def launch(self, headless: bool = False, on_download=None) -> bool:
        """Launch browser with optional download handler"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=['--disable-blink-features=AutomationControlled']
            )
            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                accept_downloads=True
            )
            # Set default timeouts to 60s
            self.context.set_default_navigation_timeout(60000)
            self.context.set_default_timeout(60000)
            
            # Auto-switch to new pages (handling new tabs)
            self.context.on("page", self._handle_new_page)
            
            self.page = await self.context.new_page()
            
            # Setup download handler
            if on_download:
                self.page.on("download", on_download)
                
            logger.info("🌐 Browser launched")
            return True
        except Exception as e:
            import traceback
            logger.error(f"Failed to launch browser: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def _handle_new_page(self, page: Page):
        """Handle new page/tab creation with proper stabilization"""
        try:
            logger.info(f"🌐 New tab detected! switching context...")
            # Wait for DOM content first
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            # Then wait briefly for network to settle (with short timeout to not block)
            try:
                await page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                pass  # Network didn't settle, but that's okay
            await page.bring_to_front()
            self.page = page
            logger.info(f"✅ Switched to new tab: {page.url}")
        except Exception as e:
            logger.warning(f"Failed to switch to new page: {e}")
    
    def get_active_page(self) -> Optional[Page]:
        """Get the currently active page, checking all open tabs"""
        if not self.context:
            return self.page
        try:
            pages = self.context.pages
            if pages:
                # Return the last page (most recently opened)
                # Or the one that's focused if we can detect it
                for p in reversed(pages):
                    if not p.is_closed():
                        return p
            return self.page
        except Exception:
            return self.page
    
    async def navigate(self, url: str, timeout: int = 60000) -> bool:
        """Navigate to URL"""
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            await self.page.goto(url, wait_until='domcontentloaded', timeout=timeout)
            await self.page.wait_for_timeout(1000)  # Brief wait for dynamic content
            logger.info(f"✅ Navigated to: {url}")
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    async def screenshot(self, timeout: int = 10000) -> Optional[bytes]:
        """Capture screenshot with explicit timeout. Uses JPEG for speed.
        
        Includes comprehensive debugging and retry logic.
        """
        import time as time_module
        start_time = time_module.time()
        
        # Pre-flight checks
        if not self.page:
            logger.warning(f"📸 SCREENSHOT DEBUG: self.page is None")
            return None
        
        if self.page.is_closed():
            logger.warning(f"📸 SCREENSHOT DEBUG: page is closed")
            return None
            
        try:
            current_url = self.page.url
            logger.info(f"📸 SCREENSHOT DEBUG: Attempting screenshot on URL: {current_url[:60]}...")
        except Exception as e:
            logger.warning(f"📸 SCREENSHOT DEBUG: Cannot get page URL: {e}")
            return None
        
        # Try multiple strategies
        strategies = [
            {"type": "jpeg", "quality": 70, "full_page": False, "timeout": timeout, "name": "JPEG viewport"},
            {"type": "jpeg", "quality": 50, "full_page": False, "timeout": timeout // 2, "name": "JPEG low-q fast"},
            {"type": "png", "full_page": False, "timeout": timeout, "name": "PNG viewport"},
        ]
        
        for strategy in strategies:
            try:
                logger.info(f"📸 Trying strategy: {strategy['name']} (timeout: {strategy.get('timeout', timeout)}ms)")
                
                if strategy["type"] == "jpeg":
                    result = await self.page.screenshot(
                        type='jpeg', 
                        quality=strategy["quality"], 
                        full_page=strategy["full_page"],
                        timeout=strategy.get("timeout", timeout)
                    )
                else:
                    result = await self.page.screenshot(
                        type='png',
                        full_page=strategy["full_page"],
                        timeout=strategy.get("timeout", timeout)
                    )
                
                elapsed = time_module.time() - start_time
                logger.info(f"📸 ✅ Screenshot SUCCESS with {strategy['name']} in {elapsed:.2f}s, size: {len(result)} bytes")
                return result
                
            except Exception as e:
                elapsed = time_module.time() - start_time
                logger.warning(f"📸 Strategy '{strategy['name']}' failed after {elapsed:.2f}s: {str(e)[:50]}")
                continue
        
        elapsed = time_module.time() - start_time
        logger.error(f"📸 ❌ ALL screenshot strategies FAILED after {elapsed:.2f}s")
        return None
    
    async def get_url(self) -> str:
        """Get current URL"""
        return self.page.url if self.page else ""
    
    async def get_title(self) -> str:
        """Get page title"""
        try:
            return await self.page.title()
        except:
            return ""
    
    async def close(self):
        """Cleanup browser resources gracefully to avoid asyncio pipe warnings"""
        try:
            if self.page:
                try:
                    await self.page.close()
                except Exception:
                    pass
                self.page = None
            if self.context:
                try:
                    await self.context.close()
                except Exception:
                    pass
                self.context = None
            if self.browser:
                try:
                    await self.browser.close()
                except Exception:
                    pass
                self.browser = None
            if self.playwright:
                # Allow subprocess pipes to flush before stopping
                import asyncio
                await asyncio.sleep(0.1)
                await self.playwright.stop()
                self.playwright = None
            logger.info("🔒 Browser closed")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
