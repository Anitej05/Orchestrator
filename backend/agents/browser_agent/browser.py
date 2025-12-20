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
            self.page = await self.context.new_page()
            
            # Setup download handler
            if on_download:
                self.page.on("download", on_download)
                
            logger.info("🌐 Browser launched")
            return True
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            return False
    
    async def navigate(self, url: str, timeout: int = 30000) -> bool:
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
    
    async def screenshot(self) -> Optional[bytes]:
        """Capture screenshot"""
        try:
            return await self.page.screenshot(type='png')
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
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
        """Cleanup browser resources"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("🔒 Browser closed")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
