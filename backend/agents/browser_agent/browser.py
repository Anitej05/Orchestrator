"""
Browser Agent - Browser Control

Enhanced Playwright browser control with:
- Stealth mode (anti-bot detection evasion)
- Auth persistence (session storage)
- Smart wait strategies
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Literal
from playwright.async_api import async_playwright, Browser as PWBrowser, Page, BrowserContext

from .config import CONFIG

logger = logging.getLogger(__name__)


# Stealth scripts to inject
STEALTH_SCRIPTS = """
// Mask webdriver property
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined
});

// Override plugins to look more realistic
Object.defineProperty(navigator, 'plugins', {
    get: () => [
        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
        { name: 'Native Client', filename: 'internal-nacl-plugin', description: '' }
    ]
});

// Override languages
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en']
});

// Override permissions query
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
);

// Chrome runtime mock
window.chrome = {
    runtime: {},
    loadTimes: function() {},
    csi: function() {},
    app: {}
};
"""


class Browser:
    """Enhanced Playwright browser wrapper with stealth and persistence"""
    
    def __init__(self, profile_name: str = "default"):
        self.playwright = None
        self.browser: Optional[PWBrowser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.profile_name = profile_name
        self._stealth_enabled = True
        self._download_handler: Optional[Callable] = None
    
    @property
    def profile_dir(self) -> Path:
        """Get profile storage directory"""
        return CONFIG.STORAGE_ROOT.parent / "profiles" / self.profile_name
    
    @property
    def storage_state_path(self) -> Path:
        """Get storage state file path"""
        return self.profile_dir / "storage_state.json"
    
    async def launch(
        self, 
        headless: bool = False, 
        on_download: Optional[Callable] = None,
        stealth: bool = True,
        restore_session: bool = True
    ) -> bool:
        """Launch browser with stealth mode and optional session restoration
        
        Args:
            headless: Run in headless mode
            on_download: Callback for download events
            stealth: Enable anti-bot detection evasion
            restore_session: Restore cookies/storage from previous session
        """
        self._stealth_enabled = stealth
        self._download_handler = on_download
        
        try:
            self.playwright = await async_playwright().start()
            
            # Stealth launch args
            launch_args = [
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-infobars',
                '--window-size=1280,800',
                '--disable-extensions',
            ]
            
            if stealth:
                launch_args.extend([
                    '--disable-blink-features=AutomationControlled',
                    '--exclude-switches=enable-automation',
                ])
            
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=launch_args
            )
            
            # Context options with realistic fingerprint
            context_options = {
                'viewport': {'width': 1280, 'height': 800},
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'accept_downloads': True,
                'locale': 'en-US',
                'timezone_id': 'America/New_York',
                'color_scheme': 'light',
                'has_touch': False,
                'is_mobile': False,
                'java_script_enabled': True,
            }
            
            # Restore session if available
            if restore_session and self.storage_state_path.exists():
                try:
                    context_options['storage_state'] = str(self.storage_state_path)
                    logger.info(f"🔐 Restoring session from {self.storage_state_path}")
                except Exception as e:
                    logger.warning(f"Failed to restore session: {e}")
            
            self.context = await self.browser.new_context(**context_options)
            
            # Set default timeouts
            self.context.set_default_navigation_timeout(CONFIG.NAVIGATION_TIMEOUT)
            self.context.set_default_timeout(CONFIG.CLICK_TIMEOUT)
            
            # Auto-switch to new pages (handling new tabs)
            self.context.on("page", self._handle_new_page)
            
            self.page = await self.context.new_page()
            
            # Inject stealth scripts before any navigation
            if stealth:
                await self._apply_stealth(self.page)
            
            # Setup download handler
            if on_download:
                self.page.on("download", on_download)
                
            logger.info(f"🌐 Browser launched (stealth={stealth}, session={'restored' if restore_session and self.storage_state_path.exists() else 'fresh'})")
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to launch browser: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def _apply_stealth(self, page: Page):
        """Apply stealth scripts to a page"""
        try:
            await page.add_init_script(STEALTH_SCRIPTS)
            logger.debug("🥷 Stealth scripts applied")
        except Exception as e:
            logger.warning(f"Failed to apply stealth: {e}")

    async def _handle_new_page(self, page: Page):
        """Handle new page/tab creation with proper stabilization"""
        try:
            logger.info(f"🌐 New tab detected! switching context...")
            
            # Apply stealth to new page
            if self._stealth_enabled:
                await self._apply_stealth(page)
            
            # Wait for DOM content first
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            
            # Then wait briefly for network to settle (with short timeout to not block)
            try:
                await page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                pass  # Network didn't settle, but that's okay
            
            await page.bring_to_front()
            self.page = page
            
            # Setup download handler on new page too
            if self._download_handler:
                page.on("download", self._download_handler)
                
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
                for p in reversed(pages):
                    if not p.is_closed():
                        return p
            return self.page
        except Exception:
            return self.page
    
    async def navigate(self, url: str, timeout: int = None) -> bool:
        """Navigate to URL with smart waiting"""
        if timeout is None:
            timeout = CONFIG.NAVIGATION_TIMEOUT
            
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            await self.page.goto(url, wait_until='domcontentloaded', timeout=timeout)
            
            # Smart wait: try networkidle briefly, but don't block too long
            try:
                await self.page.wait_for_load_state('networkidle', timeout=5000)
            except Exception:
                pass  # Page is interactive even if network isn't idle
            
            logger.info(f"✅ Navigated to: {url}")
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    async def wait_for_element(
        self, 
        selector: str, 
        state: Literal["visible", "hidden", "attached", "detached"] = "visible",
        timeout: int = 10000
    ) -> bool:
        """Smart wait for element with specific state"""
        try:
            await self.page.wait_for_selector(selector, state=state, timeout=timeout)
            return True
        except Exception:
            return False
    
    async def wait_for_network_idle(self, timeout: int = 5000) -> bool:
        """Wait for network to become idle"""
        try:
            await self.page.wait_for_load_state('networkidle', timeout=timeout)
            return True
        except Exception:
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
        except Exception:
            return ""
    
    async def save_session(self) -> bool:
        """Save current session (cookies, localStorage) for later restoration"""
        if not self.context:
            return False
            
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            await self.context.storage_state(path=str(self.storage_state_path))
            logger.info(f"💾 Session saved to {self.storage_state_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    async def clear_session(self) -> bool:
        """Clear saved session"""
        try:
            if self.storage_state_path.exists():
                self.storage_state_path.unlink()
                logger.info("🗑️ Session cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False
    
    async def close(self, save_session: bool = True):
        """Cleanup browser resources gracefully
        
        Args:
            save_session: Whether to save session state before closing
        """
        try:
            # Save session before closing
            if save_session and self.context:
                await self.save_session()
            
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
                await asyncio.sleep(0.1)
                await self.playwright.stop()
                self.playwright = None
            logger.info("🔒 Browser closed")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")

