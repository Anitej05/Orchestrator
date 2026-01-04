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
        self._browser_pids: set = set()  # Track PIDs for safe cleanup
    
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
            
            # Launch args for STABILITY and stealth
            # These prevent Windows "closed pipe" errors and browser crashes
            launch_args = [
                # Stealth
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                
                # STABILITY - Prevent crashes (critical for Windows)
                '--disable-gpu',  # GPU can cause crashes
                '--disable-gpu-sandbox',
                '--disable-software-rasterizer',
                '--disable-dev-shm-usage',  # Crucial for stability
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-extensions',
                '--disable-background-networking',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI',  # Disable translate popups
                '--disable-ipc-flooding-protection',
                # NOTE: --single-process removed - it causes crashes!
                
                # POPUP & TAB PREVENTION
                '--block-new-web-contents',  # Block popups from creating new tabs
                '--disable-popup-blocking',  # Let us handle popups ourselves
                '--disable-features=Prerender2',  # Disable prerendering (creates hidden tabs)
                '--disable-features=NetworkService',  # Reduces background page creation
                
                # Window
                '--window-size=1280,800',
            ]
            
            if stealth:
                launch_args.extend([
                    '--exclude-switches=enable-automation',
                ])
            
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=launch_args,
                handle_sigint=False,  # Let Python handle signals
                handle_sigterm=False,
                handle_sighup=False,
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
            
            self.page = await self.context.new_page()
            
            # Inject stealth scripts before any navigation
            if stealth:
                await self._apply_stealth(self.page)
            
            # Auto-switch to new pages (handling new tabs) - wrap async handler properly
            # MOVED: Register listener AFTER creating main page to prevent 'handle_new_page'
            # from killing the startup page (race condition fix)
            self.context.on("page", lambda page: asyncio.create_task(self._handle_new_page(page)))
            
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
        """Handle new page/tab creation - auto-close ads/popups, only switch if needed.
        
        We don't blindly switch to every new page because:
        1. Ads/popups create new pages we don't want
        2. About:blank pages are placeholders
        3. Switching away from a working page breaks the agent
        """
        try:
            new_url = page.url  # Get URL immediately, don't wait
            
            # FAST PATH: Close blank pages immediately (common ad pattern)
            # CRITICAL FIX: Don't close if it's the ONLY page (startup race condition)
            # or if it's the main page we just created
            is_startup_page = (self.page == page)
            if (new_url in ['about:blank', ''] or not new_url.startswith('http')) and not is_startup_page:
                logger.info(f"🔕 Auto-closing blank popup tab")
                try:
                    await page.close()
                except Exception:
                    pass
                return
            
            # Wait briefly for URL to stabilize
            try:
                await page.wait_for_load_state("commit", timeout=1000)  # Much faster!
            except Exception:
                pass
            
            new_url = page.url  # Re-check after commit
            
            # Apply stealth regardless (in case we do switch later)
            if self._stealth_enabled:
                await self._apply_stealth(page)
            
            # Check if current page is still usable
            current_ok = False
            if self.page:
                try:
                    if not self.page.is_closed() and self.page.url and self.page.url.startswith('http'):
                        current_ok = True
                except Exception:
                    pass
            
            if current_ok:
                # Current page is fine - check if this is likely an ad and close it
                is_likely_ad = any(kw in new_url.lower() for kw in [
                    'ad', 'click', 'track', 'redirect', 'banner', 'advert', 'popup',
                    'doubleclick', 'googlesyndication', 'facebook.com/tr', 'analytics'
                ])
                
                if is_likely_ad:
                    logger.info(f"🗑️ Auto-closing popup/ad tab: {new_url[:50]}...")
                    try:
                        await page.close()
                    except Exception:
                        pass
                    return
                
                # Not an ad, but still don't want to switch
                logger.info(f"📑 New tab opened (keeping current focus): {new_url[:50]}")
                if self._download_handler:
                    page.on("download", self._download_handler)
                return
            
            # Current page is dead/closed - switch to new page
            logger.info(f"🔄 Switching to new tab (current page unavailable): {new_url[:60]}")
            await page.bring_to_front()
            self.page = page
            
            if self._download_handler:
                page.on("download", self._download_handler)
                
        except Exception as e:
            logger.debug(f"New page handling skipped: {e}")
    
    def get_active_page(self) -> Optional[Page]:
        """Get the currently active page, checking all open tabs.
        CRITICAL: Also updates self.page to keep the reference fresh.
        Returns None if no usable page found (caller should recover).
        """
        if not self.context:
            try:
                if self.page and not self.page.is_closed() and self.page.url:
                    return self.page
            except Exception:
                pass
            return None
            
        try:
            pages = self.context.pages
            if pages:
                for p in reversed(pages):
                    try:
                        if not p.is_closed() and p.url:
                            # Auto-update self.page to this valid page
                            if self.page != p:
                                self.page = p
                                logger.debug(f"📌 Updated active page reference to: {p.url[:50]}")
                            return p
                    except Exception:
                        continue
            
            # Fallback to self.page
            try:
                if self.page and not self.page.is_closed():
                    return self.page
            except Exception:
                pass
                
            return None
        except Exception:
            return None
    
    async def _kill_orphaned_browsers(self):
        """Kill only the Chromium processes we launched (not user's Chrome)."""
        import os
        import signal
        
        if not self._browser_pids:
            return
        
        killed = []
        for pid in list(self._browser_pids):
            try:
                os.kill(pid, 0)  # Check if alive
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
                self._browser_pids.discard(pid)
            except (OSError, ProcessLookupError):
                self._browser_pids.discard(pid)
            except Exception as e:
                logger.debug(f"Could not kill PID {pid}: {e}")
        
        if killed:
            logger.info(f"🧹 Killed orphaned browser PIDs: {killed}")
    
    async def recover_page(self, target_url: str = None) -> Optional[Page]:
        """Create a new page when all existing pages are stale.
        If context is also dead, recreates the context.
        If browser is dead, restarts the browser.
        Optionally navigates to target_url if provided.
        """
        new_page = None
        
        # Try 1: Create page from existing context
        if self.context:
            try:
                new_page = await self.context.new_page()
                if self._stealth_enabled:
                    await self._apply_stealth(new_page)
                if self._download_handler:
                    new_page.on("download", self._download_handler)
                self.page = new_page
                logger.info("✅ New page created successfully")
            except Exception as e:
                logger.warning(f"⚠️ Context is dead ({e}), recreating context...")
                new_page = None
        
        # Try 2: Recreate context from existing browser
        if not new_page and self.browser:
            try:
                logger.info("🔄 Recreating browser context...")
                context_options = {
                    'viewport': {'width': 1280, 'height': 800},
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'accept_downloads': True,
                }
                
                # Try to restore session if available
                if self.storage_state_path.exists():
                    try:
                        context_options['storage_state'] = str(self.storage_state_path)
                    except Exception:
                        pass
                
                self.context = await self.browser.new_context(**context_options)
                self.context.on("page", lambda page: asyncio.create_task(self._handle_new_page(page)))
                
                new_page = await self.context.new_page()
                if self._stealth_enabled:
                    await self._apply_stealth(new_page)
                if self._download_handler:
                    new_page.on("download", self._download_handler)
                self.page = new_page
                logger.info("✅ New page created successfully")
            except Exception as e:
                logger.warning(f"⚠️ Browser is dead ({e}), restarting browser...")
                new_page = None
        
        # Try 3: Full browser restart
        if not new_page:
            new_page = await self._restart_browser(target_url)
            # NOTE: Do NOT return early here - let navigation happen below!
        
        # Navigate to target URL if provided and page is ready
        # This is CRITICAL for recovery - restores the last known URL
        if new_page and target_url and target_url.startswith('http'):
            try:
                logger.info(f"🔄 Restoring URL after recovery: {target_url[:60]}...")
                await new_page.goto(target_url, wait_until='domcontentloaded', timeout=30000)
                await asyncio.sleep(2)  # Extra wait for page to fully load
                logger.info(f"✅ Recovered and navigated to: {target_url}")
            except Exception as nav_err:
                logger.warning(f"Post-recovery navigation failed: {nav_err}")
        
        return new_page
    
    async def _restart_browser(self, target_url: str = None) -> Optional[Page]:
        """Restart the browser instance when it crashes."""
        try:
            logger.info("🔄 BROWSER RESTART - relaunching browser instance...")
            
            # 1. Clean up old browser/context gracefully
            try:
                if self.context:
                    await self.context.close()
                if self.browser:
                    await self.browser.close()
            except Exception:
                pass
            
            # 2. Force-kill any orphaned Chromium processes we launched
            await self._kill_orphaned_browsers()
            
            # 3. Relaunch browser with STABILITY flags (same as initial launch)
            launch_args = [
                # Stealth
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                
                # STABILITY - Prevent crashes (critical for Windows)
                '--disable-gpu',
                '--disable-gpu-sandbox',
                '--disable-software-rasterizer',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-extensions',
                '--disable-background-networking',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI',
                '--disable-ipc-flooding-protection',
                # NOTE: --single-process removed - it causes crashes!
                
                # POPUP & TAB PREVENTION
                '--block-new-web-contents',  # Block popups from creating new tabs
                '--disable-popup-blocking',  # Let us handle popups ourselves
                '--disable-features=Prerender2',  # Disable prerendering (creates hidden tabs)
                '--disable-features=NetworkService',  # Reduces background page creation
                
                # Window
                '--window-size=1280,800',
            ]
            
            if self._stealth_enabled:
                launch_args.extend([
                    '--exclude-switches=enable-automation',
                ])
            
            if not self.playwright:
                self.playwright = await async_playwright().start()

            self.browser = await self.playwright.chromium.launch(
                headless=False,
                args=launch_args,
                handle_sigint=False,
                handle_sigterm=False,
                handle_sighup=False,
            )
            
            # Track the new browser's PID
            try:
                if hasattr(self.browser, '_impl_obj') and hasattr(self.browser._impl_obj, '_process'):
                    pid = self.browser._impl_obj._process.pid
                    self._browser_pids.add(pid)
                    logger.debug(f"📝 Tracking new browser PID: {pid}")
            except Exception:
                pass
            
            # Create new context
            context_options = {
                'viewport': {'width': 1280, 'height': 800},
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'accept_downloads': True,
            }
            
            # Restore session if available
            if self.storage_state_path.exists():
                try:
                    context_options['storage_state'] = str(self.storage_state_path)
                except Exception:
                    pass
            
            self.context = await self.browser.new_context(**context_options)
            self.context.on("page", lambda page: asyncio.create_task(self._handle_new_page(page)))
            
            new_page = await self.context.new_page()
            if self._stealth_enabled:
                await self._apply_stealth(new_page)
            if self._download_handler:
                new_page.on("download", self._download_handler)
            
            self.page = new_page
            
            # NOTE: Do NOT auto-navigate here - let the agent control navigation
            logger.info("✅ Browser fully restarted (blank page ready)")
            
            return new_page
            
        except Exception as e:
            logger.error(f"❌ Browser restart failed: {e}")
            return None

    
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

