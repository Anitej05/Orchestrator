#!/usr/bin/env python3
"""
Browser Agent Test Suite
Comprehensive tests for the browser automation agent.

Usage:
    cd backend && python tests/test_browser_agent.py

Test Levels:
    1. Module Imports        - All modules import cleanly
    2. Schema Validation     - Pydantic models parse correctly
    3. Config & State        - Config dirs, memory lifecycle
    4. Message Manager       - Token counting, priority truncation
    5. Planner & LLM         - LLMClient init, prompt building
    6. Agent Instantiation   - BrowserAgent constructs without crash
    7. Live Browser E2E      - Full browser automation (requires Playwright)
"""

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add backend AND project root to path
# (some modules import as 'from backend.*' which needs the project root)
backend_dir = Path(__file__).parent.parent
project_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(backend_dir / ".env")
except ImportError:
    pass

# =============================================================================
# HELPERS
# =============================================================================

test_results: List[Dict[str, Any]] = []

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_test(name: str, success: bool, message: str = "", duration: float = 0):
    """Print test result."""
    status = "PASS" if success else "FAIL"
    icon = "✅" if success else "❌"
    duration_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"  {icon} {status} | {name}{duration_str}")
    if message:
        print(f"         {message[:120]}{'...' if len(message) > 120 else ''}")

def print_skip(name: str, reason: str):
    """Print skipped test."""
    print(f"  ⏭️  SKIP | {name}")
    print(f"         {reason}")

def save_result(test_name: str, success: bool, message: str, duration: float, skipped: bool = False):
    """Save test result."""
    test_results.append({
        "test": test_name,
        "success": success,
        "skipped": skipped,
        "message": message,
        "duration": round(duration, 3)
    })


# =============================================================================
# TEST 1: MODULE IMPORTS
# =============================================================================

async def test_module_imports():
    """Test that all browser agent modules import cleanly."""
    print_header("TEST 1: MODULE IMPORTS")
    
    modules = [
        ("agent", "agents.browser_agent.agent", "BrowserAgent"),
        ("actions", "agents.browser_agent.actions", "ActionExecutor"),
        ("browser", "agents.browser_agent.browser", "Browser"),
        ("config", "agents.browser_agent.config", "CONFIG"),
        ("dom", "agents.browser_agent.dom", "DOMExtractor"),
        ("llm", "agents.browser_agent.llm", "LLMClient"),
        ("vision", "agents.browser_agent.vision", "VisionClient"),
        ("state", "agents.browser_agent.state", "AgentMemory"),
        ("message_manager", "agents.browser_agent.message_manager", "MessageManager"),
        ("persistent_memory", "agents.browser_agent.persistent_memory", "PersistentMemory"),
        ("system_prompt", "agents.browser_agent.system_prompt", "BROWSER_AGENT_SYSTEM_PROMPT"),
        ("agent_schemas", "agents.browser_agent.agent_schemas", "ActionPlan"),
    ]
    
    for label, module_path, expected_attr in modules:
        start = time.time()
        try:
            mod = __import__(module_path, fromlist=[expected_attr])
            attr = getattr(mod, expected_attr, None)
            success = attr is not None
            message = f"Imported {expected_attr} from {module_path}" if success else f"Attribute {expected_attr} not found"
            print_test(f"Import {label}", success, message, time.time() - start)
            save_result(f"Import {label}", success, message, time.time() - start)
        except Exception as e:
            print_test(f"Import {label}", False, str(e), time.time() - start)
            save_result(f"Import {label}", False, str(e), time.time() - start)


# =============================================================================
# TEST 2: SCHEMA VALIDATION
# =============================================================================

async def test_schema_validation():
    """Test that Pydantic schemas parse correctly."""
    print_header("TEST 2: SCHEMA VALIDATION")
    
    from agents.browser_agent.agent_schemas import AtomicAction, ActionPlan, BrowserResult
    
    # Test 2a: AtomicAction - basic construction
    start = time.time()
    try:
        action = AtomicAction(name="click", params={"index": 5})
        assert action.name == "click"
        assert action.params["index"] == 5
        print_test("AtomicAction basic", True, f"name={action.name}, params={action.params}", time.time() - start)
        save_result("AtomicAction basic", True, "OK", time.time() - start)
    except Exception as e:
        print_test("AtomicAction basic", False, str(e), time.time() - start)
        save_result("AtomicAction basic", False, str(e), time.time() - start)
    
    # Test 2b: AtomicAction - flat JSON restructuring (model_validator)
    start = time.time()
    try:
        action = AtomicAction.model_validate({"name": "navigate", "url": "https://example.com"})
        assert action.name == "navigate"
        assert action.params.get("url") == "https://example.com"
        print_test("AtomicAction flat JSON", True, f"params restructured: {action.params}", time.time() - start)
        save_result("AtomicAction flat JSON", True, "OK", time.time() - start)
    except Exception as e:
        print_test("AtomicAction flat JSON", False, str(e), time.time() - start)
        save_result("AtomicAction flat JSON", False, str(e), time.time() - start)
    
    # Test 2c: ActionPlan - full plan
    start = time.time()
    try:
        plan = ActionPlan(
            reasoning="Navigate to the search page",
            actions=[
                AtomicAction(name="navigate", params={"url": "https://google.com"}),
                AtomicAction(name="type", params={"text": "test", "selector": "input"}),
                AtomicAction(name="click", params={"text": "Search"})
            ],
            confidence=0.9,
            next_mode="text",
            completed_subtasks=[]
        )
        assert len(plan.actions) == 3
        assert plan.confidence == 0.9
        print_test("ActionPlan construction", True, f"{len(plan.actions)} actions, confidence={plan.confidence}", time.time() - start)
        save_result("ActionPlan construction", True, "OK", time.time() - start)
    except Exception as e:
        print_test("ActionPlan construction", False, str(e), time.time() - start)
        save_result("ActionPlan construction", False, str(e), time.time() - start)
    
    # Test 2d: BrowserResult
    start = time.time()
    try:
        result = BrowserResult(
            success=True,
            task_summary="Found 5 search results",
            extracted_data={"results": ["a", "b", "c"]},
            metrics={"actions": {"total": 10}}
        )
        assert result.success is True
        assert result.task_summary == "Found 5 search results"
        print_test("BrowserResult construction", True, f"success={result.success}", time.time() - start)
        save_result("BrowserResult construction", True, "OK", time.time() - start)
    except Exception as e:
        print_test("BrowserResult construction", False, str(e), time.time() - start)
        save_result("BrowserResult construction", False, str(e), time.time() - start)
    
    # Test 2e: All action names are valid
    start = time.time()
    try:
        valid_actions = [
            "navigate", "click", "type", "scroll", "extract", "done",
            "hover", "press", "wait", "go_back", "go_forward", "save_screenshot",
            "save_info", "skip_subtask", "select", "upload_file", "download_file",
            "run_js", "press_keys", "save_credential", "get_credential", "save_learning"
        ]
        for action_name in valid_actions:
            a = AtomicAction(name=action_name, params={})
            assert a.name == action_name
        print_test("All action names valid", True, f"{len(valid_actions)} action types validated", time.time() - start)
        save_result("All action names valid", True, "OK", time.time() - start)
    except Exception as e:
        print_test("All action names valid", False, str(e), time.time() - start)
        save_result("All action names valid", False, str(e), time.time() - start)
    
    # Test 2f: Invalid action name should fail
    start = time.time()
    try:
        from pydantic import ValidationError
        try:
            AtomicAction(name="invalid_action", params={})
            print_test("Invalid action rejected", False, "Should have raised ValidationError", time.time() - start)
            save_result("Invalid action rejected", False, "No error raised", time.time() - start)
        except ValidationError:
            print_test("Invalid action rejected", True, "ValidationError raised as expected", time.time() - start)
            save_result("Invalid action rejected", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Invalid action rejected", False, str(e), time.time() - start)
        save_result("Invalid action rejected", False, str(e), time.time() - start)


# =============================================================================
# TEST 3: CONFIG & STATE
# =============================================================================

async def test_config_and_state():
    """Test config directory creation and state management."""
    print_header("TEST 3: CONFIG & STATE")
    
    # Test 3a: Config creates directories
    start = time.time()
    try:
        from agents.browser_agent.config import BrowserAgentConfig
        config = BrowserAgentConfig()
        
        assert config.DOWNLOADS_DIR.exists(), f"Downloads dir missing: {config.DOWNLOADS_DIR}"
        assert config.UPLOADS_DIR.exists(), f"Uploads dir missing: {config.UPLOADS_DIR}"
        assert config.SCREENSHOTS_DIR.exists(), f"Screenshots dir missing: {config.SCREENSHOTS_DIR}"
        
        print_test("Config dir creation", True, 
                   f"downloads={config.DOWNLOADS_DIR.exists()}, uploads={config.UPLOADS_DIR.exists()}, screenshots={config.SCREENSHOTS_DIR.exists()}", 
                   time.time() - start)
        save_result("Config dir creation", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Config dir creation", False, str(e), time.time() - start)
        save_result("Config dir creation", False, str(e), time.time() - start)
    
    # Test 3b: Config default values
    start = time.time()
    try:
        from agents.browser_agent.config import CONFIG
        
        assert CONFIG.MAX_STEPS == 50
        assert CONFIG.MAX_RETRIES == 3
        assert CONFIG.NAVIGATION_TIMEOUT == 60000
        assert CONFIG.LLM_TIMEOUT == 60
        
        print_test("Config defaults", True, 
                   f"max_steps={CONFIG.MAX_STEPS}, max_retries={CONFIG.MAX_RETRIES}", 
                   time.time() - start)
        save_result("Config defaults", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Config defaults", False, str(e), time.time() - start)
        save_result("Config defaults", False, str(e), time.time() - start)
    
    # Test 3c: AgentMemory - initialization
    start = time.time()
    try:
        from agents.browser_agent.state import AgentMemory
        
        memory = AgentMemory(task="Test task")
        assert memory.task == "Test task"
        assert len(memory.history) == 0
        assert len(memory.extracted_items) == 0
        
        print_test("AgentMemory initialization", True, "Memory initializes correctly", time.time() - start)
        save_result("AgentMemory initialization", True, "OK", time.time() - start)
    except Exception as e:
        print_test("AgentMemory subtask lifecycle", False, str(e), time.time() - start)
        save_result("AgentMemory subtask lifecycle", False, str(e), time.time() - start)
    
    # Test 3d: AgentMemory - extracted data accumulation
    start = time.time()
    try:
        from agents.browser_agent.state import AgentMemory
        
        memory = AgentMemory(task="Extract prices")
        
        memory.safe_add_extracted({"price": "$99"})
        memory.safe_add_extracted({"title": "Widget"})
        memory.safe_add_extracted({"structured_info": {"color": "red"}})
        memory.safe_add_extracted({"structured_info": {"size": "large"}})
        
        # extracted_data is a dict with keys merged from each call
        assert "price" in memory.extracted_data
        assert "title" in memory.extracted_data
        # structured_info accumulates into 'structured_items' list
        assert len(memory.extracted_data.get("structured_items", [])) >= 2
        # extracted_items is a list of all raw dicts appended for traceability
        assert len(memory.extracted_items) == 4
        
        print_test("AgentMemory data accumulation", True, 
                   f"extracted_data keys: {list(memory.extracted_data.keys())}, items_count: {len(memory.extracted_items)}", time.time() - start)
        save_result("AgentMemory data accumulation", True, "OK", time.time() - start)
    except Exception as e:
        print_test("AgentMemory data accumulation", False, str(e), time.time() - start)
        save_result("AgentMemory data accumulation", False, str(e), time.time() - start)
    
    # Test 3e disabled/removed due to Planner removal.


# =============================================================================
# TEST 4: MESSAGE MANAGER
# =============================================================================

async def test_message_manager():
    """Test token counting and priority-based truncation."""
    print_header("TEST 4: MESSAGE MANAGER")
    
    from agents.browser_agent.message_manager import MessageManager, format_page_content_for_prompt
    
    # Test 4a: Token counting
    start = time.time()
    try:
        count = MessageManager.count_tokens("Hello, this is a test sentence.")
        assert count > 0, f"Token count should be positive, got {count}"
        assert count < 50, f"Token count seems too high: {count}"
        
        # Longer text should have more tokens
        long_count = MessageManager.count_tokens("Hello world " * 100)
        assert long_count > count
        
        print_test("Token counting", True, f"short={count}, long={long_count}", time.time() - start)
        save_result("Token counting", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Token counting", False, str(e), time.time() - start)
        save_result("Token counting", False, str(e), time.time() - start)
    
    # Test 4b: Message manager initialization
    start = time.time()
    try:
        mm = MessageManager(max_total_tokens=8000)
        mm.set_system_prompt("You are a browser agent.")
        
        stats = mm.get_token_stats()
        assert stats["system_prompt"] > 0
        assert stats["history_total"] == 0
        
        print_test("MessageManager init", True, f"system_prompt_tokens={stats['system_prompt']}", time.time() - start)
        save_result("MessageManager init", True, "OK", time.time() - start)
    except Exception as e:
        print_test("MessageManager init", False, str(e), time.time() - start)
        save_result("MessageManager init", False, str(e), time.time() - start)
    
    # Test 4c: Adding steps and history management
    start = time.time()
    try:
        mm = MessageManager(max_total_tokens=8000)
        mm.set_system_prompt("System prompt.")
        
        # Add several steps
        for i in range(10):
            mm.add_step(
                step_number=i,
                action_names=["click", "type"],
                reasoning=f"Step {i}: Performing action",
                result_success=i % 3 != 0,  # Some failures
                result_message=f"Result of step {i}",
                url=f"https://example.com/page{i}"
            )
        
        stats = mm.get_token_stats()
        assert stats["history_messages"] >= 10
        
        # Get history for prompt (should be truncated within budget)
        history = mm.get_history_for_prompt()
        assert isinstance(history, str)
        assert len(history) > 0
        
        print_test("Step history management", True, 
                   f"messages={stats['history_messages']}, history_len={len(history)}", time.time() - start)
        save_result("Step history management", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Step history management", False, str(e), time.time() - start)
        save_result("Step history management", False, str(e), time.time() - start)
    
    # Test 4d: Page content formatting
    start = time.time()
    try:
        page_content = {
            "url": "https://example.com",
            "title": "Example Page",
            "body_text": "This is a sample page with some content. " * 50,
            "elements": [
                {"tag": "button", "text": "Click me", "index": 1},
                {"tag": "input", "text": "", "placeholder": "Search", "index": 2},
            ]
        }
        
        formatted = format_page_content_for_prompt(page_content, max_tokens=2000)
        assert isinstance(formatted, str)
        assert "example.com" in formatted or "Example Page" in formatted
        
        print_test("Page content formatting", True, f"formatted_len={len(formatted)}", time.time() - start)
        save_result("Page content formatting", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Page content formatting", False, str(e), time.time() - start)
        save_result("Page content formatting", False, str(e), time.time() - start)


# =============================================================================
# TEST 5: PLANNER & LLM INITIALIZATION
# =============================================================================

async def test_planner_and_llm():
    """Test LLMClient and Planner initialization."""
    print_header("TEST 5: PLANNER & LLM")
    
    # Test 5a: LLMClient initialization
    start = time.time()
    try:
        from agents.browser_agent.llm import LLMClient
        client = LLMClient()
        assert client is not None
        print_test("LLMClient init", True, "LLMClient created successfully", time.time() - start)
        save_result("LLMClient init", True, "OK", time.time() - start)
    except Exception as e:
        print_test("LLMClient init", False, str(e), time.time() - start)
        save_result("LLMClient init", False, str(e), time.time() - start)
    
    # Test 5b disabled/removed due to Planner removal.
    
    # Test 5c: System prompt is non-empty and substantial
    start = time.time()
    try:
        from agents.browser_agent.system_prompt import BROWSER_AGENT_SYSTEM_PROMPT
        
        assert len(BROWSER_AGENT_SYSTEM_PROMPT) > 1000, "System prompt too short"
        
        # Check key sections exist
        assert "AVAILABLE ACTIONS" in BROWSER_AGENT_SYSTEM_PROMPT
        assert "navigate" in BROWSER_AGENT_SYSTEM_PROMPT
        assert "click" in BROWSER_AGENT_SYSTEM_PROMPT
        assert "save_info" in BROWSER_AGENT_SYSTEM_PROMPT
        
        print_test("System prompt content", True, 
                   f"Length={len(BROWSER_AGENT_SYSTEM_PROMPT)} chars, key sections present", time.time() - start)
        save_result("System prompt content", True, "OK", time.time() - start)
    except Exception as e:
        print_test("System prompt content", False, str(e), time.time() - start)
        save_result("System prompt content", False, str(e), time.time() - start)
    
    # Test 5d: VisionClient initialization
    start = time.time()
    try:
        from agents.browser_agent.vision import VisionClient
        vision = VisionClient()
        assert vision is not None
        is_available = vision.available
        print_test("VisionClient init", True, f"available={is_available}", time.time() - start)
        save_result("VisionClient init", True, "OK", time.time() - start)
    except Exception as e:
        print_test("VisionClient init", False, str(e), time.time() - start)
        save_result("VisionClient init", False, str(e), time.time() - start)
    
    # Test 5e disabled/removed as LLM build_prompt has been replaced by build_state_message via chat history.


# =============================================================================
# TEST 6: AGENT INSTANTIATION
# =============================================================================

async def test_agent_instantiation():
    """Test that BrowserAgent constructs without crashing."""
    print_header("TEST 6: AGENT INSTANTIATION")
    
    # Test 6a: BrowserAgent construction
    start = time.time()
    try:
        from agents.browser_agent.agent import BrowserAgent
        
        agent = BrowserAgent(
            task="Test task - do not run",
            headless=True,
            thread_id="test-thread-001"
        )
        
        assert agent is not None
        assert agent.task == "Test task - do not run"
        assert agent.headless is True
        assert agent.thread_id == "test-thread-001"
        assert agent.memory is not None
        assert agent.llm is not None
        assert agent.vision is not None
        assert agent.browser is not None
        assert agent.executor is not None
        assert agent.dom is not None
        
        print_test("BrowserAgent construction", True, 
                   "All components initialized: memory, llm, vision, browser, executor, dom", 
                   time.time() - start)
        save_result("BrowserAgent construction", True, "OK", time.time() - start)
    except Exception as e:
        print_test("BrowserAgent construction", False, str(e), time.time() - start)
        save_result("BrowserAgent construction", False, str(e), time.time() - start)
    
    # Test 6b: Metrics initialization
    start = time.time()
    try:
        from agents.browser_agent.agent import BrowserAgent
        agent = BrowserAgent(task="Metrics test", headless=True)
        
        assert "actions" in agent.metrics
        assert "llm_calls" in agent.metrics
        assert "performance" in agent.metrics
        assert "navigation" in agent.metrics
        assert "vision" in agent.metrics
        assert "errors" in agent.metrics
        assert "tokens" in agent.metrics
        assert "dom" in agent.metrics
        
        assert agent.metrics["actions"]["total"] == 0
        assert agent.metrics["llm_calls"]["total"] == 0
        
        print_test("Metrics initialization", True, 
                   f"All metric categories present: {list(agent.metrics.keys())}", time.time() - start)
        save_result("Metrics initialization", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Metrics initialization", False, str(e), time.time() - start)
        save_result("Metrics initialization", False, str(e), time.time() - start)
    
    # Test 6c: DOMExtractor construction
    start = time.time()
    try:
        from agents.browser_agent.dom import DOMExtractor
        dom = DOMExtractor()
        
        assert dom.MAX_IFRAME_DEPTH == 3
        assert dom.MAX_IFRAMES == 3
        assert dom.MAX_ELEMENTS == 500
        
        print_test("DOMExtractor construction", True, 
                   f"MAX_ELEMENTS={dom.MAX_ELEMENTS}, MAX_IFRAMES={dom.MAX_IFRAMES}", time.time() - start)
        save_result("DOMExtractor construction", True, "OK", time.time() - start)
    except Exception as e:
        print_test("DOMExtractor construction", False, str(e), time.time() - start)
        save_result("DOMExtractor construction", False, str(e), time.time() - start)
    
    # Test 6d: ActionExecutor construction
    start = time.time()
    try:
        from agents.browser_agent.actions import ActionExecutor
        executor = ActionExecutor(thread_id="test-thread")
        
        assert executor is not None
        assert executor.dom is not None
        
        print_test("ActionExecutor construction", True, "ActionExecutor created with DOMExtractor", time.time() - start)
        save_result("ActionExecutor construction", True, "OK", time.time() - start)
    except Exception as e:
        print_test("ActionExecutor construction", False, str(e), time.time() - start)
        save_result("ActionExecutor construction", False, str(e), time.time() - start)
    
    # Test 6e: BaseAgent implementation
    start = time.time()
    try:
        from agents.browser_agent.base_agent_impl import BrowserAgent as BaseBrowserAgent
        
        agent = BaseBrowserAgent()
        assert agent is not None
        assert agent.agent_id == "browser_agent"
        
        print_test("BaseAgent implementation", True, 
                   f"agent_id={agent.agent_id}", time.time() - start)
        save_result("BaseAgent implementation", True, "OK", time.time() - start)
    except Exception as e:
        print_test("BaseAgent implementation", False, str(e), time.time() - start)
        save_result("BaseAgent implementation", False, str(e), time.time() - start)


# =============================================================================
# TEST 7: LIVE BROWSER E2E (requires Playwright)
# =============================================================================

async def test_live_browser():
    """Full end-to-end browser test: launch → navigate → extract → close."""
    print_header("TEST 7: LIVE BROWSER E2E (requires Playwright)")
    
    # Check Playwright availability
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print_skip("Live Browser E2E", "Playwright not installed (pip install playwright && playwright install)")
        save_result("Live Browser E2E", True, "SKIPPED - Playwright not installed", 0, skipped=True)
        return
    
    # Test 7a: Browser launch and close
    start = time.time()
    try:
        from agents.browser_agent.browser import Browser
        
        browser = Browser()
        launched = await browser.launch(headless=True, stealth=False, restore_session=False)
        
        assert launched, "Browser launch returned False"
        assert browser.page is not None, "No page after launch"
        
        url = await browser.get_url()
        
        await browser.close(save_session=False)
        
        print_test("Browser launch & close", True, f"Launched headless, initial URL: {url}", time.time() - start)
        save_result("Browser launch & close", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Browser launch & close", False, str(e), time.time() - start)
        save_result("Browser launch & close", False, str(e), time.time() - start)
        return  # Skip remaining live tests if browser can't launch
    
    # Test 7b: Navigation
    start = time.time()
    try:
        from agents.browser_agent.browser import Browser
        
        browser = Browser()
        await browser.launch(headless=True, stealth=False, restore_session=False)
        
        await browser.navigate("https://example.com")
        url = await browser.get_url()
        title = await browser.get_title()
        
        assert "example.com" in url, f"URL doesn't match: {url}"
        assert title, "No page title"
        
        await browser.close(save_session=False)
        
        print_test("Navigation", True, f"URL={url}, title={title}", time.time() - start)
        save_result("Navigation", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Navigation", False, str(e), time.time() - start)
        save_result("Navigation", False, str(e), time.time() - start)
    
    # Test 7c: DOM extraction
    start = time.time()
    try:
        from agents.browser_agent.browser import Browser
        from agents.browser_agent.dom import DOMExtractor
        
        browser = Browser()
        await browser.launch(headless=True, stealth=False, restore_session=False)
        await browser.navigate("https://example.com")
        
        dom = DOMExtractor()
        page_content = await dom.get_page_content(browser.page)
        
        assert page_content.get("url"), "No URL in page content"
        assert page_content.get("title"), "No title in page content"
        assert len(page_content.get("elements", [])) > 0, "No elements extracted"
        
        await browser.close(save_session=False)
        
        print_test("DOM extraction", True, 
                   f"elements={len(page_content['elements'])}, title={page_content['title']}", time.time() - start)
        save_result("DOM extraction", True, "OK", time.time() - start)
    except Exception as e:
        print_test("DOM extraction", False, str(e), time.time() - start)
        save_result("DOM extraction", False, str(e), time.time() - start)
    
    # Test 7d: Screenshot capture
    start = time.time()
    try:
        from agents.browser_agent.browser import Browser
        
        browser = Browser()
        await browser.launch(headless=True, stealth=False, restore_session=False)
        await browser.navigate("https://example.com")
        
        screenshot_bytes = await browser.screenshot()
        
        assert screenshot_bytes is not None, "Screenshot returned None"
        assert len(screenshot_bytes) > 1000, f"Screenshot too small: {len(screenshot_bytes)} bytes"
        
        await browser.close(save_session=False)
        
        print_test("Screenshot capture", True, 
                   f"screenshot_size={len(screenshot_bytes)} bytes", time.time() - start)
        save_result("Screenshot capture", True, "OK", time.time() - start)
    except Exception as e:
        print_test("Screenshot capture", False, str(e), time.time() - start)
        save_result("Screenshot capture", False, str(e), time.time() - start)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all browser agent tests."""
    print("\n" + "=" * 70)
    print("  BROWSER AGENT TEST SUITE")
    print("  Comprehensive tests for browser automation agent")
    print("=" * 70)
    
    overall_start = time.time()
    
    # Run all test groups
    test_groups = [
        ("Module Imports", test_module_imports),
        ("Schema Validation", test_schema_validation),
        ("Config & State", test_config_and_state),
        ("Message Manager", test_message_manager),
        ("Planner & LLM", test_planner_and_llm),
        ("Agent Instantiation", test_agent_instantiation),
        ("Live Browser E2E", test_live_browser),
    ]
    
    for name, test_fn in test_groups:
        try:
            await test_fn()
        except Exception as e:
            print(f"\n  ❌ FATAL ERROR in {name}: {e}")
            traceback.print_exc()
            save_result(f"{name} (FATAL)", False, str(e), 0)
    
    total_duration = time.time() - overall_start
    
    # Print summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in test_results if r["success"] and not r["skipped"])
    failed = sum(1 for r in test_results if not r["success"])
    skipped = sum(1 for r in test_results if r["skipped"])
    total = len(test_results)
    
    print(f"\n  Total:   {total} tests")
    print(f"  Passed:  {passed} ✅")
    print(f"  Failed:  {failed} ❌")
    print(f"  Skipped: {skipped} ⏭️")
    print(f"  Time:    {total_duration:.2f}s")
    
    if failed > 0:
        print("\n  Failed tests:")
        for r in test_results:
            if not r["success"]:
                print(f"    ❌ {r['test']}: {r['message'][:80]}")
    
    print("\n" + "=" * 70)
    
    # Save results
    results_file = backend_dir.parent / "browser_agent_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration": round(total_duration, 2),
            "results": test_results
        }, f, indent=2)
    
    print(f"  Results saved to: {results_file}")
    
    return passed, failed, skipped


if __name__ == "__main__":
    asyncio.run(run_all_tests())
