# agents/custom_browser_agent.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import asyncio
from dotenv import load_dotenv
import uvicorn
from openai import OpenAI
import logging
from pathlib import Path
import uuid
from datetime import datetime
import json
import base64
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not CEREBRAS_API_KEY and not GROQ_API_KEY:
    raise RuntimeError("At least one of CEREBRAS_API_KEY or GROQ_API_KEY must be set")

# --- Agent Definition ---
AGENT_DEFINITION = {
    "id": "custom_browser_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Custom Browser Automation Agent",
    "description": "A powerful custom browser automation agent with full control over web interactions",
    "capabilities": [
        "web browsing", "data extraction", "form filling", "screenshot capture",
        "web scraping", "page navigation", "element interaction"
    ],
    "price_per_call_usd": 0.01,
    "status": "active",
    "endpoints": [{
        "endpoint": "http://localhost:8070/browse",
        "http_method": "POST",
        "description": "Execute browser automation task",
        "parameters": [{
            "name": "task",
            "param_type": "string",
            "required": True,
            "description": "Task description"
        }]
    }]
}

app = FastAPI(title="Custom Browser Automation Agent")

# Storage
STORAGE_DIR = Path("storage/browser_screenshots")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Request/Response Models
class BrowseRequest(BaseModel):
    task: str
    extract_data: Optional[bool] = False
    max_steps: Optional[int] = 10

class BrowseResponse(BaseModel):
    success: bool
    task_summary: str
    actions_taken: List[Dict[str, Any]]
    extracted_data: Optional[Dict[str, Any]] = None
    screenshot_files: Optional[List[str]] = None
    error: Optional[str] = None

# LLM Client
def get_llm_client():
    """Get LLM client with fallback"""
    if GROQ_API_KEY:
        return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"), "groq"
    elif CEREBRAS_API_KEY:
        return OpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1"), "cerebras"
    raise RuntimeError("No LLM API key available")

class BrowserAgent:
    """Custom browser automation agent"""
    
    def __init__(self, task: str, max_steps: int = 10):
        self.task = task
        self.max_steps = max_steps
        self.actions_taken = []
        self.screenshots = []
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.llm_client, self.llm_provider = get_llm_client()
        self.task_id = str(uuid.uuid4())[:8]
        
    async def __aenter__(self):
        """Initialize browser"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.page = await self.context.new_page()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup browser"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    async def capture_screenshot(self, name: str = "screenshot") -> str:
        """Capture and save screenshot"""
        filename = f"{self.task_id}_{name}_{len(self.screenshots)}.png"
        filepath = STORAGE_DIR / filename
        await self.page.screenshot(path=str(filepath), full_page=False)
        self.screenshots.append(str(filepath))
        logger.info(f"📸 Screenshot saved: {filename}")
        return str(filepath)
    
    async def get_page_content(self) -> Dict[str, Any]:
        """Extract page content and structure"""
        # Get page text content
        text_content = await self.page.evaluate("""
            () => {
                return document.body.innerText.substring(0, 5000);
            }
        """)
        
        # Get interactive elements
        elements = await self.page.evaluate("""
            () => {
                const elements = [];
                // Links
                document.querySelectorAll('a[href]').forEach((el, i) => {
                    if (i < 20) elements.push({
                        type: 'link',
                        text: el.innerText.trim().substring(0, 50),
                        href: el.href
                    });
                });
                // Buttons
                document.querySelectorAll('button, input[type="button"], input[type="submit"]').forEach((el, i) => {
                    if (i < 20) elements.push({
                        type: 'button',
                        text: el.innerText.trim() || el.value || el.getAttribute('aria-label') || '',
                    });
                });
                // Input fields
                document.querySelectorAll('input:not([type="hidden"]), textarea').forEach((el, i) => {
                    if (i < 20) elements.push({
                        type: 'input',
                        name: el.name || el.id || el.placeholder || '',
                        inputType: el.type
                    });
                });
                return elements;
            }
        """)
        
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "text": text_content,
            "elements": elements
        }
    
    def plan_next_action(self, page_content: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        """Use LLM to plan next action"""
        
        prompt = f"""You are a browser automation agent. Analyze the current page and decide the next action.

TASK: {self.task}

CURRENT PAGE:
URL: {page_content['url']}
Title: {page_content['title']}

TEXT CONTENT (first 1000 chars):
{page_content['text'][:1000]}

AVAILABLE ELEMENTS:
{json.dumps(page_content['elements'][:15], indent=2)}

ACTIONS TAKEN SO FAR:
{json.dumps(self.actions_taken[-3:], indent=2) if self.actions_taken else 'None'}

STEP: {step_num}/{self.max_steps}

Decide the next action. Respond with ONLY a JSON object:
{{
    "action": "navigate|click|type|scroll|extract|done",
    "reasoning": "why this action",
    "params": {{
        "url": "for navigate",
        "text": "text to click or link text",
        "input_text": "for type action",
        "selector": "CSS selector if needed (e.g., input[name='q'])"
    }},
    "is_complete": true/false,
    "result_summary": "if complete, summarize findings"
}}

Rules:
- Start with navigate if no URL visited yet
- Use click for buttons/links (provide text to click)
- Use type for input fields (system will auto-detect search inputs)
- Use extract when you have the information needed
- Use done when task is complete
- Be specific and decisive
- If you see the information needed in TEXT CONTENT, mark as complete"""

        try:
            model = "llama-3.3-70b-versatile" if self.llm_provider == "groq" else "llama-3.3-70b"
            
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning(f"No JSON found in response: {content[:200]}")
                return {"action": "done", "is_complete": True, "result_summary": "Could not parse action"}
                
        except Exception as e:
            logger.error(f"Error planning action: {e}")
            return {"action": "done", "is_complete": True, "result_summary": f"Error: {e}"}
    
    async def execute_action(self, action_plan: Dict[str, Any]) -> bool:
        """Execute the planned action"""
        action = action_plan.get("action", "done")
        params = action_plan.get("params", {})
        
        try:
            if action == "navigate":
                url = params.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                logger.info(f"🌐 Navigating to: {url}")
                await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await self.page.wait_for_timeout(2000)
                
            elif action == "click":
                text = params.get("text", "")
                selector = params.get("selector")
                
                if selector:
                    logger.info(f"🖱️  Clicking selector: {selector}")
                    await self.page.click(selector, timeout=5000)
                elif text:
                    logger.info(f"🖱️  Clicking text: {text}")
                    # Try multiple strategies
                    try:
                        await self.page.get_by_text(text, exact=False).first.click(timeout=5000)
                    except:
                        await self.page.get_by_role("link", name=text).first.click(timeout=5000)
                        
                await self.page.wait_for_timeout(2000)
                
            elif action == "type":
                selector = params.get("selector")
                input_text = params.get("input_text", "")
                logger.info(f"⌨️  Typing: {input_text}")
                
                # Try multiple strategies to find input field
                if selector:
                    await self.page.fill(selector, input_text, timeout=5000)
                else:
                    # Try common search input selectors
                    selectors = [
                        'input[name="q"]',
                        'input[type="search"]',
                        'textarea[name="q"]',
                        'input[aria-label*="Search"]',
                        'input[placeholder*="Search"]'
                    ]
                    
                    filled = False
                    for sel in selectors:
                        try:
                            await self.page.fill(sel, input_text, timeout=2000)
                            filled = True
                            break
                        except:
                            continue
                    
                    if not filled:
                        # Last resort: find first visible input
                        await self.page.locator('input:visible').first.fill(input_text, timeout=5000)
                
            elif action == "scroll":
                logger.info("📜 Scrolling page")
                await self.page.evaluate("window.scrollBy(0, window.innerHeight)")
                await self.page.wait_for_timeout(1000)
                
            elif action == "extract":
                logger.info("📊 Extracting data")
                await self.capture_screenshot("extract")
                
            elif action == "done":
                logger.info("✅ Task marked as complete")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Error executing {action}: {e}")
            self.actions_taken.append({
                "action": action,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return False
    
    async def run(self) -> Dict[str, Any]:
        """Main execution loop"""
        logger.info(f"🚀 Starting browser agent for task: {self.task}")
        
        result_summary = ""
        
        for step in range(1, self.max_steps + 1):
            logger.info(f"\n📍 Step {step}/{self.max_steps}")
            
            # Get current page state
            page_content = await self.get_page_content()
            
            # Plan next action
            action_plan = self.plan_next_action(page_content, step)
            logger.info(f"💭 Plan: {action_plan.get('action')} - {action_plan.get('reasoning', '')}")
            
            # Record action
            self.actions_taken.append({
                "step": step,
                "action": action_plan.get("action"),
                "reasoning": action_plan.get("reasoning"),
                "url": page_content["url"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if complete
            if action_plan.get("is_complete"):
                result_summary = action_plan.get("result_summary", "Task completed")
                await self.capture_screenshot("final")
                break
            
            # Execute action
            is_done = await self.execute_action(action_plan)
            if is_done:
                result_summary = action_plan.get("result_summary", "Task completed")
                await self.capture_screenshot("final")
                break
            
            # Capture screenshot after action
            await self.capture_screenshot(f"step_{step}")
        
        logger.info(f"✅ Browser agent completed: {result_summary}")
        
        return {
            "success": True,
            "summary": result_summary or "Completed all steps",
            "actions": self.actions_taken,
            "screenshots": self.screenshots
        }

@app.post("/browse", response_model=BrowseResponse)
async def browse(request: BrowseRequest):
    """Execute browser automation task"""
    logger.info(f"📥 Received task: {request.task}")
    
    try:
        async with BrowserAgent(request.task, request.max_steps) as agent:
            result = await agent.run()
            
            return BrowseResponse(
                success=result["success"],
                task_summary=result["summary"],
                actions_taken=result["actions"],
                screenshot_files=result["screenshots"],
                extracted_data=None
            )
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return BrowseResponse(
            success=False,
            task_summary=f"Error: {str(e)}",
            actions_taken=[],
            error=str(e)
        )

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "custom_browser_agent"}

@app.get("/info")
async def info():
    return AGENT_DEFINITION

if __name__ == "__main__":
    port = int(os.getenv("BROWSER_AGENT_PORT", 8070))
    logger.info(f"Starting Custom Browser Automation Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
