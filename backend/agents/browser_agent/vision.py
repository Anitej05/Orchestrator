"""
Browser Agent - Vision Module

Vision model integration for screenshot-based action planning.
Supports image analysis and coordinate-based actions.
"""

import os
import re
import json
import base64
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .schemas import ActionPlan

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")


class VisionClient:
    """Vision model client for screenshot-based action planning"""
    
    def __init__(self):
        self.ollama = OpenAI(
            api_key=OLLAMA_API_KEY,
            base_url="https://ollama.com/v1"
        ) if OLLAMA_API_KEY else None
        
        self.model = "qwen3-vl:235b-cloud"
    
    @property
    def available(self) -> bool:
        """Check if vision is available"""
        return self.ollama is not None
    
    async def plan_action_with_vision(
        self,
        task: str,
        screenshot_base64: str,
        page_content: Dict[str, Any],
        history: list,
        step: int,
        max_steps: int
    ) -> Optional[ActionPlan]:
        """Plan action based on screenshot analysis"""
        
        if not self.available:
            logger.warning("Vision not available (no OLLAMA_API_KEY)")
            return None
        
        try:
            prompt = self._build_vision_prompt(task, page_content, history, step, max_steps)
            
            response = self.ollama.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if not content or not content.strip():
                logger.warning("Vision returned empty response")
                return None
            
            logger.info(f"🎨 Vision response: {len(content)} chars")
            return self._parse_action(content)
            
        except Exception as e:
            logger.error(f"Vision failed: {e}")
            return None
    
    async def analyze_image(
        self,
        screenshot_base64: str,
        task: str,
        page_url: str
    ) -> Optional[str]:
        """Analyze/describe an image on the page"""
        
        if not self.available:
            return None
        
        try:
            prompt = f"""Analyze this screenshot and describe what you see.

TASK: {task}
URL: {page_url}

Describe any images, logos, doodles, or visual content you see on the page.
Focus on visual elements that are relevant to the task.

Provide a detailed description of what you observe."""

            response = self.ollama.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if content:
                # Strip any thinking tags
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
                return content.strip()
            return None
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return None
    
    def _build_vision_prompt(
        self,
        task: str,
        page_content: Dict[str, Any],
        history: list,
        step: int,
        max_steps: int
    ) -> str:
        """Build prompt for vision-based action planning"""
        
        # Recent actions
        history_str = ""
        if history:
            recent = history[-3:]
            lines = [f"  - {h['action']['action']}: {h['result']['message']}" for h in recent]
            history_str = "\n".join(lines)
        
        return f"""You are a browser automation agent. Look at this screenshot and decide the next action.

TASK: {task}

CURRENT STATE:
- URL: {page_content.get('url', 'about:blank')}
- Title: {page_content.get('title', '')}
- Step: {step}/{max_steps}

RECENT ACTIONS:
{history_str if history_str else "None yet"}

AVAILABLE ACTIONS:
- click: Params: {{"x": 100, "y": 200}} or {{"selector": "..."}}
- type: Params: {{"text": "...", "selector": "..."}}
- scroll: Params: {{"direction": "down/up"}}
- hover: Params: {{"selector": "..."}}
- press: Params: {{"key": "Enter|Esc|..."}}
- wait: Params: {{"seconds": 1}}
- navigate: Params: {{"url": "..."}}
- go_back / go_forward: Params: {{}}
- screenshot: Save to disk. Params: {{"label": "description"}}
- done: Params: {{}}

INSTRUCTIONS:
1. Look at the screenshot carefully.
2. Return a SEQUENCE of actions to execute.
3. Be efficient: Type and Click in one response if possible.
4. "next_mode": "vision" if you need to keep seeing the screen.
5. Use "hover" for dropdowns/menus.
6. Use "press" for keyboard shortcuts.

Respond with ONLY valid JSON:
{{
    "reasoning": "What you see and plan",
    "actions": [
        {{"name": "click", "params": {{"x": 100, "y": 200}}}},
        {{"name": "type", "params": {{"text": "hello"}}}},
        {{"name": "press", "params": {{"key": "Enter"}}}}
    ],
    "confidence": 0.85,
    "next_mode": "text|vision",
    "completed_subtasks": []
}}"""
    
    def _parse_action(self, response: str) -> Optional[ActionPlan]:
        """Parse vision response into ActionPlan"""
        try:
            # Strip thinking tags
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
            
            # Extract JSON from code blocks if present
            code_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response, re.IGNORECASE)
            if code_match:
                json_str = code_match.group(1)
            else:
                # Find JSON object
                json_match = re.search(r'\{[\s\S]*\}', response)
                json_str = json_match.group() if json_match else None
            
            if json_str:
                data = json.loads(json_str)
                
                # Handle actions list or single action fallback
                actions = []
                if 'actions' in data:
                    for act in data['actions']:
                        actions.append(AtomicAction(name=act['name'], params=act.get('params', {})))
                else:
                    actions.append(AtomicAction(
                        name=data.get('action', 'done'), 
                        params=data.get('params', {})
                    ))

                return ActionPlan(
                    reasoning=data.get('reasoning', ''),
                    actions=actions,
                    confidence=data.get('confidence', 0.8),
                    next_mode=data.get('next_mode', 'text'),
                    completed_subtasks=data.get('completed_subtasks', [])
                )
        except Exception as e:
            logger.error(f"Failed to parse vision response: {e}")
        
        return None
