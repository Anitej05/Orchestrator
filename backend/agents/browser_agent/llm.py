"""
Browser Agent - LLM Client

Simple, focused LLM client for action planning.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .schemas import ActionPlan, AtomicAction

load_dotenv()
logger = logging.getLogger(__name__)

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class LLMClient:
    """LLM client for planning browser actions"""
    
    def __init__(self):
        # Primary: Cerebras (fast)
        self.cerebras = OpenAI(
            api_key=CEREBRAS_API_KEY,
            base_url="https://api.cerebras.ai/v1"
        ) if CEREBRAS_API_KEY else None
        
        # Fallback: Groq
        self.groq = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        ) if GROQ_API_KEY else None
        
        # Model Configuration
        self.model_cerebras = "gpt-oss-120b"
        self.model_groq = "openai/gpt-oss-120b"
    
    async def plan_action(
        self, 
        task: str, 
        page_content: Dict[str, Any], 
        history: List[Dict[str, Any]],
        step: int,
        max_steps: int
    ) -> ActionPlan:
        """Plan next action based on task and current page state"""
        
        prompt = self._build_prompt(task, page_content, history, step, max_steps)
        
        # Try Cerebras first, then Groq
        response = await self._call_llm(prompt)
        
        if response:
            return self._parse_action(response)
        
        # Default fallback
        return ActionPlan(
            reasoning="Could not plan action",
            actions=[AtomicAction(name="done", params={})],
            confidence=0.5
        )
    
    def _build_prompt(
        self, 
        task: str, 
        page_content: Dict[str, Any], 
        history: List[Dict[str, Any]],
        step: int,
        max_steps: int
    ) -> str:
        """Build a focused prompt for action planning"""
        
        # Format history (last 3 actions)
        history_str = ""
        if history:
            recent = history[-3:]
            history_lines = []
            for h in recent:
                try:
                    action_plan = h.get('action', {})
                    actions = action_plan.get('actions', [])
                    action_names = [a['name'] for a in actions] if isinstance(actions, list) else [action_plan.get('action', '?')]
                    result = h.get('result', {})
                    history_lines.append(f"  - Sequence {action_names}: {result.get('message', '')}")
                except:
                    history_lines.append(f"  - Action: {h}")
            history_str = "\n".join(history_lines)
        
        # Format elements (top 20)
        elements = page_content.get('elements', [])[:20]
        elements_str = ""
        if elements:
            elem_lines = []
            for el in elements:
                elem_lines.append(f"  [{el.get('tag')}] \"{el.get('text')}\" | selector: {el.get('selector')} | coords: ({el.get('x')}, {el.get('y')})")
            elements_str = "\n".join(elem_lines)
        
        prompt = f"""You are a browser automation agent. Analyze the current page and decide the next action.

TASK: {task}

CURRENT PAGE:
- URL: {page_content.get('url', 'about:blank')}
- Title: {page_content.get('title', '')}
- Step: {step}/{max_steps}

PAGE CONTENT (first 1500 chars):
{page_content.get('body_text', '')[:1500]}

INTERACTIVE ELEMENTS:
{elements_str if elements_str else "No interactive elements found"}

RECENT ACTIONS:
{history_str if history_str else "None yet"}

AVAILABLE ACTIONS:
- navigate: Go to a URL. Params: {{"url": "https://..."}}
- click: Click element. Params: {{"selector": "...", "text": "...", "x": 0, "y": 0}}
- type: Type text. Params: {{"text": "...", "selector": "...", "submit": true}}
- hover: Hover element. Params: {{"selector": "..."}}
- press: Press key. Params: {{"key": "Enter|Escape|Tab|..."}}
- scroll: Scroll page. Params: {{"direction": "down/up"}}
- wait: Wait for seconds. Params: {{"seconds": 2}}
- go_back: Navigate back. Params: {{}}
- go_forward: Navigate forward. Params: {{}}
- screenshot: Save screenshot to disk. Params: {{"label": "description"}}
- extract: Get page data. Params: {{}}
- done: Task complete. Params: {{}}

INSTRUCTIONS:
1. **BE EFFICIENT**: Do as much as possible in a single turn.
2. Return a SEQUENCE of actions to execute (e.g., [Type "search", Click "Button"]).
3. Group logical steps: Navigation -> Interaction -> Extraction.
4. If the task requires vision next, set "next_mode": "vision".

Respond with ONLY valid JSON:
{{
    "reasoning": "brief plan",
    "actions": [
        {{"name": "navigate", "params": {{"url": "https://..."}}}},
        {{"name": "type", "params": {{"text": "query", "submit": true}}}}
    ],
    "confidence": 0.85,
    "next_mode": "text|vision",
    "completed_subtasks": []
}}"""
        
        return prompt
    
    async def call_llm_direct(self, prompt: str) -> Optional[str]:
        """Directly call LLM with prompt (for planning/analysis)"""
        return await self._call_llm(prompt)
    
    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with fallback"""
        
        # Try Cerebras first
        if self.cerebras:
            try:
                response = self.cerebras.chat.completions.create(
                    model=self.model_cerebras,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                content = response.choices[0].message.content
                logger.info(f"🤖 Cerebras response: {len(content)} chars")
                return content
            except Exception as e:
                logger.warning(f"Cerebras failed: {e}")
        
        # Fallback to Groq
        if self.groq:
            try:
                response = self.groq.chat.completions.create(
                    model=self.model_groq,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                content = response.choices[0].message.content
                logger.info(f"🤖 Groq response: {len(content)} chars")
                return content
            except Exception as e:
                logger.warning(f"Groq failed: {e}")
        
        return None
    
    def _parse_action(self, response: str) -> ActionPlan:
        """Parse LLM response into ActionPlan"""
        try:
            # Strip thinking tags
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
            
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                actions = []
                if 'actions' in data:
                    for act in data['actions']:
                        actions.append(AtomicAction(name=act['name'], params=act.get('params', {})))
                else:
                     # Fallback for old single action format from LLM
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
            logger.error(f"Failed to parse LLM response: {e}")
        
        return ActionPlan(
            reasoning="Failed to parse action",
            actions=[AtomicAction(name="done", params={})],
            confidence=0.3
        )
