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
from .system_prompt import get_system_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


class LLMClient:
    """LLM client for planning browser actions"""
    
    def __init__(self):
        # Primary: NVIDIA (minimaxai/minimax-m2 - better reasoning)
        self.nvidia = OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1"
        ) if NVIDIA_API_KEY else None
        
        # Fallback 1: Cerebras (fast)
        self.cerebras = OpenAI(
            api_key=CEREBRAS_API_KEY,
            base_url="https://api.cerebras.ai/v1"
        ) if CEREBRAS_API_KEY else None
        
        # Fallback 2: Groq
        self.groq = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        ) if GROQ_API_KEY else None
        
        # Model Configuration
        self.model_nvidia = "minimaxai/minimax-m2"
        self.model_cerebras = "gpt-oss-120b"
        self.model_groq = "openai/gpt-oss-120b"

    
    async def plan_action(
        self, 
        task: str, 
        page_content: Dict[str, Any], 
        history: List[Dict[str, Any]],
        step: int,
        last_error: Optional[str] = None
    ) -> ActionPlan:
        """Plan next action based on task and current page state. Retries on failure."""
        
        MAX_RETRIES = 3
        last_parse_error = last_error
        
        for attempt in range(MAX_RETRIES):
            prompt = self._build_prompt(task, page_content, history, step, last_parse_error)
            
            response = await self._call_llm(prompt)
            
            if response:
                result = self._parse_action(response)
                if result.confidence > 0.3:  # Valid parse
                    return result
                else:
                    # Parse failed, capture error for retry
                    last_parse_error = f"Attempt {attempt+1} failed: Your response was not valid JSON. Response was: {response[:200]}..."
                    logger.warning(f"⚠️ Parse failed (attempt {attempt+1}/{MAX_RETRIES}): {last_parse_error[:100]}")
            else:
                last_parse_error = f"Attempt {attempt+1}: LLM returned empty response"
                logger.warning(f"⚠️ LLM returned empty (attempt {attempt+1}/{MAX_RETRIES})")
        
        # All retries failed - return a smart fallback based on current state
        logger.error(f"❌ All {MAX_RETRIES} attempts failed to get valid action")
        
        # Smart fallback: if on blank page, try to navigate
        url = page_content.get('url', '')
        if url == 'about:blank' or not url:
            # Extract URL from task if possible
            import re
            url_match = re.search(r'(google|flipkart|amazon|reliance)', task.lower())
            if url_match:
                site = url_match.group(1)
                fallback_url = f"https://www.{site}.com"
                if site == 'reliance':
                    fallback_url = "https://www.reliancedigital.in"
                elif site == 'amazon':
                    fallback_url = "https://www.amazon.in"
                elif site == 'flipkart':
                    fallback_url = "https://www.flipkart.com"
                    
                return ActionPlan(
                    reasoning=f"LLM failed to respond properly. Fallback: navigating to {fallback_url}",
                    actions=[AtomicAction(name="navigate", params={"url": fallback_url})],
                    confidence=0.6,
                    next_mode="text"
                )
        
        # If already on a page, try scrolling to load more content
        return ActionPlan(
            reasoning="LLM failed to respond. Fallback: scrolling to see more content.",
            actions=[AtomicAction(name="scroll", params={"direction": "down"})],
            confidence=0.4,
            next_mode="text"
        )
    
    def _build_prompt(
        self, 
        task: str, 
        page_content: Dict[str, Any], 
        history: List[Dict[str, Any]],
        step: int,
        last_error: Optional[str] = None
    ) -> str:
        """Build a focused prompt for action planning"""
        
        # Format history (last 5 actions with success/failure status)
        history_str = ""
        if history:
            # Show FULL history (up to 50 steps) so agent remembers everything
            # User explicitly requested "literally everything it has been doing"
            recent = history[-50:] 
            history_lines = []
            for h in recent:
                try:
                    step_num = h.get('step', '?')
                    action_plan = h.get('action', {})
                    reasoning = action_plan.get('reasoning', 'No reasoning')
                    actions = action_plan.get('actions', [])
                    action_names = [a['name'] for a in actions] if isinstance(actions, list) else ['?']
                    
                    result = h.get('result', {})
                    success = "✅" if result.get('success', False) else "❌"
                    msg = result.get('message', '')[:100] # Increased log length
                    
                    # Highlight Failed Attempts
                    status_icon = success
                    if not result.get('success', False):
                         status_icon = "🛑 FAILED"

                    line = f"  {status_icon} Step {step_num}: {action_names} \n    Reasoning: {reasoning[:150]}\n    Result: {msg}"
                    history_lines.append(line)
                except:
                    pass
            history_str = "\n".join(history_lines)
        
        # Format elements with BOTH xpath AND role+name for maximum flexibility
        elements = page_content.get('elements', [])[:100]  # Include more elements
        elements_str = ""
        if elements:
            elem_lines = []
            for el in elements:
                role = el.get('role', 'element')
                name = el.get('name', '')
                xpath = el.get('xpath', '')
                options = el.get('options', [])  # Dropdown options if present
                
                # Build element line
                if xpath and name:
                    line = f'  [{role}] "{name}"\n      xpath: {xpath}\n      OR: {{"role": "{role}", "name": "{name}"}}'
                elif xpath:
                    line = f'  [{role}] "(no text)" → xpath: {xpath}'
                elif name:
                    line = f'  [{role}] "{name}" → {{"role": "{role}", "name": "{name}"}}'
                else:
                    continue
                
                # Add dropdown options if present
                if options:
                    line += f'\n      OPTIONS: {options}'
                
                elem_lines.append(line)
            elements_str = "\n".join(elem_lines)
        
        # Format accessibility tree (backup for elements not in list)
        a11y_tree = page_content.get('a11y_tree', '')
        if a11y_tree:
            a11y_section = f"""
ACCESSIBILITY TREE (for role+name clicks):
Elements with #N are interactive - you can click them using role+name!
Example: #5 [link] "Samsung Galaxy" → {{"role": "link", "name": "Samsung Galaxy"}}
{a11y_tree}"""
        else:
            a11y_section = ""
        
        # Build scroll position context
        scroll_pos = page_content.get('scroll_position', 0)
        max_scroll = page_content.get('max_scroll', 0)
        scroll_pct = page_content.get('scroll_percent', 100)
        scroll_context = f"Scroll Position: {scroll_pos}px / {max_scroll}px ({scroll_pct}% down the page)"
        if scroll_pct == 0:
            scroll_context += " [AT TOP - can scroll down]"
        elif scroll_pct >= 100:
            scroll_context += " [AT BOTTOM - cannot scroll down further]"
        else:
            scroll_context += " [CAN SCROLL UP AND DOWN]"
        
        # Build the current context prompt (system prompt is sent separately)
        prompt = f"""
═══════════════════════════════════════════════════════════════════════════════
CURRENT TASK
═══════════════════════════════════════════════════════════════════════════════
{task}

═══════════════════════════════════════════════════════════════════════════════
PAGE STATE
═══════════════════════════════════════════════════════════════════════════════
URL: {page_content.get('url', 'about:blank')}
Title: {page_content.get('title', '(empty)')}
Step: {step}
{scroll_context}
{a11y_section}

PAGE TEXT (excerpt):
{page_content.get('body_text', '')}

═══════════════════════════════════════════════════════════════════════════════
INTERACTIVE ELEMENTS (with XPaths)
═══════════════════════════════════════════════════════════════════════════════
{elements_str if elements_str else "(no elements detected)"}

═══════════════════════════════════════════════════════════════════════════════
PREVIOUS ACTIONS
═══════════════════════════════════════════════════════════════════════════════
{history_str if history_str else "(none)"}

{f'''
⚠️ PREVIOUS ATTEMPT FAILED:
{last_error}

Please fix your response. Return ONLY valid JSON.
''' if last_error else ''}

NOW RESPOND WITH YOUR ACTION (JSON ONLY):"""
        
        return prompt
    
    async def call_llm_direct(self, prompt: str) -> Optional[str]:
        """Directly call LLM with prompt (for planning/analysis)"""
        return await self._call_llm(prompt)
    
    async def _call_llm(self, prompt: str, use_system_prompt: bool = True) -> Optional[str]:
        """Call LLM with fallback. Uses system prompt by default for action planning."""
        
        # Get system prompt for browser automation context
        system_prompt = get_system_prompt() if use_system_prompt else None
        
        logger.info(f"📤 LLM Request (prompt length: {len(prompt)} chars, system: {bool(system_prompt)})")
        logger.debug(f"===== FULL PROMPT START =====\n{prompt}\n===== FULL PROMPT END =====")
        
        # Build messages - use system + user pattern if system prompt available
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Try NVIDIA first (minimaxai/minimax-m2 for better reasoning)
        if self.nvidia:
            try:
                logger.info(f"🤖 Calling NVIDIA ({self.model_nvidia})...")
                response = self.nvidia.chat.completions.create(
                    model=self.model_nvidia,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.1
                )
                
                content = None
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                
                if content:
                    logger.info(f"✅ NVIDIA response ({len(content)} chars)")
                    logger.debug(f"===== FULL RESPONSE START =====\n{content}\n===== FULL RESPONSE END =====")
                    return content
                else:
                    logger.warning("⚠️ NVIDIA returned empty content")
            except Exception as e:
                logger.error(f"❌ NVIDIA failed: {e}")
        
        # Fallback 1: Cerebras
        if self.cerebras:
            try:
                logger.info(f"🤖 Calling Cerebras ({self.model_cerebras})...")
                response = self.cerebras.chat.completions.create(
                    model=self.model_cerebras,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.1
                )
                
                content = None
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                
                if content:
                    logger.info(f"✅ Cerebras response ({len(content)} chars)")
                    logger.debug(f"===== FULL RESPONSE START =====\n{content}\n===== FULL RESPONSE END =====")
                    return content
                else:
                    logger.warning("⚠️ Cerebras returned empty content")
            except Exception as e:
                logger.error(f"❌ Cerebras failed: {e}")

        
        # Fallback to Groq
        if self.groq:
            try:
                logger.info(f"🤖 Calling Groq ({self.model_groq})...")
                response = self.groq.chat.completions.create(
                    model=self.model_groq,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.1
                )
                
                content = None
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    
                if content:
                    logger.info(f"✅ Groq response ({len(content)} chars)")
                    logger.debug(f"===== FULL RESPONSE START =====\n{content}\n===== FULL RESPONSE END =====")
                    return content
                else:
                    logger.warning("⚠️ Groq returned empty content")
                    
            except Exception as e:
                logger.error(f"❌ Groq failed: {e}")
        
        logger.error("❌ All LLMs failed!")
        return None
    
    async def should_extend_timeout(
        self, 
        task: str, 
        current_step: int, 
        action_name: str, 
        context: Dict[str, Any],
        retry_count: int
    ) -> Dict[str, Any]:
        """Ask LLM if we should extend timeout for an action"""
        
        # Default decision if LLM fails
        default_decision = {"decision": "FAIL", "reasoning": "LLM failed to respond"}
        if retry_count < 2 and action_name in ["navigate", "click"]:
            default_decision = {"decision": "EXTEND", "multiplier": 2.0, "reasoning": "Automatic fallback extension"}

        try:
            prompt = f"""You are a browser automation agent. An action has timed out. You need to decide if we should retry with a longer timeout or fail.

TASK: {task}
STEP: {current_step}

TIMED OUT ACTION:
Action: {action_name}
Context: {context}
Current Retry: {retry_count + 1}

DECISION CRITERIA:
- If this is a heavy page load (e.g. Flipkart, Amazon) and retry count is low: EXTEND.
- If this looks like a genuine error or infinite loop: SKIP or FAIL.
- If we have already retried twice: FAIL.

OPTIONS:
1. EXTEND: Retry with increased timeout (1.5x - 2x).
2. SKIP: Mark this action as failed but continue to next step (if possible).
3. FAIL: Mark the entire task as failed.

Respond with ONLY valid JSON:
{{
    "decision": "EXTEND|SKIP|FAIL",
    "multiplier": 2.0,  # If EXTEND (1.0 - 3.0)
    "reasoning": "Brief reason why"
}}"""
            
            response = await self._call_llm(prompt)
            if not response:
                return default_decision
                
            # Parse JSON
            try:
                # Clean markdown
                cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
                # Find JSON object
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(0)
                    
                import json
                data = json.loads(cleaned)
                return data
            except Exception as e:
                logger.error(f"Failed to parse timeout decision: {e}")
                return default_decision
                
        except Exception as e:
            logger.error(f"Error in timeout decision: {e}")
            return default_decision

    def _parse_action(self, response: str) -> ActionPlan:
        """Parse LLM response into ActionPlan"""
        try:
            # Strip thinking tags
            # Strip thinking tags (handle both closed and unclosed/malformed)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
            # Handle unclosed tags at start
            response = re.sub(r'^<think>.*?(?=\{)', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'^<thinking>.*?(?=\{)', '', response, flags=re.DOTALL | re.IGNORECASE)
            
            # Try to extract JSON from code blocks first
            code_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response, re.IGNORECASE)
            if code_match:
                json_str = code_match.group(1)
            else:
                # Find JSON object - be more careful with nested braces
                json_match = re.search(r'\{[\s\S]*\}', response)
                json_str = json_match.group() if json_match else None
            
            if json_str:
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Try to extract just the first valid JSON object
                    logger.warning(f"JSON parse error: {e}, attempting repair...")
                    # Find matching braces
                    depth = 0
                    start = json_str.find('{')
                    if start >= 0:
                        for i, c in enumerate(json_str[start:], start):
                            if c == '{': depth += 1
                            elif c == '}': depth -= 1
                            if depth == 0:
                                try:
                                    data = json.loads(json_str[start:i+1])
                                    break
                                except:
                                    pass
                        else:
                            raise e
                    else:
                        raise e
                
                actions = []
                if 'actions' in data:
                    for act in data['actions']:
                        actions.append(AtomicAction(name=act['name'], params=act.get('params', {})))
                else:
                     # Fallback for old single action format from LLM
                     actions.append(AtomicAction(
                        name=data.get('action', 'wait'), 
                        params=data.get('params', {'seconds': 1})
                    ))

                return ActionPlan(
                    reasoning=data.get('reasoning', ''),
                    actions=actions,
                    confidence=data.get('confidence', 0.8),
                    next_mode=data.get('next_mode', 'text'),
                    completed_subtasks=data.get('completed_subtasks', []),
                    updated_plan=data.get('updated_plan', None)
                )
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response[:500]}...")
        
        # IMPORTANT: Fallback to wait, NOT done - don't prematurely complete!
        return ActionPlan(
            reasoning="Failed to parse action - will retry",
            actions=[AtomicAction(name="wait", params={"seconds": 2})],
            confidence=0.3,
            next_mode="text"  # Stay in text mode to avoid vision trap
        )

