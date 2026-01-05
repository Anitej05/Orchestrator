"""
Browser Agent - LLM Client

Simple, focused LLM client for action planning.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

from .schemas import ActionPlan, AtomicAction
from .system_prompt import get_system_prompt
from .message_manager import MessageManager, format_page_content_for_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Cerebras key pool for rotation (5 keys, skipping 6th)
CEREBRAS_KEYS = [
    "csk-52m4dv4chcpf9vy9jcmjrevnp5ft22y2vctd68wyr8dewndw",
    "csk-nnj93n833cr4c9rd2vttjeew3nwv494px62jfy45fmwjdch8",
    "csk-c2jjpt5k9kttxd44t9jwyn55vje4m2vmrvdjjkd6h2wphv6m",
    "csk-f428j58fvtm3n5vent8mjm3nfnwvk2rrvx6vww95e9wctv6c",
    "csk-hhcmv35w3kcvt9nffdyhp5f6m6epre8w3mcx32hwxxmyx85y",
]
# Filter out None/empty keys
CEREBRAS_KEYS = [k for k in CEREBRAS_KEYS if k]


class LLMClient:
    """LLM client for planning browser actions - uses true async calls"""
    
    def __init__(self):
        # Cerebras key pool for rotation
        self._cerebras_keys = CEREBRAS_KEYS.copy()
        self._cerebras_key_index = 0
        
        # Initialize Cerebras clients for each key
        self._cerebras_clients = []
        for key in self._cerebras_keys:
            client = AsyncOpenAI(
                api_key=key,
                base_url="https://api.cerebras.ai/v1",
                timeout=15.0  # Text model timeout
            )
            self._cerebras_clients.append(client)
        
        # Primary Cerebras client (will rotate on rate limit)
        self.cerebras = self._cerebras_clients[0] if self._cerebras_clients else None
        
        # Fallback 1: NVIDIA
        self.nvidia = AsyncOpenAI(
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=15.0  # Text model timeout
        ) if NVIDIA_API_KEY else None
        
        # Fallback 2: Groq (last resort)
        self.groq = AsyncOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=15.0  # Text model timeout
        ) if GROQ_API_KEY else None
        
        # Model Configuration
        self.model_cerebras = "zai-glm-4.6"  # Primary (with key rotation)
        self.model_nvidia = "minimaxai/minimax-m2"  # Fallback 1 (interleaved thinking)
        self.model_groq = "moonshotai/kimi-k2-instruct-0905"  # Fallback 2
        
        # Track temporarily disabled providers (rate limited)
        self._disabled_providers = {}  # provider -> disabled_until_time
        self._disabled_cerebras_keys = {}  # key_index -> disabled_until_time
        self._cerebras_key_index = 0  # For round-robin rotation

    
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
            
            response, usage = await self._call_llm(prompt)
            
            if response:
                result = self._parse_action(response)
                # Attach usage stats to the plan
                result.usage = usage
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
        """Build a focused prompt for action planning with token-aware history"""
        
        # Use MessageManager for token-aware history formatting
        history_str = ""
        if history:
            # Create a temporary manager to format history (12K tokens for history budget)
            manager = MessageManager(max_total_tokens=32000)
            
            for h in history:
                try:
                    step_num = h.get('step', 0)
                    action_plan = h.get('action', {})
                    reasoning = action_plan.get('reasoning', 'No reasoning')
                    actions = action_plan.get('actions', [])
                    action_names = [a['name'] for a in actions] if isinstance(actions, list) else ['?']
                    
                    result = h.get('result', {})
                    success = result.get('success', False)
                    msg = result.get('message', '')
                    url = h.get('url', '')
                    
                    manager.add_step(
                        step_number=step_num,
                        action_names=action_names,
                        reasoning=reasoning,
                        result_success=success,
                        result_message=msg,
                        url=url,
                        extracted_data=result.get('data'),
                        observation=h.get('observation', '')  # What was seen on page
                    )
                except Exception:
                    pass
            
            # Get token-aware formatted history
            history_str = manager.get_history_for_prompt()
            
            # Log token stats
            stats = manager.get_token_stats()
            logger.debug(f"📊 History tokens: {stats['history_total']}, "
                        f"messages: {stats['history_messages']}, "
                        f"failures: {stats['failures_count']}")
        
        # Get unified page tree (combines a11y hierarchy + elements + selectors)
        # This is the NEW approach - one unified structure instead of separate sections
        unified_page_tree = page_content.get('unified_page_tree', '')
        
        # Fallback: Build elements string if unified tree not available
        if not unified_page_tree:
            elements = page_content.get('elements', [])
            if elements:
                elem_lines = []
                for idx, el in enumerate(elements[:200]):  # Limit to 200
                    role = el.get('role', 'element')
                    name = el.get('name', '')[:60]
                    elem_idx = idx + 1
                    xpath = el.get('xpath', '')
                    
                    if name:
                        line = f"#{elem_idx} [{role}] \"{name}\""
                        if xpath and len(xpath) < 50:
                            line += f" → {xpath}"
                    else:
                        line = f"#{elem_idx} [{role}]"
                    elem_lines.append(line)
                
                unified_page_tree = "\n".join(elem_lines)
        
        # Build OVERLAY/MODAL section - CRITICAL for intelligent modal handling
        overlay_info = page_content.get('overlays', page_content.get('overlay_info', {}))
        overlay_str = ""
        if overlay_info and overlay_info.get('hasOverlay'):
            overlays = overlay_info.get('overlays', [])
            close_btns = overlay_info.get('closeButtons', [])
            overlay_lines = ["🚨 MODAL/OVERLAY DETECTED - Dismiss before continuing!"]
            for ov in overlays[:3]:
                title = ov.get('title', 'Unknown')
                ov_type = ov.get('type', 'modal')
                overlay_lines.append(f"  • Type: {ov_type}, Title: \"{title}\"")
            if close_btns:
                overlay_lines.append(f"  • Close: {[btn.get('text', 'X') for btn in close_btns[:3]]}")
            overlay_lines.append("  → Use press_keys: 'Escape' OR click Close/X button")
            overlay_str = "\n".join(overlay_lines)
        
        # Build scroll position context
        scroll_pct = page_content.get('scroll_percent', 100)
        
        if scroll_pct == 0:
            scroll_hint = "📍 AT TOP - scroll down to see more"
        elif scroll_pct >= 95:
            scroll_hint = "📍 AT BOTTOM - no more content below"
        else:
            scroll_hint = f"📍 {scroll_pct}% scrolled - can scroll up/down"
        
        # Build the current context prompt (system prompt is sent separately)
        prompt = f"""
═══════════════════════════════════════════════════════════════════════════════
🎯 TASK: {task}
═══════════════════════════════════════════════════════════════════════════════
📍 Step: {step} | URL: {page_content.get('url', 'about:blank')}
📄 Title: {page_content.get('title', '(loading...')}
{scroll_hint}
{f'''
{overlay_str}
''' if overlay_str else ''}
{self._format_selector_hints(page_content.get('selector_hints'))}
═══════════════════════════════════════════════════════════════════════════════
📄 PAGE (use #N index to click elements)
═══════════════════════════════════════════════════════════════════════════════
{unified_page_tree if unified_page_tree else "(no elements detected)"}

═══════════════════════════════════════════════════════════════════════════════
📜 PREVIOUS ACTIONS
═══════════════════════════════════════════════════════════════════════════════
{history_str if history_str else "(First step - no previous actions)"}

{f'''
⚠️ PREVIOUS ATTEMPT FAILED:
{last_error}

Please fix your response. Return ONLY valid JSON.
''' if last_error else ''}
NOW RESPOND WITH YOUR ACTION (JSON ONLY):"""
        
        return prompt
    
    def _format_selector_hints(self, hints: Optional[Dict[str, Any]]) -> str:
        """Format discovered selectors for the prompt"""
        if not hints:
            return ""
            
        lines = []
        # Handle the new 'patterns' style extracted by selector_discovery.py
        # It typically returns: {'patterns': [{'selector': '.foo', 'count': 10}, ...]}
        # Or sometimes a flat dict. Let's handle both safely.
        
        if isinstance(hints, dict):
            # 1. Recommended "Semantic" patterns
            if 'recommended' in hints:
                lines.append("✨ RECOMMENDED SELECTORS (High Confidence):")
                for k, v in hints.get('recommended', {}).items():
                     lines.append(f"  • {k}: {v}")
            
            # 2. Semantic Content Maps (Titles, Prices, etc.) - CRITICAL for extraction
            content_selectors = hints.get('contentSelectors', {}) or hints.get('content_selectors', {})
            if content_selectors:
                lines.append("\n🏷️ SEMANTIC CONTENT MAPS:")
                for category, items in content_selectors.items():
                    # items is a list of dicts: [{'selector': '.foo', 'count': 5, 'sample': '...'}]
                    if items and isinstance(items, list):
                        top_item = items[0]
                        sel = top_item.get('selector')
                        sample = top_item.get('sample')
                        if sel:
                            lines.append(f"  • {category.upper()}: {sel} (e.g., '{sample}')")

            # 3. Complete Container Schema (Nested Structure)
            # This shows: "Inside .product-card, we have .title, .price, etc."
            selector_map = hints.get('selectorMap', {}) or hints.get('selector_map', {})
            container = selector_map.get('container')
            if container:
                lines.append(f"\n📦 FOUND LIST SCHEMA (Container: '{container}'):")
                child_selectors = selector_map.get('childSelectors', [])
                if child_selectors:
                    # Sort by meaningfulness (price, text)
                    sorted_children = sorted(child_selectors, key=lambda x: (x.get('likelyPrice', False), x.get('hasContent', False)), reverse=True)[:5]
                    for child in sorted_children:
                        sel = child.get('selector')
                        tag = child.get('tag', 'element')
                        sample = (child.get('samples', []) or [''])[0]
                        lines.append(f"  • Child: {sel} <{tag}> (e.g., '{sample}')")

            # 3. Data Attributes (Robust technical hooks)
            data_attrs = hints.get('dataAttributes', []) or hints.get('data_attributes', [])
            if data_attrs:
                # Top 5 most frequent data attributes
                lines.append("\n⚓ DATA ATTRIBUTES (Robust Hooks):")
                # Format: [{'attr': 'data-testid', 'count': 10}, ...]
                sorted_attrs = sorted(data_attrs, key=lambda x: x.get('count', 0), reverse=True)[:5]
                for dp in sorted_attrs:
                    lines.append(f"  • [{dp.get('attr')}] ({dp.get('count')} elements)")

            # 4. General patterns (Fallback)
            patterns = hints.get('patterns', []) or hints.get('general', [])
            if patterns:
                lines.append("\n🔍 DISCOVERED PATTERNS (Repeating Structures):")
                # Take top 5 patterns by count
                sorted_pats = sorted(patterns, key=lambda x: x.get('count', 0), reverse=True)[:5]
                for p in sorted_pats:
                    sel = p.get('selector')
                    count = p.get('count', 0)
                    if sel:
                        lines.append(f"  • {sel} ({count} items)")
                        
        if not lines:
            return ""
            
        return "\n" + "\n".join(lines) + "\n"
    
    async def call_llm_direct(self, prompt: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Directly call LLM with prompt (for planning/analysis)"""
        return await self._call_llm(prompt)
    
    async def _call_llm(self, prompt: str, use_system_prompt: bool = True) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Call LLM with smart provider rotation. Immediately switches on rate limits."""
        import time
        
        # Get system prompt for browser automation context
        system_prompt = get_system_prompt() if use_system_prompt else None
        
        logger.info(f"📤 LLM Request (prompt length: {len(prompt)} chars, system: {bool(system_prompt)})")
        logger.debug(f"===== FULL PROMPT START =====\n{prompt}\n===== FULL PROMPT END =====")
        
        # Build messages
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        current_time = time.time()
        
        # Clean up expired disabled keys/providers
        self._disabled_providers = {
            k: v for k, v in self._disabled_providers.items() 
            if v > current_time
        }
        self._disabled_cerebras_keys = {
            k: v for k, v in self._disabled_cerebras_keys.items()
            if v > current_time
        }
        
        # ===== PHASE 1: Try all Cerebras keys (Primary - Round Robin) =====
        num_keys = len(self._cerebras_clients)
        start_index = self._cerebras_key_index
        
        for i in range(num_keys):
            # Round-robin selection: start from last success + 1
            key_idx = (start_index + i) % num_keys
            client = self._cerebras_clients[key_idx]
            
            # Skip disabled keys
            if key_idx in self._disabled_cerebras_keys:
                remaining = int(self._disabled_cerebras_keys[key_idx] - current_time)
                # Only log if it's the first attempt or verbose
                logger.info(f"⏭️ Skipping Cerebras key #{key_idx+1} (rate limited for {remaining}s more)")
                continue
            
            try:
                # Log usage
                logger.info(f"🤖 Calling Cerebras key #{key_idx+1} ({self.model_cerebras})...")
                
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model_cerebras,
                        messages=messages,
                        temperature=0.1
                    ),
                    timeout=15.0  # Text model timeout
                )
                
                content = None
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                
                if content:
                    logger.info(f"✅ Cerebras key #{key_idx+1} response ({len(content)} chars)")
                    logger.debug(f"===== FULL RESPONSE START =====\n{content}\n===== FULL RESPONSE END =====")
                    
                    # SUCCESS: Update start index for NEXT call to be this key + 1
                    # This ensures we distribute load across all keys over time
                    self._cerebras_key_index = (key_idx + 1) % num_keys
                    usage = response.usage.model_dump() if response.usage else None
                    return content, usage
                else:
                    logger.warning(f"⚠️ Cerebras key #{key_idx+1} returned empty content")
            
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Cerebras key #{key_idx+1} timed out after 15s - trying next key")
                continue
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit - disable this key and try next
                if any(keyword in error_str for keyword in [
                    '429', '413', 'rate_limit', 'too many requests', 
                    'quota exceeded', 'tokens per', 'tpm:', 'rpm:'
                ]):
                    # Cerebras has 10 RPM per key, so 60s cooldown
                    self._disabled_cerebras_keys[key_idx] = time.time() + 60  # Use fresh timestamp
                    logger.warning(f"⚡ Cerebras key #{key_idx+1} rate limited - trying next key (disabled for 60s)")
                    continue
                else:
                    logger.error(f"❌ Cerebras key #{key_idx+1} failed: {e}")
                    continue
        
        # ===== PHASE 2: Fallback to NVIDIA (5s timeout) =====
        if self.nvidia and "NVIDIA" not in self._disabled_providers:
            try:
                logger.info(f"🤖 Calling NVIDIA ({self.model_nvidia}) with 15s timeout...")
                response = await asyncio.wait_for(
                    self.nvidia.chat.completions.create(
                        model=self.model_nvidia,
                        messages=messages,
                        temperature=0.1
                    ),
                    timeout=15.0  # Text model timeout
                )
                
                content = None
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                
                if content:
                    # Strip thinking content for minimax
                    content = self._strip_thinking_content(content)
                    logger.info(f"✅ NVIDIA response ({len(content)} chars)")
                    logger.debug(f"===== FULL RESPONSE START =====\n{content}\n===== FULL RESPONSE END =====")
                    usage = response.usage.model_dump() if response.usage else None
                    return content, usage
                else:
                    logger.warning("⚠️ NVIDIA returned empty content")
            
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ NVIDIA timed out after 15s - moving to next provider")
                    
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['429', '413', 'rate_limit', 'too many requests']):
                    self._disabled_providers["NVIDIA"] = time.time() + 15  # Use fresh timestamp
                    logger.warning(f"⚡ NVIDIA rate limited - switching (disabled for 15s)")
                else:
                    logger.error(f"❌ NVIDIA failed: {e}")
        elif "NVIDIA" in self._disabled_providers:
            remaining = int(self._disabled_providers["NVIDIA"] - current_time)
            logger.info(f"⏭️ Skipping NVIDIA (rate limited for {remaining}s more)")
        
        # ===== PHASE 3: Last resort - Groq =====
        if self.groq and "Groq" not in self._disabled_providers:
            try:
                logger.info(f"🤖 Calling Groq ({self.model_groq}) with 15s timeout...")
                response = await asyncio.wait_for(
                    self.groq.chat.completions.create(
                        model=self.model_groq,
                        messages=messages,
                        temperature=0.1
                    ),
                    timeout=15.0  # Text model timeout
                )
                
                content = None
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                
                if content:
                    logger.info(f"✅ Groq response ({len(content)} chars)")
                    logger.debug(f"===== FULL RESPONSE START =====\n{content}\n===== FULL RESPONSE END =====")
                    usage = response.usage.model_dump() if response.usage else None
                    return content, usage
                else:
                    logger.warning("⚠️ Groq returned empty content")
            
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Groq timed out after 15s")
                    
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['429', '413', 'rate_limit', 'too many requests']):
                    self._disabled_providers["Groq"] = time.time() + 10  # Use fresh timestamp
                    logger.warning(f"⚡ Groq rate limited (disabled for 10s)")
                else:
                    logger.error(f"❌ Groq failed: {e}")
        elif "Groq" in self._disabled_providers:
            remaining = int(self._disabled_providers["Groq"] - current_time)
            logger.info(f"⏭️ Skipping Groq (rate limited for {remaining}s more)")
        
        logger.error("❌ All LLMs failed!")
        return None, None
    
    def _strip_thinking_content(self, content: str) -> str:
        """Strip thinking tags from model output, including interleaved thinking.
        
        Handles:
        - <think>...</think> and <thinking>...</thinking> (standard)
        - 【thinking】...【/thinking】 (minimax style)
        - Multiple interleaved thinking blocks
        """
        import re
        
        # Strip standard thinking tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Strip minimax-style interleaved thinking tags (【thinking】 format)
        content = re.sub(r'【thinking】.*?【/thinking】', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'【思考】.*?【/思考】', '', content, flags=re.DOTALL | re.IGNORECASE)  # Chinese variant
        
        # Strip markdown-style thinking blocks that some models use
        content = re.sub(r'\*\*Thinking:\*\*.*?(?=\*\*(?:Response|Answer|Output):\*\*|\{)', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Handle unclosed tags at the start (model might still be thinking)
        content = re.sub(r'^<think>.*?(?=\{)', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'^<thinking>.*?(?=\{)', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'^【thinking】.*?(?=\{)', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace that might result from stripping
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()
    
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
            
            response, _ = await self._call_llm(prompt)
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
                        # Use model_validate to allow AtomicAction validator to restructure params
                        try:
                            action = AtomicAction.model_validate(act)
                            # DEBUG: Verify action params
                            logger.info(f"DEBUG_VALIDATE success: {act} -> {action.params}")
                            actions.append(action)
                        except Exception as e:
                            logger.warning(f"Failed to validate action {act}: {e}")
                            # Fallback to manual verify if validation completely fails
                            try:
                                params = act.get('params', {})
                                # ROBUST FALLBACK: If params is empty, check for top-level keys (common LLM error)
                                if not params:
                                    params = {k: v for k, v in act.items() if k != 'name'}
                                    
                                action = AtomicAction(
                                    name=act['name'], 
                                    params=params
                                )
                                actions.append(action)
                            except:
                                pass
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

