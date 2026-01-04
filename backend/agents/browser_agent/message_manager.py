"""
Browser Agent - Message Manager

Token-aware message management for LLM prompts.
Implements intelligent truncation and prioritization similar to browser-use.
"""

import logging
import tiktoken
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class MessagePriority(IntEnum):
    """Priority levels for messages - higher = more important"""
    LOW = 1        # Old successful steps
    NORMAL = 2     # Recent steps
    HIGH = 3       # Recent failures, observations
    CRITICAL = 4   # System prompt, current goal, extracted data


@dataclass
class Message:
    """A single message in the conversation"""
    role: Literal["system", "user", "assistant"]
    content: str
    priority: MessagePriority = MessagePriority.NORMAL
    token_count: int = 0
    step_number: Optional[int] = None
    is_failure: bool = False
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = MessageManager.count_tokens(self.content)


class MessageManager:
    """
    Token-aware message manager that intelligently truncates history
    to fit within model context limits while preserving important information.
    """
    
    # Default token budgets for different parts
    # REDUCED FURTHER: 30k still causes ~58k char prompts hitting timeouts
    DEFAULT_MAX_TOKENS = 15000  # Reduced from 30k - forces ~20k char prompts max
    SYSTEM_PROMPT_BUDGET = 2000  # Reserved for system prompt
    CURRENT_STATE_BUDGET = 3000  # Reserved for current page state (reduced from 5k)
    HISTORY_BUDGET_RATIO = 0.5  # 50% of remaining for history
    
    # Tokenizer - using cl100k_base (works with GPT-4, Claude, etc.)
    _encoder = None
    
    @classmethod
    def _get_encoder(cls):
        """Lazy load tokenizer"""
        if cls._encoder is None:
            try:
                cls._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Fallback: estimate 4 chars per token
                logger.warning("tiktoken not available, using char-based estimation")
                cls._encoder = None
        return cls._encoder
    
    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        encoder = cls._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        else:
            # Fallback: ~4 characters per token
            return len(text) // 4
    
    def __init__(
        self,
        max_total_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt_budget: int = SYSTEM_PROMPT_BUDGET,
        current_state_budget: int = CURRENT_STATE_BUDGET,
    ):
        self.max_total_tokens = max_total_tokens
        self.system_prompt_budget = system_prompt_budget
        self.current_state_budget = current_state_budget
        
        self.messages: List[Message] = []
        self.system_message: Optional[Message] = None
        
        # Calculate history budget
        remaining = max_total_tokens - system_prompt_budget - current_state_budget
        self.history_budget = int(remaining * self.HISTORY_BUDGET_RATIO)
        
        logger.debug(f"MessageManager initialized: total={max_total_tokens}, "
                    f"system={system_prompt_budget}, state={current_state_budget}, "
                    f"history={self.history_budget}")
    
    def set_system_prompt(self, content: str) -> int:
        """Set the system prompt (always included)"""
        tokens = self.count_tokens(content)
        self.system_message = Message(
            role="system",
            content=content,
            priority=MessagePriority.CRITICAL,
            token_count=tokens
        )
        return tokens
    
    def add_step(
        self,
        step_number: int,
        action_names: List[str],
        reasoning: str,
        result_success: bool,
        result_message: str,
        url: Optional[str] = None,
        extracted_data: Optional[Dict] = None,
        observation: Optional[str] = None  # What was seen on page
    ):
        """Add a step to history with automatic priority assignment"""
        # Build step content
        status = "✅" if result_success else "🛑 FAILED"
        content_parts = [
            f"Step {step_number}: {', '.join(action_names)}",
            f"Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"Reasoning: {reasoning}",
            f"Result: {status} - {result_message[:150]}",
        ]
        if url:
            content_parts.append(f"URL: {url[:80]}...")
        if observation:
            content_parts.append(f"Page: {observation[:150]}")  # Key page observations
        if extracted_data:
            content_parts.append(f"Extracted: {str(extracted_data)[:200]}")
        
        content = "\n".join(content_parts)
        
        # Assign priority
        if not result_success:
            priority = MessagePriority.HIGH  # Failures are important
        elif extracted_data:
            priority = MessagePriority.HIGH  # Extracted data is important
        else:
            priority = MessagePriority.NORMAL
        
        message = Message(
            role="assistant",
            content=content,
            priority=priority,
            step_number=step_number,
            is_failure=not result_success
        )
        
        self.messages.append(message)
        return message.token_count
    
    def add_observation(self, key: str, value: str, priority: MessagePriority = MessagePriority.HIGH):
        """Add an observation (key learning)"""
        content = f"OBSERVATION [{key}]: {value}"
        message = Message(
            role="assistant",
            content=content,
            priority=priority
        )
        self.messages.append(message)
        return message.token_count
    
    def get_history_for_prompt(self, budget_override: Optional[int] = None) -> str:
        """
        Get history formatted for prompt, intelligently truncated to fit budget.
        
        Strategy:
        1. Always include last 3 steps (most recent context)
        2. Always include all failures (learning from mistakes)
        3. Always include observations (key learnings)
        4. Fill remaining budget with older steps by priority
        """
        budget = budget_override or self.history_budget
        
        if not self.messages:
            return "No previous actions yet."
        
        # Categorize messages
        must_include = []  # Always include these
        optional = []      # Include if budget allows
        
        # Separate by importance
        for i, msg in enumerate(reversed(self.messages)):
            original_idx = len(self.messages) - 1 - i
            
            # Last 3 steps are always included
            if i < 3:
                must_include.append((original_idx, msg))
            # Failures are always included
            elif msg.is_failure:
                must_include.append((original_idx, msg))
            # High priority always included
            elif msg.priority >= MessagePriority.HIGH:
                must_include.append((original_idx, msg))
            else:
                optional.append((original_idx, msg))
        
        # Sort must_include by original order
        must_include.sort(key=lambda x: x[0])
        
        # Calculate must-have tokens
        must_have_tokens = sum(msg.token_count for _, msg in must_include)
        
        # If must-haves exceed budget, truncate older ones
        if must_have_tokens > budget:
            logger.warning(f"Must-have messages ({must_have_tokens} tokens) exceed budget ({budget})")
            # Keep only the most recent must-haves
            trimmed = []
            running_tokens = 0
            for idx, msg in reversed(must_include):
                if running_tokens + msg.token_count <= budget:
                    trimmed.insert(0, (idx, msg))
                    running_tokens += msg.token_count
            must_include = trimmed
            must_have_tokens = running_tokens
        
        # Fill remaining budget with optional messages (by priority, then recency)
        remaining_budget = budget - must_have_tokens
        optional.sort(key=lambda x: (x[1].priority, x[0]), reverse=True)  # Higher priority first, then more recent
        
        selected_optional = []
        for idx, msg in optional:
            if msg.token_count <= remaining_budget:
                selected_optional.append((idx, msg))
                remaining_budget -= msg.token_count
        
        # Combine and sort by original order
        all_selected = must_include + selected_optional
        all_selected.sort(key=lambda x: x[0])
        
        # Format for prompt
        if not all_selected:
            return "No previous actions yet."
        
        lines = ["PREVIOUS ACTIONS:"]
        for idx, msg in all_selected:
            lines.append(msg.content)
            lines.append("")  # Blank line between steps
        
        result = "\n".join(lines)
        
        # Log stats
        total_tokens = sum(msg.token_count for _, msg in all_selected)
        logger.debug(f"History: {len(all_selected)}/{len(self.messages)} messages, "
                    f"{total_tokens}/{budget} tokens")
        
        return result
    
    def get_token_stats(self) -> Dict[str, int]:
        """Get current token usage statistics"""
        system_tokens = self.system_message.token_count if self.system_message else 0
        history_tokens = sum(m.token_count for m in self.messages)
        
        return {
            "system_prompt": system_tokens,
            "history_total": history_tokens,
            "history_messages": len(self.messages),
            "budget_remaining": self.history_budget - history_tokens,
            "failures_count": sum(1 for m in self.messages if m.is_failure),
        }
    
    def clear(self):
        """Clear all messages except system prompt"""
        self.messages = []
    
    def trim_old_messages(self, keep_recent: int = 30):
        """Remove oldest low-priority messages if history grows too large"""
        if len(self.messages) <= keep_recent:
            return 0
        
        # Keep recent messages
        recent = self.messages[-keep_recent:]
        older = self.messages[:-keep_recent]
        
        # From older, keep only high-priority
        kept_older = [m for m in older if m.priority >= MessagePriority.HIGH]
        
        removed_count = len(self.messages) - len(kept_older) - len(recent)
        self.messages = kept_older + recent
        
        if removed_count > 0:
            logger.info(f"Trimmed {removed_count} old low-priority messages")
        
        return removed_count


def format_page_content_for_prompt(
    page_content: Dict[str, Any],
    max_tokens: int = 8000
) -> str:
    """
    Format page content for LLM prompt with token-aware truncation.
    Prioritizes: URL, title, interactive elements over full text.
    """
    encoder = MessageManager._get_encoder()
    
    def count(text: str) -> int:
        if encoder:
            return len(encoder.encode(text))
        return len(text) // 4
    
    parts = []
    current_tokens = 0
    
    # Always include URL and title (critical)
    url = page_content.get('url', 'unknown')
    title = page_content.get('title', 'unknown')
    header = f"URL: {url}\nTITLE: {title}\n"
    parts.append(header)
    current_tokens += count(header)
    
    # Scroll info (small, useful)
    scroll_info = page_content.get('scroll_info', {})
    if scroll_info:
        scroll_text = f"SCROLL: {scroll_info.get('scroll_percent', 0):.0f}% down\n"
        parts.append(scroll_text)
        current_tokens += count(scroll_text)
    
    # Interactive elements (high priority - this is what the agent clicks)
    elements = page_content.get('elements', [])
    if elements:
        elements_header = "\nPAGE ELEMENTS (Interactive & Text):\n"
        parts.append(elements_header)
        current_tokens += count(elements_header)
        
        # Reserve 60% of remaining budget for elements
        element_budget = int((max_tokens - current_tokens) * 0.6)
        element_lines = []
        element_tokens = 0
        
        for i, elem in enumerate(elements):
            role = elem.get('role', 'element')
            name = elem.get('name', '')[:60]
            # XPath removed to save tokens (20k+ chars)
            # The LLM acts by index, so XPath is unnecessary payload
            # xpath = elem.get('xpath', '')[:80]
            
            # Markers
            interactive_mark = "[INT]" if elem.get('interactive') else ""
            offscreen_mark = "[OFF-SCREEN]" if elem.get('visible') is False else ""
            
            # Format attributes compactly
            attrs = elem.get('attributes', {})
            attr_parts = []
            if attrs.get('href'): attr_parts.append(f"href='{attrs['href'][:40]}...'")
            if attrs.get('value'): attr_parts.append(f"val='{attrs['value'][:20]}'")
            if attrs.get('type'): attr_parts.append(f"type='{attrs['type']}'")
            if attrs.get('checked'): attr_parts.append("checked")
            if attrs.get('src'): attr_parts.append(f"src='{attrs['src'][:30]}...'")
            if attrs.get('title'): attr_parts.append(f"title='{attrs['title'][:30]}'")
            
            attr_str = " ".join(attr_parts)
            if attr_str: attr_str = f" {{{attr_str}}}"
            
            # Format: #Index [Mark] [Role] "Name" {Attrs}
            elem_text = f"#{i+1} {interactive_mark} {offscreen_mark} [{role}] \"{name}\"{attr_str}\n"
            elem_tok = count(elem_text)
            
            if element_tokens + elem_tok > element_budget:
                element_lines.append(f"... and {len(elements) - i} more elements (truncated)\n")
                break
            
            element_lines.append(elem_text)
            element_tokens += elem_tok
        
        parts.append("".join(element_lines))
        current_tokens += element_tokens
    
    # Accessibility tree - REMOVED for efficiency
    # It duplicates the element list and bloats the prompt, causing rate limits.
    # We rely on 'INTERACTIVE ELEMENTS' and visual context instead.
    # ax_tree = page_content.get('accessibility_tree', '')
    # if ax_tree...
    
    return "".join(parts)
