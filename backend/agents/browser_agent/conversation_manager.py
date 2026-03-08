"""
Browser Agent - Conversation Manager

Manages a persistent multi-turn conversation thread between the agent and LLM.
Replaces the stateless single-prompt approach with true conversational context,
so the LLM can see its own previous reasoning and action results.

When the conversation exceeds token limits, older turns are structurally
summarized (preserving metadata like action types, extracted data, patterns)
rather than dropped.
"""

import logging
import tiktoken
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages a multi-turn conversation thread for the browser agent.
    
    Instead of building a fresh prompt each step, this maintains a persistent
    message list that accumulates context. When token limits are approached,
    older turns are summarized into a structured summary that preserves
    all critical metadata.
    """
    
    MAX_CONVERSATION_TOKENS = 28000   # Total budget for conversation
    SUMMARIZE_THRESHOLD = 25000       # Trigger summarization at ~89% (raised to avoid over-summarizing)
    KEEP_RECENT_TURNS = 6             # Keep last N turn-pairs verbatim (lower = fewer summarization events, each does more)
    MAX_SUMMARY_TOKENS = 3000         # Cap summary block to prevent infinite growth
    SUMMARIZE_COOLDOWN = 3            # Minimum turns between summarizations
    
    # Tokenizer
    _encoder = None
    
    @classmethod
    def _get_encoder(cls):
        if cls._encoder is None:
            try:
                cls._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                cls._encoder = None
        return cls._encoder
    
    @classmethod
    def count_tokens(cls, text: str) -> int:
        if not text:
            return 0
        encoder = cls._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        return len(text) // 4  # Fallback
    
    def __init__(self):
        self.system_message: Optional[SystemMessage] = None
        self.turns: List[Dict[str, Any]] = []  # Each turn = {human: str, ai: str, step: int, metadata: {}}
        self.summary_block: Optional[str] = None  # Structured summary of older turns
        self.total_tokens: int = 0
        self.data_inventory: Dict[str, Any] = {}  # Running extracted data
        self._system_tokens: int = 0
        self._last_summarize_turn: int = 0  # Cooldown tracker
        
    def reset(self):
        """Reset the conversation for a new task (keeps system prompt)."""
        self.turns = []
        self.summary_block = None
        self.total_tokens = 0
        self.data_inventory = {}
        self._last_summarize_turn = 0
        logger.info("🔄 ConversationManager reset for new task")

    def set_system_prompt(self, prompt: str):
        """Set the system prompt (called once at start)."""
        self.system_message = SystemMessage(content=prompt)
    def build_state_message(
        self,
        step: int,
        page_content: Dict[str, Any],
        extracted_items: List[Dict],
        task: str,
        prev_result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        screenshot_b64: Optional[str] = None,
        agent_memory: str = "",
    ) -> str:
        """
        Build the HumanMessage content for the current step.
        Includes page state, previous action result, and running data inventory.
        """
        parts = []
        
        # === Task context ===
        parts.append(
            f"🎯 CURRENT TASK & CONTEXT:\n{task}\n"
        )
        
        # === Persistent Memory ===
        if agent_memory and str(agent_memory).strip() not in ('', 'None', 'UNKNOWN'):
            parts.append(f"🧠 YOUR MEMORY (From previous step):\n{agent_memory}\n")
        
        # === Previous action result (if any) ===
        if prev_result:
            result_section = self._format_action_result(prev_result)
            if result_section:
                parts.append(result_section)
            
            # FIX D: Inject last run_js result PROMINENTLY so LLM cannot ignore it
            if prev_result.get('action') == 'run_js' and prev_result.get('data'):
                js_data = prev_result['data'].get('result', '')
                if js_data and str(js_data).strip() not in ('', '[]', '{}', 'null', 'None'):
                    parts.append(
                        f"\n📊 LAST run_js OUTPUT (USE THIS DATA — do NOT re-extract!):\n"
                        f"{str(js_data)[:500]}\n"
                        f"→ If this contains data you need, call save_info for each field + done immediately."
                    )
        
        # === Error context ===
        if error:
            parts.append(
                f"⚠️ YOUR PREVIOUS ACTION FAILED:\n{error}\n"
                "DO NOT repeat the same approach. Try a completely different method.\n"
                "If run_js failed, use save_info instead. If click failed, try run_js."
            )
        
        # === Current state header ===
        url = page_content.get('url', 'about:blank')
        title = page_content.get('title', '(loading...)')
        scroll_info = page_content.get('scroll_info', {})
        scroll_pct = scroll_info.get('scroll_percent', 0)
        
        parts.append(
            f"═══════════════════════════════════════════════════\n"
            f"📍 Step {step} | URL: {url}\n"
            f"📄 Title: {title}\n"
            f"📜 Scroll: {scroll_pct:.0f}%"
        )
        
        # === FIX B: Task-aware completion check ===
        # Parse task for required fields and show explicit ✅/❌ checklist
        task_lower = task.lower()
        KNOWN_FIELDS = ['ram', 'storage', 'display', 'screen size', 'battery', 'price', 
                        'camera', 'processor', 'weight', 'dimensions', 'color', 'os']
        required_fields = [f for f in KNOWN_FIELDS if f in task_lower]
        
        # Build set of saved data keys (include auto_ keys by stripping prefix!)
        saved_keys_lower = set()
        for item in extracted_items:
            key = item.get('structured_info', {}).get('key', '').lower()
            if key:
                saved_keys_lower.add(key)
                # Strip auto_ prefix so 'auto_display' matches 'display'
                if key.startswith('auto_'):
                    saved_keys_lower.add(key[5:])  # 'auto_display' → 'display'
        # Also include extracted_data keys from the data inventory
        for k in self.data_inventory:
            saved_keys_lower.add(k.lower())
        
        if extracted_items:
            data_lines = [f"📦 Data collected ({len(extracted_items)} items):"]
            seen_keys = set()
            for item in extracted_items[-10:]:
                info = item.get('structured_info', item)
                if isinstance(info, dict):
                    key = info.get('key', '?')
                    # Deduplicate: skip if we already showed this key OR its non-auto version
                    display_key = key[5:] if key.startswith('auto_') else key
                    if display_key in seen_keys or key.startswith('js_result_'):
                        continue
                    seen_keys.add(display_key)
                    value = str(info.get('value', ''))[:60]
                    tag = "[auto]" if key.startswith('auto_') else ("✅" if info.get('verified') else "⚠️")
                    data_lines.append(f"  {tag} {display_key}: {value}")
            parts.append("\n".join(data_lines))
            
            # Show task requirements checklist
            if required_fields:
                checklist = []
                for field in required_fields:
                    found = any(field in k for k in saved_keys_lower)
                    checklist.append(f"  {'✅' if found else '❌'} {field}")
                missing = [f for f in required_fields if not any(f in k for k in saved_keys_lower)]
                
                parts.append(
                    "\n🎯 TASK REQUIREMENTS:\n" + "\n".join(checklist) +
                    (f"\n→ Missing: {', '.join(missing)}. Extract these and call done."
                     if missing else
                     "\n→ ✅ ALL FIELDS FOUND. Call `done` NOW! Do NOT continue browsing.")
                )
            elif len(extracted_items) >= 2:
                parts.append(
                    f"\n🔔 You have {len(extracted_items)} data items. "
                    f"If you have what the task needs → call `done` immediately."
                )
        elif step >= 5:
            parts.append(
                f"⚠️ NO DATA SAVED after {step} steps. "
                f"Use run_js or save_info NOW, then call done."
            )
        
        # === Title data extraction - parse product specs from page title ===
        import re
        title_specs = {}
        if title and len(title) > 20:
            ram_m = re.search(r'(\d+)\s*GB\s*RAM', title, re.I)
            if ram_m: title_specs['RAM'] = f"{ram_m.group(1)}GB"
            storage_m = re.search(r'(\d+)\s*GB\s*Storage', title, re.I)
            if storage_m: title_specs['Storage'] = f"{storage_m.group(1)}GB"
            battery_m = re.search(r'(\d+)\s*mAh', title, re.I)
            if battery_m: title_specs['Battery'] = f"{battery_m.group(1)}mAh"
            display_m = re.search(r'(\d+\.?\d*)\s*(?:inch|")', title, re.I)
            if display_m: title_specs['Display'] = f"{display_m.group(1)} inches"
        
        if title_specs:
            # Check which ones aren't saved yet
            unsaved_specs = {k: v for k, v in title_specs.items() if k.lower() not in saved_keys_lower}
            if unsaved_specs:
                specs_str = ", ".join(f"{k}={v}" for k, v in unsaved_specs.items())
                parts.append(
                    f"\n📋 DATA IN PAGE TITLE (save these NOW with save_info):\n"
                    f"  {specs_str}\n"
                    f"→ Call save_info for each, then done. No need to scroll or interact."
                )
        
        # === Overlay info (NON-TRAPPING — don't force dismiss) ===
        overlays = page_content.get('overlays', {})
        if overlays and overlays.get('has_overlay'):
            overlay_items = overlays.get('overlays', [])
            close_btns = overlays.get('close_buttons', [])
            
            # Count if we have no data saved — if so, prioritize extraction over dismissal
            has_saved_data = bool(extracted_items)
            
            if has_saved_data or step < 5:
                # Normal overlay: try to dismiss it
                overlay_text = "\n⚠️ OVERLAY/MODAL DETECTED. Try dismissing it once.\n"
                for ov in overlay_items[:2]:
                    ov_title = ov.get('title', 'Untitled')
                    overlay_text += f"  📋 Modal: \"{ov_title}\"\n"
                if close_btns:
                    overlay_text += f"  🔘 Close: \"{close_btns[0].get('text', 'X')}\"\n"
                overlay_text += "  → If dismissing fails, IGNORE the overlay and use run_js to extract data directly."
            else:
                # No data and stuck with overlay — DON'T tell agent to dismiss
                overlay_text = (
                    "\n⚠️ OVERLAY BLOCKING PAGE — do NOT keep trying to close it.\n"
                    "  → Use run_js to extract data from the page behind the overlay.\n"
                    "  → Or save data from the page TITLE above with save_info.\n"
                    "  → Then call done. Do NOT waste more steps on the overlay."
                )
            parts.append(overlay_text)
        
        # === Selector hints ===
        hints = page_content.get('selector_hints')
        if hints:
            hint_lines = ["🔍 Selector Hints:"]
            if isinstance(hints, dict):
                for pattern, count in list(hints.items())[:5]:
                    hint_lines.append(f"  • {pattern} ({count} items)")
            parts.append("\n".join(hint_lines))
        
        # === Page content (DOM tree) ===
        page_tree = page_content.get('unified_tree', '') or page_content.get('page_tree', '')
        if page_tree:
            parts.append(
                f"═══════════════════════════════════════════════════\n"
                f"📄 PAGE ELEMENTS (use #N index to click)\n"
                f"═══════════════════════════════════════════════════\n"
                f"{page_tree}"
            )
        else:
            parts.append("(no elements detected)")
        
        # === Screenshot hint ===
        if screenshot_b64:
            parts.append(
                "\n📸 SCREENSHOT ATTACHED — Examine it FIRST before reading the DOM tree. "
                "Look for: overlays/modals blocking the page, popups, cookie banners. "
                "If ANY overlay is visible, your FIRST action MUST dismiss it."
            )
        
        parts.append("\nNOW RESPOND WITH YOUR ACTION (JSON ONLY):")
        
        return "\n\n".join(parts)
    
    def _format_action_result(self, result: Dict[str, Any]) -> str:
        """Format the previous action's result for inclusion in the next turn."""
        action_name = result.get('action', 'unknown')
        success = result.get('success', False)
        message = result.get('message', '')[:200]
        data = result.get('data')
        url_changed = result.get('url_changed', False)
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        lines = [
            "📋 RESULT OF YOUR PREVIOUS ACTION:",
            f"  Action: {action_name} → {status}",
            f"  Detail: {message}",
        ]
        
        if url_changed:
            lines.append(f"  🔗 URL changed: {result.get('new_url', '?')[:80]}")
        elif success:
            lines.append("  🔗 URL: unchanged")
        
        if data:
            # INCREASED TRUNCATION LIMIT: In pure tool-calling, the LLM needs to read the full output
            # of run_js in the conversation history to decide what to save explicitly via save_info.
            data_str = json.dumps(data, default=str, ensure_ascii=False)[:4000]
            lines.append(f"  📦 Data returned: {data_str}")
        elif success and action_name in ('run_js', 'save_info', 'extract'):
            lines.append("  📦 No new data extracted")
        
        return "\n".join(lines)
    
    def add_turn(
        self,
        step: int,
        human_content: str,
        ai_response: str,
        action_name: str = "unknown",
        success: bool = True,
        data_extracted: Optional[Dict] = None,
        url: str = "",
        url_changed: bool = False,
    ):
        """Record a complete turn (human state + AI response + metadata).
        
        IMPORTANT: The human_content is compacted before storage to remove
        DOM trees and screenshot hints that bloat context. Only the current
        step (built fresh by build_state_message) gets the full DOM.
        """
        # COMPACT: Strip DOM trees and screenshot hints from stored content.
        # Old DOM trees are useless — the LLM needs history of ACTIONS + RESULTS,
        # not stale page structures it can no longer interact with.
        compact_content = self._compact_human_content(human_content)
        
        human_tokens = self.count_tokens(compact_content)
        ai_tokens = self.count_tokens(ai_response)
        
        # Extract action params from AI response for richer metadata
        action_params = ""
        failure_reason = ""
        try:
            ai_json = json.loads(ai_response)
            actions = ai_json.get('actions', [])
            if actions:
                params_list = []
                for a in actions[:3]:  # Max 3 actions
                    a_name = a.get('name', '?')
                    a_params = a.get('params', {})
                    # Compact param representation
                    if a_name == 'click':
                        target = a_params.get('text', a_params.get('index', a_params.get('xpath', '?')))
                        params_list.append(f"click({str(target)[:60]})")
                    elif a_name == 'run_js':
                        code = str(a_params.get('code', a_params.get('javascript', '?')))[:80]
                        params_list.append(f"run_js({code})")
                    elif a_name == 'navigate':
                        params_list.append(f"navigate({a_params.get('url', '?')[:60]})")
                    elif a_name == 'type':
                        params_list.append(f"type({a_params.get('text', '?')[:40]})")
                    elif a_name == 'scroll':
                        params_list.append(f"scroll({a_params.get('direction', '?')})")
                    elif a_name in ('save_info', 'done', 'skip_subtask'):
                        params_list.append(a_name)
                    else:
                        params_list.append(f"{a_name}({str(a_params)[:40]})")
                action_params = " → ".join(params_list)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        
        turn = {
            'step': step,
            'human': compact_content,
            'ai': ai_response,
            'tokens': human_tokens + ai_tokens,
            'metadata': {
                'action': action_name,
                'action_params': action_params,  # NEW: what exactly was tried
                'success': success,
                'data_extracted': data_extracted,
                'url': url[:100],
                'url_changed': url_changed,
                'failure_reason': failure_reason,  # Populated later by update_last_turn_result
            }
        }
        self.turns.append(turn)
        self.total_tokens += turn['tokens']
        
        # Update data inventory
        if data_extracted:
            self.data_inventory.update(data_extracted)
        
        logger.debug(
            f"📝 Turn {step} recorded ({turn['tokens']} tokens, "
            f"total: {self.total_tokens}/{self.MAX_CONVERSATION_TOKENS})"
        )
    
    def get_messages(self, screenshot_b64: Optional[str] = None) -> List[BaseMessage]:
        """
        Build the full message list for the LLM call.
        
        Structure:
        1. SystemMessage (always)
        2. [Optional] Summary of older turns (if summarized)
        3. Recent turns as Human/AI pairs
        4. Current state as final HumanMessage (added by caller)
        """
        messages = []
        
        # 1. System prompt
        if self.system_message:
            messages.append(self.system_message)
        
        # 2. Summary of older turns (if any)
        if self.summary_block:
            messages.append(HumanMessage(
                content=f"═══ CONVERSATION HISTORY (summarized) ═══\n{self.summary_block}"
            ))
            messages.append(AIMessage(
                content='{"reasoning": "Acknowledged conversation history summary.", '
                        '"actions": [], "confidence": 1.0}'
            ))
        
        # 3. Recent turns as Human/AI message pairs
        for turn in self.turns:
            messages.append(HumanMessage(content=turn['human']))
            messages.append(AIMessage(content=turn['ai']))
        
        return messages
    
    def needs_summarization(self) -> bool:
        """Check if we need to summarize older turns (with cooldown guard)."""
        # Cooldown: don't summarize if we just did recently
        current_turn = self.turns[-1]['step'] if self.turns else 0
        if current_turn - self._last_summarize_turn < self.SUMMARIZE_COOLDOWN:
            return False
        
        effective_tokens = self._system_tokens + self.total_tokens
        if self.summary_block:
            effective_tokens += self.count_tokens(self.summary_block)
        return effective_tokens > self.SUMMARIZE_THRESHOLD and len(self.turns) > self.KEEP_RECENT_TURNS
    
    def summarize_older_turns(self):
        """
        Summarize older turns into a structured summary block.
        Preserves all critical metadata: actions, results, data, patterns.
        This is a LOCAL operation (no LLM call) — fast and deterministic.
        """
        if len(self.turns) <= self.KEEP_RECENT_TURNS:
            return  # Nothing to summarize
        
        # Split: older turns → summarize, recent turns → keep
        older = self.turns[:-self.KEEP_RECENT_TURNS]
        recent = self.turns[-self.KEEP_RECENT_TURNS:]
        
        # Build structured summary preserving metadata
        summary = self._build_structured_summary(older)
        
        # Merge with existing summary (if re-summarizing)
        if self.summary_block:
            self.summary_block = self.summary_block + "\n\n" + summary
        else:
            self.summary_block = summary
        
        # CAP summary block to prevent infinite growth
        summary_tokens = self.count_tokens(self.summary_block)
        if summary_tokens > self.MAX_SUMMARY_TOKENS:
            # Keep only the most recent summary sections (split by section markers)
            sections = self.summary_block.split("── Steps")
            if len(sections) > 2:
                # Keep header + last 2 sections
                self.summary_block = "── Steps".join(sections[-2:])
                logger.info(f"✂️ Trimmed summary block from {summary_tokens} to ~{self.count_tokens(self.summary_block)} tokens")
        
        # Keep only recent turns
        old_tokens = sum(t['tokens'] for t in older)
        self.turns = recent
        self.total_tokens -= old_tokens
        
        # Update cooldown tracker
        self._last_summarize_turn = recent[0]['step'] if recent else 0
        
        # Add back summary tokens
        summary_tokens = self.count_tokens(self.summary_block)
        
        logger.info(
            f"📝 Summarized {len(older)} turns → saved ~{old_tokens - summary_tokens} tokens "
            f"(summary: {summary_tokens} tokens, remaining turns: {len(recent)})"
        )
    
    def _build_structured_summary(self, turns: List[Dict]) -> str:
        """
        Build a structured summary of turns that preserves:
        - Action type, target, and PARAMS (so the LLM knows exactly what was tried)
        - Success/failure with reason
        - Data extracted (key-value)
        - URL changes
        - Detected repetition patterns
        """
        if not turns:
            return ""
        
        step_range = f"Steps {turns[0]['step']}-{turns[-1]['step']}"
        lines = [f"── {step_range} ──"]
        
        # Track patterns for loop detection
        action_counts: Dict[str, int] = {}
        failed_actions: List[str] = []
        data_collected: Dict[str, str] = {}
        
        for turn in turns:
            meta = turn.get('metadata', {})
            action = meta.get('action', '?')
            action_params = meta.get('action_params', '')  # NEW: actual params
            success = meta.get('success', True)
            meta.get('url', '')
            url_changed = meta.get('url_changed', False)
            data = meta.get('data_extracted')
            failure_reason = meta.get('failure_reason', '')
            
            # Count actions
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Track failures with WHAT specifically failed
            if not success:
                fail_detail = action_params or action
                if failure_reason:
                    fail_detail += f" — {failure_reason[:60]}"
                failed_actions.append(f"Step {turn['step']}: {fail_detail}")
            
            # Track data
            if data and isinstance(data, dict):
                for k, v in data.items():
                    if k not in ('url', 'step', 'action_type', 'llm_reasoning'):
                        data_collected[k] = str(v)[:50]
            
            # Build compact line WITH action params
            status = "✅" if success else "❌"
            url_marker = " (URL changed)" if url_changed else ""
            data_marker = ""
            if data and isinstance(data, dict):
                keys = [k for k in data.keys() if k not in ('url', 'step', 'action_type', 'llm_reasoning')]
                if keys:
                    data_marker = f" → DATA: {', '.join(keys[:3])}"
            
            # Use action_params (e.g. "click(Samsung Galaxy S25...)") instead of bare action name
            action_display = action_params if action_params else action
            fail_suffix = f" — {failure_reason[:50]}" if failure_reason and not success else ""
            
            lines.append(
                f"  Step {turn['step']}: {action_display} {status}{fail_suffix}{url_marker}{data_marker}"
            )
        
        # Detect and surface repetition patterns
        patterns = []
        for action, count in action_counts.items():
            if count >= 3:
                patterns.append(f"⚠️ PATTERN: '{action}' used {count}x in {step_range}")
        
        if failed_actions:
            lines.append(f"\n  ❌ FAILURES (DO NOT REPEAT THESE): {'; '.join(failed_actions[:5])}")
        
        if patterns:
            lines.append("\n  " + "\n  ".join(patterns))
        
        if data_collected:
            lines.append(f"\n  📦 DATA COLLECTED: {json.dumps(data_collected, ensure_ascii=False)}")
        
        return "\n".join(lines)
    
    def _compact_human_content(self, content: str) -> str:
        """Strip DOM trees and screenshot hints from stored content.
        
        Old DOM trees are USELESS in conversation history — the LLM can't
        interact with past page states. Keeping them wastes ~2000-3000 tokens
        per turn, forcing aggressive summarization that loses critical context.
        
        What we KEEP (matters for reasoning):
        - Step header (URL, title, subtask info)
        - Previous action result (✅/❌ + details)
        - Error context (failure warnings)
        - Data inventory (what was extracted)
        - Overlay info (modal warnings)
        - Selector hints
        
        What we STRIP (stale, useless for history):
        - The entire PAGE ELEMENTS section (DOM tree)
        - Screenshot hints (screenshots aren't replayed in history)
        - The "NOW RESPOND WITH YOUR ACTION" footer
        """
        
        lines = content.split('\n')
        result_lines = []
        in_dom_section = False
        
        for line in lines:
            # Detect START of DOM tree section
            if '📄 PAGE ELEMENTS' in line or '── INTERACTIVE ELEMENTS ──' in line:
                in_dom_section = True
                result_lines.append("  [DOM tree stripped — see current step for live elements]")
                continue
            
            # Detect END of DOM tree section (next major section marker or end)
            if in_dom_section:
                # These markers indicate we've left the DOM section
                if (line.startswith('📸 SCREENSHOT') or 
                    line.startswith('NOW RESPOND') or
                    line.startswith('═══') and '📄 PAGE ELEMENTS' not in line and 'Step' not in line):
                    in_dom_section = False
                    # Fall through to check if this line should also be stripped
                else:
                    continue  # Skip DOM tree lines
            
            # Strip screenshot hints (screenshots aren't replayed in history)
            if '📸 SCREENSHOT ATTACHED' in line:
                continue
            if 'Examine it FIRST before reading the DOM tree' in line:
                continue
            if 'your FIRST action MUST dismiss it' in line:
                continue
                
            # Strip the "NOW RESPOND" footer
            if 'NOW RESPOND WITH YOUR ACTION' in line:
                continue
            
            result_lines.append(line)
        
        compacted = '\n'.join(result_lines).strip()
        
        # Safety: ensure we didn't strip everything
        if len(compacted) < 50:
            # Something went wrong, return a minimal version
            return content[:500] + "\n[content truncated for history]"
        
        return compacted
    
    def reset(self):
        """Reset conversation state (e.g. for a new task)."""
        self.turns = []
        self.summary_block = None
        self.total_tokens = 0
        self.data_inventory = {}
        logger.info("🔄 Conversation reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        summary_tokens = self.count_tokens(self.summary_block) if self.summary_block else 0
        return {
            "total_turns": len(self.turns),
            "total_tokens": self._system_tokens + self.total_tokens + summary_tokens,
            "system_tokens": self._system_tokens,
            "turn_tokens": self.total_tokens,
            "summary_tokens": summary_tokens,
            "has_summary": self.summary_block is not None,
            "data_inventory_size": len(self.data_inventory),
            "budget_used_pct": (
                (self._system_tokens + self.total_tokens + summary_tokens) 
                / self.MAX_CONVERSATION_TOKENS * 100
            ),
        }
