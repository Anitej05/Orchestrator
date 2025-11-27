"""
Context Optimizer Service - Smart Context Window Management

This service optimizes context for LLM consumption by:
1. Prioritizing relevant information
2. Compressing verbose content
3. Managing artifact references
4. Implementing sliding window for conversation history
5. Semantic relevance scoring for context selection

Inspired by production systems like:
- Claude's artifact system
- ChatGPT's memory and context management
- Anthropic's context caching
"""

import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from services.artifact_service import (
    ArtifactStore,
    ArtifactContextManager,
    ArtifactType,
    ArtifactPriority,
    ArtifactReference,
    get_artifact_store,
    get_context_manager
)

logger = logging.getLogger("ContextOptimizer")


class ContextPriority(str, Enum):
    """Priority levels for context inclusion"""
    ESSENTIAL = "essential"      # Must include (current prompt, critical state)
    HIGH = "high"                # Should include (recent messages, active tasks)
    MEDIUM = "medium"            # Include if space (completed tasks, file refs)
    LOW = "low"                  # Include only if plenty of space
    OPTIONAL = "optional"        # Can be omitted


@dataclass
class ContextBlock:
    """A block of context with metadata"""
    content: str
    priority: ContextPriority
    token_estimate: int
    source: str  # Where this context came from
    artifact_id: Optional[str] = None  # If this is an artifact reference
    
    @property
    def is_artifact_ref(self) -> bool:
        return self.artifact_id is not None


@dataclass
class OptimizedContext:
    """Result of context optimization"""
    system_prompt: str
    user_context: str
    artifact_references: List[Dict[str, Any]]
    included_blocks: List[ContextBlock]
    excluded_blocks: List[ContextBlock]
    total_tokens: int
    tokens_saved: int
    compression_ratio: float


class ContextOptimizer:
    """
    Optimizes context for LLM consumption within token limits.
    """
    
    # Token estimation constants
    CHARS_PER_TOKEN = 4
    
    # Default token budgets
    DEFAULT_MAX_TOKENS = 8000
    SYSTEM_PROMPT_BUDGET = 1000
    USER_CONTEXT_BUDGET = 6000
    ARTIFACT_REF_BUDGET = 1000
    
    def __init__(
        self,
        artifact_store: Optional[ArtifactStore] = None,
        context_manager: Optional[ArtifactContextManager] = None
    ):
        self.store = artifact_store or get_artifact_store()
        self.context_manager = context_manager or get_context_manager()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(text) // self.CHARS_PER_TOKEN
    
    def optimize_state_for_context(
        self,
        state: Dict[str, Any],
        thread_id: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        focus_fields: List[str] = None
    ) -> OptimizedContext:
        """
        Optimize orchestrator state for LLM context.
        
        Args:
            state: The full orchestrator state
            thread_id: Conversation thread ID
            max_tokens: Maximum tokens for context
            focus_fields: Fields to prioritize in context
        
        Returns:
            OptimizedContext with optimized content
        """
        focus_fields = focus_fields or []
        blocks: List[ContextBlock] = []
        
        # 1. Extract essential context (always include)
        essential_blocks = self._extract_essential_context(state)
        blocks.extend(essential_blocks)
        
        # 2. Extract high-priority context
        high_priority_blocks = self._extract_high_priority_context(state, focus_fields)
        blocks.extend(high_priority_blocks)
        
        # 3. Extract medium-priority context (may be compressed)
        medium_blocks = self._extract_medium_priority_context(state, thread_id)
        blocks.extend(medium_blocks)
        
        # 4. Extract low-priority context (likely compressed or excluded)
        low_blocks = self._extract_low_priority_context(state, thread_id)
        blocks.extend(low_blocks)
        
        # 5. Select blocks within token budget
        included, excluded = self._select_blocks_within_budget(blocks, max_tokens)
        
        # 6. Build final context
        return self._build_optimized_context(included, excluded, thread_id)
    
    def _extract_essential_context(self, state: Dict[str, Any]) -> List[ContextBlock]:
        """Extract essential context that must always be included"""
        blocks = []
        
        # Original prompt
        if prompt := state.get("original_prompt"):
            blocks.append(ContextBlock(
                content=f"User Request: {prompt}",
                priority=ContextPriority.ESSENTIAL,
                token_estimate=self.estimate_tokens(prompt) + 10,
                source="original_prompt"
            ))
        
        # Current question for user (if any)
        if question := state.get("question_for_user"):
            blocks.append(ContextBlock(
                content=f"Pending Question: {question}",
                priority=ContextPriority.ESSENTIAL,
                token_estimate=self.estimate_tokens(question) + 10,
                source="question_for_user"
            ))
        
        # User response (if any)
        if response := state.get("user_response"):
            blocks.append(ContextBlock(
                content=f"User Response: {response}",
                priority=ContextPriority.ESSENTIAL,
                token_estimate=self.estimate_tokens(response) + 10,
                source="user_response"
            ))
        
        return blocks
    
    def _extract_high_priority_context(
        self,
        state: Dict[str, Any],
        focus_fields: List[str]
    ) -> List[ContextBlock]:
        """Extract high-priority context"""
        blocks = []
        
        # Recent messages (last 5)
        if messages := state.get("messages"):
            recent = messages[-5:] if len(messages) > 5 else messages
            for msg in recent:
                content = self._format_message(msg)
                blocks.append(ContextBlock(
                    content=content,
                    priority=ContextPriority.HIGH,
                    token_estimate=self.estimate_tokens(content),
                    source="recent_messages"
                ))
        
        # Parsed tasks (current work)
        if tasks := state.get("parsed_tasks"):
            task_summary = self._summarize_tasks(tasks)
            blocks.append(ContextBlock(
                content=f"Current Tasks:\n{task_summary}",
                priority=ContextPriority.HIGH,
                token_estimate=self.estimate_tokens(task_summary) + 10,
                source="parsed_tasks"
            ))
        
        # Focus fields requested by caller
        for field in focus_fields:
            if value := state.get(field):
                content = self._format_field(field, value)
                blocks.append(ContextBlock(
                    content=content,
                    priority=ContextPriority.HIGH,
                    token_estimate=self.estimate_tokens(content),
                    source=f"focus_{field}"
                ))
        
        return blocks
    
    def _extract_medium_priority_context(
        self,
        state: Dict[str, Any],
        thread_id: str
    ) -> List[ContextBlock]:
        """Extract medium-priority context, potentially as artifact references"""
        blocks = []
        
        # Completed tasks (summarized or as artifact)
        if completed := state.get("completed_tasks"):
            if len(completed) > 3:
                # Store as artifact, include reference
                ref = self.context_manager.compress_for_context(
                    content=completed,
                    name="completed_tasks",
                    thread_id=thread_id,
                    artifact_type=ArtifactType.RESULT
                )
                if isinstance(ref, ArtifactReference):
                    blocks.append(ContextBlock(
                        content=ref.to_context_string(),
                        priority=ContextPriority.MEDIUM,
                        token_estimate=self.estimate_tokens(ref.to_context_string()),
                        source="completed_tasks",
                        artifact_id=ref.id
                    ))
                else:
                    # Small enough to include inline
                    summary = self._summarize_completed_tasks(completed)
                    blocks.append(ContextBlock(
                        content=f"Completed Tasks:\n{summary}",
                        priority=ContextPriority.MEDIUM,
                        token_estimate=self.estimate_tokens(summary) + 10,
                        source="completed_tasks"
                    ))
            else:
                summary = self._summarize_completed_tasks(completed)
                blocks.append(ContextBlock(
                    content=f"Completed Tasks:\n{summary}",
                    priority=ContextPriority.MEDIUM,
                    token_estimate=self.estimate_tokens(summary) + 10,
                    source="completed_tasks"
                ))
        
        # Task-agent pairs (summarized)
        if pairs := state.get("task_agent_pairs"):
            summary = self._summarize_task_agent_pairs(pairs)
            blocks.append(ContextBlock(
                content=f"Assigned Agents:\n{summary}",
                priority=ContextPriority.MEDIUM,
                token_estimate=self.estimate_tokens(summary) + 10,
                source="task_agent_pairs"
            ))
        
        # Uploaded files (as references)
        if files := state.get("uploaded_files"):
            file_refs = self._format_file_references(files)
            blocks.append(ContextBlock(
                content=f"Uploaded Files:\n{file_refs}",
                priority=ContextPriority.MEDIUM,
                token_estimate=self.estimate_tokens(file_refs) + 10,
                source="uploaded_files"
            ))
        
        return blocks
    
    def _extract_low_priority_context(
        self,
        state: Dict[str, Any],
        thread_id: str
    ) -> List[ContextBlock]:
        """Extract low-priority context"""
        blocks = []
        
        # Canvas content (always as artifact reference)
        if canvas := state.get("canvas_content"):
            ref = self.context_manager.compress_for_context(
                content=canvas,
                name="canvas_content",
                thread_id=thread_id,
                artifact_type=ArtifactType.CANVAS,
                force_artifact=True  # Always store canvas as artifact
            )
            if isinstance(ref, ArtifactReference):
                blocks.append(ContextBlock(
                    content=ref.to_context_string(),
                    priority=ContextPriority.LOW,
                    token_estimate=self.estimate_tokens(ref.to_context_string()),
                    source="canvas_content",
                    artifact_id=ref.id
                ))
        
        # Older messages (beyond recent 5)
        if messages := state.get("messages"):
            if len(messages) > 5:
                older = messages[:-5]
                # Store as artifact
                ref = self.context_manager.compress_for_context(
                    content=older,
                    name="conversation_history",
                    thread_id=thread_id,
                    artifact_type=ArtifactType.CONVERSATION
                )
                if isinstance(ref, ArtifactReference):
                    blocks.append(ContextBlock(
                        content=f"[Earlier conversation: {len(older)} messages stored as artifact {ref.id}]",
                        priority=ContextPriority.LOW,
                        token_estimate=50,
                        source="older_messages",
                        artifact_id=ref.id
                    ))
        
        return blocks
    
    def _select_blocks_within_budget(
        self,
        blocks: List[ContextBlock],
        max_tokens: int
    ) -> Tuple[List[ContextBlock], List[ContextBlock]]:
        """Select blocks that fit within token budget"""
        # Sort by priority
        priority_order = {
            ContextPriority.ESSENTIAL: 0,
            ContextPriority.HIGH: 1,
            ContextPriority.MEDIUM: 2,
            ContextPriority.LOW: 3,
            ContextPriority.OPTIONAL: 4
        }
        sorted_blocks = sorted(blocks, key=lambda b: priority_order[b.priority])
        
        included = []
        excluded = []
        current_tokens = 0
        
        for block in sorted_blocks:
            if current_tokens + block.token_estimate <= max_tokens:
                included.append(block)
                current_tokens += block.token_estimate
            elif block.priority == ContextPriority.ESSENTIAL:
                # Essential blocks must be included even if over budget
                included.append(block)
                current_tokens += block.token_estimate
                logger.warning(f"Essential block exceeds budget: {block.source}")
            else:
                excluded.append(block)
        
        return included, excluded
    
    def _build_optimized_context(
        self,
        included: List[ContextBlock],
        excluded: List[ContextBlock],
        thread_id: str
    ) -> OptimizedContext:
        """Build the final optimized context"""
        # Separate artifact references
        artifact_refs = [
            {"id": b.artifact_id, "source": b.source}
            for b in included if b.is_artifact_ref
        ]
        
        # Build user context string
        context_parts = []
        for block in included:
            context_parts.append(block.content)
        
        user_context = "\n\n".join(context_parts)
        
        # Calculate metrics
        total_tokens = sum(b.token_estimate for b in included)
        excluded_tokens = sum(b.token_estimate for b in excluded)
        original_tokens = total_tokens + excluded_tokens
        compression_ratio = total_tokens / original_tokens if original_tokens > 0 else 1.0
        
        return OptimizedContext(
            system_prompt="",  # To be filled by caller
            user_context=user_context,
            artifact_references=artifact_refs,
            included_blocks=included,
            excluded_blocks=excluded,
            total_tokens=total_tokens,
            tokens_saved=excluded_tokens,
            compression_ratio=compression_ratio
        )
    
    # ========================================================================
    # FORMATTING HELPERS
    # ========================================================================
    
    def _format_message(self, msg: Any) -> str:
        """Format a message for context"""
        if isinstance(msg, dict):
            role = msg.get("type", msg.get("role", "unknown"))
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            role = msg.__class__.__name__.replace("Message", "").lower()
            content = msg.content
        else:
            return str(msg)[:200]
        
        # Truncate long messages
        if len(content) > 500:
            content = content[:500] + "... [truncated]"
        
        return f"[{role}]: {content}"
    
    def _summarize_tasks(self, tasks: List[Any]) -> str:
        """Summarize parsed tasks"""
        lines = []
        for i, task in enumerate(tasks, 1):
            if isinstance(task, dict):
                name = task.get("task_name", f"Task {i}")
                desc = task.get("task_description", "")[:100]
            else:
                name = getattr(task, "task_name", f"Task {i}")
                desc = getattr(task, "task_description", "")[:100]
            lines.append(f"{i}. {name}: {desc}")
        return "\n".join(lines)
    
    def _summarize_completed_tasks(self, tasks: List[Any]) -> str:
        """Summarize completed tasks"""
        lines = []
        for task in tasks:
            if isinstance(task, dict):
                name = task.get("task_name", "Unknown")
                result = str(task.get("result", ""))[:100]
            else:
                name = getattr(task, "task_name", "Unknown")
                result = str(getattr(task, "result", ""))[:100]
            lines.append(f"✓ {name}: {result}")
        return "\n".join(lines)
    
    def _summarize_task_agent_pairs(self, pairs: List[Any]) -> str:
        """Summarize task-agent assignments"""
        lines = []
        for pair in pairs:
            if isinstance(pair, dict):
                task = pair.get("task_name", "Unknown")
                agent = pair.get("primary", {})
                agent_name = agent.get("name", "Unknown") if isinstance(agent, dict) else "Unknown"
            else:
                task = getattr(pair, "task_name", "Unknown")
                agent = getattr(pair, "primary", None)
                agent_name = getattr(agent, "name", "Unknown") if agent else "Unknown"
            lines.append(f"• {task} → {agent_name}")
        return "\n".join(lines)
    
    def _format_file_references(self, files: List[Any]) -> str:
        """Format file references"""
        lines = []
        for f in files:
            if isinstance(f, dict):
                name = f.get("file_name", "Unknown")
                ftype = f.get("file_type", "file")
            else:
                name = getattr(f, "file_name", "Unknown")
                ftype = getattr(f, "file_type", "file")
            lines.append(f"📎 {name} ({ftype})")
        return "\n".join(lines)
    
    def _format_field(self, field: str, value: Any) -> str:
        """Format a generic field for context"""
        if isinstance(value, str):
            return f"{field}: {value[:500]}"
        elif isinstance(value, (list, dict)):
            json_str = json.dumps(value, default=str)[:500]
            return f"{field}: {json_str}"
        return f"{field}: {str(value)[:500]}"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_optimizer: Optional[ContextOptimizer] = None


def get_context_optimizer() -> ContextOptimizer:
    """Get the global context optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ContextOptimizer()
    return _optimizer


def optimize_context(
    state: Dict[str, Any],
    thread_id: str,
    max_tokens: int = 8000
) -> OptimizedContext:
    """Convenience function to optimize context"""
    return get_context_optimizer().optimize_state_for_context(
        state=state,
        thread_id=thread_id,
        max_tokens=max_tokens
    )
