"""
Automatic Artifact Integration for Orchestrator

This module provides automatic artifact management that integrates seamlessly
with the orchestrator graph. It automatically:

1. Compresses large task results to artifacts
2. Optimizes context before LLM calls
3. Stores canvas content as artifacts
4. Manages conversation history with artifact compression
5. Provides context-aware retrieval

The orchestrator doesn't need to explicitly call artifact functions -
this module wraps key operations to handle artifacts transparently.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

from services.artifact_service import (
    ArtifactStore,
    ArtifactType,
    ArtifactPriority,
    ArtifactReference,
    ArtifactMetadata,
    get_artifact_store,
    get_context_manager,
    store_artifact,
    retrieve_artifact
)
from services.context_optimizer import (
    ContextOptimizer,
    OptimizedContext,
    ContextPriority,
    get_context_optimizer
)

logger = logging.getLogger("ArtifactIntegration")


# ============================================================================
# CONFIGURATION
# ============================================================================

class ArtifactConfig:
    """Configuration for automatic artifact management"""
    
    # Size thresholds (in characters) for automatic artifact creation
    THRESHOLDS = {
        "task_result": 2000,        # Task results > 2KB
        "canvas_content": 500,      # Canvas always stored
        "screenshot": 100,          # Screenshots always stored (base64)
        "conversation": 5000,       # Conversation history > 5KB
        "agent_response": 3000,     # Raw agent responses > 3KB
    }
    
    # Fields to automatically compress in state
    AUTO_COMPRESS_FIELDS = [
        "canvas_content",
        "completed_tasks",
        "task_agent_pairs",
    ]
    
    # Maximum context tokens for LLM calls
    MAX_CONTEXT_TOKENS = 8000
    
    # Enable/disable automatic artifact management
    ENABLED = True
    
    # Log artifact operations
    VERBOSE_LOGGING = True


# ============================================================================
# AUTOMATIC ARTIFACT DECORATORS
# ============================================================================

def auto_artifact_result(func: Callable) -> Callable:
    """
    Decorator that automatically stores large task results as artifacts.
    
    Use on functions that return task execution results.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        
        if not ArtifactConfig.ENABLED:
            return result
        
        # Check if result should be stored as artifact
        if isinstance(result, dict):
            thread_id = kwargs.get('thread_id') or _extract_thread_id(args, kwargs)
            result = _compress_result_if_needed(result, thread_id)
        
        return result
    
    return wrapper


def auto_optimize_context(func: Callable) -> Callable:
    """
    Decorator that automatically optimizes context before LLM calls.
    
    Use on functions that prepare prompts for LLM.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not ArtifactConfig.ENABLED:
            return func(*args, **kwargs)
        
        # Extract state from args/kwargs
        state = _extract_state(args, kwargs)
        thread_id = _extract_thread_id(args, kwargs)
        
        if state and thread_id:
            # Optimize context
            optimized = get_context_optimizer().optimize_state_for_context(
                state=state,
                thread_id=thread_id,
                max_tokens=ArtifactConfig.MAX_CONTEXT_TOKENS
            )
            
            if ArtifactConfig.VERBOSE_LOGGING:
                logger.info(f"Context optimized: {optimized.total_tokens} tokens, "
                           f"saved {optimized.tokens_saved}, "
                           f"ratio {optimized.compression_ratio:.2%}")
            
            # Inject optimized context
            kwargs['_optimized_context'] = optimized
        
        return func(*args, **kwargs)
    
    return wrapper


# ============================================================================
# AUTOMATIC COMPRESSION FUNCTIONS
# ============================================================================

def compress_task_result(
    result: Dict[str, Any],
    task_name: str,
    thread_id: str
) -> Dict[str, Any]:
    """
    Automatically compress a task result if it exceeds threshold.
    
    Returns the result with large fields replaced by artifact references.
    """
    if not ArtifactConfig.ENABLED:
        return result
    
    result_str = json.dumps(result, default=str)
    
    if len(result_str) < ArtifactConfig.THRESHOLDS["task_result"]:
        return result
    
    # Store full result as artifact
    store = get_artifact_store()
    metadata = store.store(
        content=result,
        name=f"task_result_{task_name}",
        artifact_type=ArtifactType.RESULT,
        thread_id=thread_id,
        description=f"Result for task: {task_name}",
        priority=ArtifactPriority.MEDIUM,
        tags=["task_result", task_name]
    )
    
    if ArtifactConfig.VERBOSE_LOGGING:
        logger.info(f"Compressed task result '{task_name}' to artifact {metadata.id} "
                   f"({metadata.size_bytes} bytes)")
    
    # Return compressed version with reference
    return {
        "_artifact_ref": {
            "id": metadata.id,
            "type": "result",
            "summary": metadata.summary or f"Task result for {task_name}"
        },
        "task_name": task_name,
        "status": result.get("status", "completed"),
        "summary": _generate_result_summary(result)
    }


def compress_canvas_content(
    canvas_content: str,
    canvas_type: str,
    thread_id: str
) -> Dict[str, Any]:
    """
    Store canvas content as artifact and return reference.
    
    Canvas content is always stored as artifact since it's typically large.
    """
    if not ArtifactConfig.ENABLED or not canvas_content:
        return {"content": canvas_content, "type": canvas_type}
    
    store = get_artifact_store()
    metadata = store.store(
        content=canvas_content,
        name=f"canvas_{canvas_type}_{int(time.time())}",
        artifact_type=ArtifactType.CANVAS,
        thread_id=thread_id,
        description=f"{canvas_type.upper()} canvas content",
        priority=ArtifactPriority.HIGH,
        tags=["canvas", canvas_type]
    )
    
    if ArtifactConfig.VERBOSE_LOGGING:
        logger.info(f"Stored canvas as artifact {metadata.id} ({metadata.size_bytes} bytes)")
    
    return {
        "_artifact_ref": {
            "id": metadata.id,
            "type": "canvas",
            "canvas_type": canvas_type
        },
        "canvas_type": canvas_type,
        "preview": canvas_content[:200] + "..." if len(canvas_content) > 200 else canvas_content
    }


def compress_screenshot(
    screenshot_base64: str,
    step_name: str,
    thread_id: str
) -> Dict[str, Any]:
    """
    Store screenshot as artifact (base64 images are always large).
    """
    if not ArtifactConfig.ENABLED or not screenshot_base64:
        return {"base64": screenshot_base64}
    
    store = get_artifact_store()
    metadata = store.store(
        content=screenshot_base64,
        name=f"screenshot_{step_name}_{int(time.time())}",
        artifact_type=ArtifactType.SCREENSHOT,
        thread_id=thread_id,
        description=f"Screenshot from {step_name}",
        priority=ArtifactPriority.LOW,
        ttl_hours=24,  # Screenshots expire after 24 hours
        tags=["screenshot", step_name]
    )
    
    if ArtifactConfig.VERBOSE_LOGGING:
        logger.info(f"Stored screenshot as artifact {metadata.id} ({metadata.size_bytes} bytes)")
    
    return {
        "_artifact_ref": {
            "id": metadata.id,
            "type": "screenshot"
        },
        "step": step_name
    }


# ============================================================================
# STATE COMPRESSION FOR SAVING
# ============================================================================

def compress_state_for_saving(
    state: Dict[str, Any],
    thread_id: str
) -> Dict[str, Any]:
    """
    Compress a state dict before saving to conversation history.
    
    Large fields are stored as artifacts and replaced with references.
    """
    if not ArtifactConfig.ENABLED:
        return state
    
    compressed = dict(state)
    artifact_refs = {}
    
    # Compress canvas content
    if canvas := compressed.get("canvas_content"):
        if len(str(canvas)) > ArtifactConfig.THRESHOLDS["canvas_content"]:
            ref = compress_canvas_content(
                canvas_content=canvas,
                canvas_type=compressed.get("canvas_type", "html"),
                thread_id=thread_id
            )
            if "_artifact_ref" in ref:
                artifact_refs["canvas_content"] = ref["_artifact_ref"]
                compressed["canvas_content"] = f"[ARTIFACT:{ref['_artifact_ref']['id']}]"
    
    # Compress completed tasks
    if tasks := compressed.get("completed_tasks"):
        tasks_str = json.dumps(tasks, default=str)
        if len(tasks_str) > ArtifactConfig.THRESHOLDS["task_result"]:
            store = get_artifact_store()
            metadata = store.store(
                content=tasks,
                name=f"completed_tasks_{thread_id}",
                artifact_type=ArtifactType.RESULT,
                thread_id=thread_id,
                description=f"Completed tasks ({len(tasks)} tasks)",
                priority=ArtifactPriority.MEDIUM
            )
            artifact_refs["completed_tasks"] = {
                "id": metadata.id,
                "type": "result",
                "count": len(tasks)
            }
            # Keep summary in state
            compressed["completed_tasks"] = [
                {"task_name": t.get("task_name"), "status": "completed"}
                for t in tasks
            ]
    
    # Add artifact references
    if artifact_refs:
        compressed["_artifact_refs"] = artifact_refs
        if ArtifactConfig.VERBOSE_LOGGING:
            logger.info(f"Compressed {len(artifact_refs)} fields to artifacts for thread {thread_id}")
    
    return compressed


def expand_state_from_saved(
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Expand artifact references when loading a saved state.
    """
    if not ArtifactConfig.ENABLED:
        return state
    
    artifact_refs = state.pop("_artifact_refs", {})
    if not artifact_refs:
        return state
    
    expanded = dict(state)
    store = get_artifact_store()
    
    for field, ref_info in artifact_refs.items():
        artifact_id = ref_info.get("id")
        if not artifact_id:
            continue
        
        artifact = store.retrieve(artifact_id)
        if artifact:
            expanded[field] = artifact.content
            if ArtifactConfig.VERBOSE_LOGGING:
                logger.info(f"Expanded artifact {artifact_id} for field {field}")
        else:
            logger.warning(f"Failed to expand artifact {artifact_id} for field {field}")
    
    return expanded


# ============================================================================
# CONTEXT OPTIMIZATION FOR LLM CALLS
# ============================================================================

def get_optimized_llm_context(
    state: Dict[str, Any],
    thread_id: str,
    focus_on: List[str] = None,
    max_tokens: int = None
) -> Dict[str, Any]:
    """
    Get optimized context for an LLM call.
    
    Args:
        state: The current orchestrator state
        thread_id: Conversation thread ID
        focus_on: Fields to prioritize in context
        max_tokens: Maximum tokens (default from config)
    
    Returns:
        Dict with:
        - context: Optimized context string
        - artifact_refs: List of artifact references
        - tokens_used: Estimated tokens
        - tokens_saved: Tokens saved by compression
    """
    if not ArtifactConfig.ENABLED:
        # Return full state as context
        return {
            "context": json.dumps(state, default=str),
            "artifact_refs": [],
            "tokens_used": len(json.dumps(state, default=str)) // 4,
            "tokens_saved": 0
        }
    
    optimizer = get_context_optimizer()
    result = optimizer.optimize_state_for_context(
        state=state,
        thread_id=thread_id,
        max_tokens=max_tokens or ArtifactConfig.MAX_CONTEXT_TOKENS,
        focus_fields=focus_on or []
    )
    
    return {
        "context": result.user_context,
        "artifact_refs": result.artifact_references,
        "tokens_used": result.total_tokens,
        "tokens_saved": result.tokens_saved
    }


def build_prompt_with_artifacts(
    base_prompt: str,
    state: Dict[str, Any],
    thread_id: str,
    include_artifact_context: bool = True
) -> str:
    """
    Build a prompt with artifact-aware context.
    
    Automatically includes artifact references and summaries.
    """
    if not ArtifactConfig.ENABLED or not include_artifact_context:
        return base_prompt
    
    # Get optimized context
    optimized = get_optimized_llm_context(state, thread_id)
    
    # Build prompt with context
    artifact_section = ""
    if optimized["artifact_refs"]:
        artifact_section = "\n\n**Available Artifacts** (use ID to request full content):\n"
        for ref in optimized["artifact_refs"]:
            artifact_section += f"- [{ref['id']}] {ref.get('source', 'unknown')}\n"
    
    return f"""
{base_prompt}

**Context:**
{optimized['context']}
{artifact_section}
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_thread_id(args: tuple, kwargs: dict) -> Optional[str]:
    """Extract thread_id from function arguments"""
    if 'thread_id' in kwargs:
        return kwargs['thread_id']
    if 'config' in kwargs:
        return kwargs['config'].get('configurable', {}).get('thread_id')
    # Check args for state dict
    for arg in args:
        if isinstance(arg, dict):
            if 'thread_id' in arg:
                return arg['thread_id']
    return None


def _extract_state(args: tuple, kwargs: dict) -> Optional[Dict[str, Any]]:
    """Extract state from function arguments"""
    if 'state' in kwargs:
        return kwargs['state']
    # First dict arg is usually state
    for arg in args:
        if isinstance(arg, dict) and 'original_prompt' in arg:
            return arg
    return None


def _compress_result_if_needed(
    result: Dict[str, Any],
    thread_id: Optional[str]
) -> Dict[str, Any]:
    """Compress result dict if it exceeds threshold"""
    if not thread_id:
        return result
    
    result_str = json.dumps(result, default=str)
    if len(result_str) < ArtifactConfig.THRESHOLDS["task_result"]:
        return result
    
    task_name = result.get('task_name', 'unknown')
    return compress_task_result(result, task_name, thread_id)


def _generate_result_summary(result: Dict[str, Any]) -> str:
    """Generate a brief summary of a task result"""
    if isinstance(result, dict):
        status = result.get('status', 'completed')
        if 'error' in result:
            return f"Failed: {str(result['error'])[:100]}"
        if 'summary' in result:
            return result['summary'][:200]
        if 'result' in result:
            return f"Completed with result: {str(result['result'])[:100]}"
        return f"Status: {status}"
    return str(result)[:200]


# ============================================================================
# INTEGRATION HOOKS FOR ORCHESTRATOR
# ============================================================================

class OrchestratorArtifactHooks:
    """
    Hooks that can be called from orchestrator nodes to automatically
    manage artifacts.
    """
    
    @staticmethod
    def on_task_complete(
        task_name: str,
        result: Dict[str, Any],
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Called when a task completes. Automatically compresses large results.
        """
        return compress_task_result(result, task_name, thread_id)
    
    @staticmethod
    def on_canvas_generated(
        canvas_content: str,
        canvas_type: str,
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Called when canvas content is generated. Stores as artifact.
        """
        return compress_canvas_content(canvas_content, canvas_type, thread_id)
    
    @staticmethod
    def on_screenshot_captured(
        screenshot_base64: str,
        step_name: str,
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Called when a screenshot is captured. Stores as artifact.
        """
        return compress_screenshot(screenshot_base64, step_name, thread_id)
    
    @staticmethod
    def before_llm_call(
        state: Dict[str, Any],
        thread_id: str,
        focus_fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Called before an LLM call. Returns optimized context.
        """
        return get_optimized_llm_context(state, thread_id, focus_fields)
    
    @staticmethod
    def before_save(
        state: Dict[str, Any],
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Called before saving conversation. Compresses state.
        """
        return compress_state_for_saving(state, thread_id)
    
    @staticmethod
    def after_load(
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Called after loading conversation. Expands artifacts.
        """
        return expand_state_from_saved(state)


# Global hooks instance
hooks = OrchestratorArtifactHooks()
