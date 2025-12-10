"""
Conversation Artifact Integration

This module provides integration between the artifact management system
and the existing conversation history storage. It enables:

1. Automatic compression of large conversation fields
2. Seamless expansion when loading conversations
3. Backward compatibility with existing conversation files
4. Optimized context generation for LLM calls
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from services.artifact_service import (
    ArtifactStore,
    ArtifactContextManager,
    ArtifactAwareStateManager,
    ArtifactType,
    ArtifactPriority,
    ArtifactReference,
    get_artifact_store,
    get_context_manager,
    get_state_manager
)
from services.context_optimizer import (
    ContextOptimizer,
    OptimizedContext,
    get_context_optimizer
)

logger = logging.getLogger("ConversationArtifactIntegration")

# Directory for conversation history
CONVERSATION_HISTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "conversation_history"
)


class ConversationArtifactManager:
    """
    Manages the integration between conversations and artifacts.
    Provides methods for saving/loading conversations with artifact compression.
    """
    
    # Fields that benefit from artifact storage
    ARTIFACT_CANDIDATE_FIELDS = [
        "canvas_content",       # HTML/Markdown - often very large
        "completed_tasks",      # Can accumulate many results
        "task_agent_pairs",     # Contains full agent definitions
        "task_plan",            # Execution plans with details
        "messages",             # Conversation history
    ]
    
    # Size thresholds for artifact creation (in characters)
    THRESHOLDS = {
        "canvas_content": 500,      # Canvas is usually large
        "completed_tasks": 2000,    # Multiple task results
        "task_agent_pairs": 3000,   # Agent definitions are verbose
        "task_plan": 2000,          # Plans can be detailed
        "messages": 5000,           # Long conversations
    }
    
    def __init__(self):
        self.store = get_artifact_store()
        self.context_manager = get_context_manager()
        self.state_manager = get_state_manager()
        self.optimizer = get_context_optimizer()
    
    def save_conversation_with_artifacts(
        self,
        conversation_data: Dict[str, Any],
        thread_id: str,
        compress_large_fields: bool = True
    ) -> Dict[str, Any]:
        """
        Save a conversation, optionally compressing large fields to artifacts.
        
        Args:
            conversation_data: The full conversation data
            thread_id: The conversation thread ID
            compress_large_fields: Whether to compress large fields
        
        Returns:
            The saved conversation data (with artifact references if compressed)
        """
        if not compress_large_fields:
            # Save as-is
            return self._save_raw_conversation(conversation_data, thread_id)
        
        # Create a copy to avoid modifying original
        data = dict(conversation_data)
        artifact_refs = {}
        
        # Check each candidate field for compression
        for field in self.ARTIFACT_CANDIDATE_FIELDS:
            if field not in data or data[field] is None:
                continue
            
            value = data[field]
            threshold = self.THRESHOLDS.get(field, 2000)
            
            # Check if field should be compressed
            if self._should_compress(value, threshold):
                # Store as artifact
                artifact_type = self._get_artifact_type(field)
                ref = self._store_field_as_artifact(
                    content=value,
                    field_name=field,
                    thread_id=thread_id,
                    artifact_type=artifact_type
                )
                
                if ref:
                    artifact_refs[field] = {
                        "artifact_id": ref.id,
                        "type": ref.type.value,
                        "summary": ref.summary,
                        "size_bytes": ref.size_bytes
                    }
                    # Replace with summary in saved data
                    data[field] = f"[ARTIFACT:{ref.id}] {ref.summary}"
                    logger.info(f"Compressed {field} to artifact {ref.id} ({ref.size_bytes} bytes)")
        
        # Add artifact references to data
        if artifact_refs:
            data["_artifact_refs"] = artifact_refs
        
        # Save the conversation
        return self._save_raw_conversation(data, thread_id)
    
    def load_conversation_with_artifacts(
        self,
        thread_id: str,
        expand_artifacts: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load a conversation, optionally expanding artifact references.
        
        Args:
            thread_id: The conversation thread ID
            expand_artifacts: Whether to expand artifact references
        
        Returns:
            The conversation data (with artifacts expanded if requested)
        """
        data = self._load_raw_conversation(thread_id)
        if data is None:
            return None
        
        if not expand_artifacts:
            return data
        
        # Check for artifact references
        artifact_refs = data.pop("_artifact_refs", {})
        if not artifact_refs:
            return data
        
        # Expand each artifact reference
        for field, ref_info in artifact_refs.items():
            artifact_id = ref_info.get("artifact_id")
            if not artifact_id:
                continue
            
            artifact = self.store.retrieve(artifact_id)
            if artifact:
                data[field] = artifact.content
                logger.info(f"Expanded artifact {artifact_id} for field {field}")
            else:
                logger.warning(f"Failed to expand artifact {artifact_id} for field {field}")
                # Keep the reference string as fallback
        
        return data
    
    def get_optimized_context_for_llm(
        self,
        conversation_data: Dict[str, Any],
        thread_id: str,
        max_tokens: int = 8000,
        focus_fields: List[str] = None
    ) -> OptimizedContext:
        """
        Get optimized context for LLM consumption.
        
        Args:
            conversation_data: The conversation data
            thread_id: The conversation thread ID
            max_tokens: Maximum tokens for context
            focus_fields: Fields to prioritize
        
        Returns:
            OptimizedContext with compressed content
        """
        return self.optimizer.optimize_state_for_context(
            state=conversation_data,
            thread_id=thread_id,
            max_tokens=max_tokens,
            focus_fields=focus_fields or []
        )
    
    def get_lightweight_conversation_summary(
        self,
        thread_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a lightweight summary of a conversation for listing/preview.
        
        Returns minimal data without loading full content.
        """
        data = self._load_raw_conversation(thread_id)
        if data is None:
            return None
        
        # Extract summary fields only
        summary = {
            "thread_id": data.get("thread_id", thread_id),
            "status": data.get("status", "unknown"),
            "timestamp": data.get("timestamp"),
            "message_count": len(data.get("messages", [])),
            "has_canvas": data.get("has_canvas", False),
            "has_artifacts": "_artifact_refs" in data,
        }
        
        # Get original prompt preview
        if prompt := data.get("metadata", {}).get("original_prompt"):
            summary["prompt_preview"] = prompt[:100] + ("..." if len(prompt) > 100 else "")
        
        # Get final response preview
        if response := data.get("final_response"):
            summary["response_preview"] = response[:100] + ("..." if len(response) > 100 else "")
        
        return summary
    
    def cleanup_thread_artifacts(self, thread_id: str) -> int:
        """
        Clean up all artifacts associated with a thread.
        
        Returns the number of artifacts deleted.
        """
        artifacts = self.store.list_by_thread(thread_id)
        deleted = 0
        
        for meta in artifacts:
            if self.store.delete(meta.id):
                deleted += 1
        
        logger.info(f"Cleaned up {deleted} artifacts for thread {thread_id}")
        return deleted
    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    def _should_compress(self, value: Any, threshold: int) -> bool:
        """Check if a value should be compressed to an artifact"""
        if isinstance(value, str):
            return len(value) > threshold
        elif isinstance(value, (list, dict)):
            json_str = json.dumps(value, default=str)
            return len(json_str) > threshold
        return False
    
    def _get_artifact_type(self, field_name: str) -> ArtifactType:
        """Map field names to artifact types"""
        mapping = {
            "canvas_content": ArtifactType.CANVAS,
            "completed_tasks": ArtifactType.RESULT,
            "task_agent_pairs": ArtifactType.DATA,
            "task_plan": ArtifactType.PLAN,
            "messages": ArtifactType.CONVERSATION,
        }
        return mapping.get(field_name, ArtifactType.DATA)
    
    def _store_field_as_artifact(
        self,
        content: Any,
        field_name: str,
        thread_id: str,
        artifact_type: ArtifactType
    ) -> Optional[ArtifactReference]:
        """Store a field as an artifact and return reference"""
        try:
            ref = self.context_manager.compress_for_context(
                content=content,
                name=f"{thread_id}_{field_name}",
                thread_id=thread_id,
                artifact_type=artifact_type,
                force_artifact=True
            )
            
            if isinstance(ref, ArtifactReference):
                return ref
            return None
        except Exception as e:
            logger.error(f"Failed to store artifact for {field_name}: {e}")
            return None
    
    def _save_raw_conversation(
        self,
        data: Dict[str, Any],
        thread_id: str
    ) -> Dict[str, Any]:
        """Save conversation data to file"""
        os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
        file_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved conversation to {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to save conversation {thread_id}: {e}")
            raise
    
    def _load_raw_conversation(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation data from file"""
        file_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
        
        if not os.path.exists(file_path):
            logger.warning(f"Conversation file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversation {thread_id}: {e}")
            return None


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_manager: Optional[ConversationArtifactManager] = None


def get_conversation_artifact_manager() -> ConversationArtifactManager:
    """Get the global conversation artifact manager instance"""
    global _manager
    if _manager is None:
        _manager = ConversationArtifactManager()
    return _manager


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def save_conversation(
    conversation_data: Dict[str, Any],
    thread_id: str,
    compress: bool = True
) -> Dict[str, Any]:
    """Save a conversation with optional artifact compression"""
    return get_conversation_artifact_manager().save_conversation_with_artifacts(
        conversation_data=conversation_data,
        thread_id=thread_id,
        compress_large_fields=compress
    )


def load_conversation(
    thread_id: str,
    expand: bool = True
) -> Optional[Dict[str, Any]]:
    """Load a conversation with optional artifact expansion"""
    return get_conversation_artifact_manager().load_conversation_with_artifacts(
        thread_id=thread_id,
        expand_artifacts=expand
    )


def get_optimized_llm_context(
    conversation_data: Dict[str, Any],
    thread_id: str,
    max_tokens: int = 8000
) -> OptimizedContext:
    """Get optimized context for LLM"""
    return get_conversation_artifact_manager().get_optimized_context_for_llm(
        conversation_data=conversation_data,
        thread_id=thread_id,
        max_tokens=max_tokens
    )
