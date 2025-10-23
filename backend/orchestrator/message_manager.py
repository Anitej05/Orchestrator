"""
Message Manager - Single source of truth for conversation messages
Handles all message operations to prevent duplicates and ensure consistency
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

logger = logging.getLogger("MessageManager")

class MessageManager:
    """Manages conversation messages with deduplication and consistency checks"""
    
    @staticmethod
    def create_message_id(content: str, msg_type: str, timestamp: float) -> str:
        """Create a unique, deterministic message ID"""
        import hashlib
        # Create ID based on content + type + timestamp to ensure uniqueness
        unique_string = f"{msg_type}:{content}:{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    @staticmethod
    def deduplicate_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Remove duplicate messages based on content and type"""
        seen = set()
        unique_messages = []
        
        for msg in messages:
            # Create a signature for the message
            msg_type = msg.__class__.__name__
            content = msg.content if hasattr(msg, 'content') else str(msg)
            signature = f"{msg_type}:{content}"
            
            if signature not in seen:
                seen.add(signature)
                unique_messages.append(msg)
            else:
                logger.debug(f"Skipping duplicate message: {signature[:50]}...")
        
        return unique_messages
    
    @staticmethod
    def add_message(messages: List[BaseMessage], new_message: BaseMessage) -> List[BaseMessage]:
        """Add a new message to the list, ensuring no duplicates"""
        # Check if this exact message already exists
        new_content = new_message.content if hasattr(new_message, 'content') else str(new_message)
        new_type = new_message.__class__.__name__
        
        for existing_msg in messages:
            existing_content = existing_msg.content if hasattr(existing_msg, 'content') else str(existing_msg)
            existing_type = existing_msg.__class__.__name__
            
            if existing_content == new_content and existing_type == new_type:
                logger.debug(f"Message already exists, not adding duplicate: {new_content[:50]}...")
                return messages
        
        # Message is unique, add it
        return messages + [new_message]
    
    @staticmethod
    def merge_messages(existing: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
        """Merge two message lists, removing duplicates"""
        # Start with existing messages
        merged = list(existing)
        
        # Add new messages one by one, checking for duplicates
        for new_msg in new:
            merged = MessageManager.add_message(merged, new_msg)
        
        return merged
    
    @staticmethod
    def validate_message_order(messages: List[BaseMessage]) -> bool:
        """Validate that messages alternate between human and AI (with some flexibility)"""
        if not messages:
            return True
        
        # Just check that we don't have too many consecutive messages of the same type
        consecutive_count = 1
        prev_type = messages[0].__class__.__name__
        
        for msg in messages[1:]:
            curr_type = msg.__class__.__name__
            if curr_type == prev_type:
                consecutive_count += 1
                if consecutive_count > 2:  # Allow up to 2 consecutive
                    logger.warning(f"Found {consecutive_count} consecutive {curr_type} messages")
                    return False
            else:
                consecutive_count = 1
            prev_type = curr_type
        
        return True
