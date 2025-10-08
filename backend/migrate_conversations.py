import os
import json
import shutil
import time
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversation_migration.log'),
        logging.StreamHandler()
    ]
)

CONVERSATION_HISTORY_DIR = "conversation_history"
BACKUP_DIR = os.path.join(CONVERSATION_HISTORY_DIR, "backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

def create_backup() -> str:
    """Create a backup of all conversation files."""
    if not os.path.exists(CONVERSATION_HISTORY_DIR):
        logging.error(f"Conversation directory {CONVERSATION_HISTORY_DIR} does not exist")
        raise FileNotFoundError(f"Directory {CONVERSATION_HISTORY_DIR} not found")
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    files_backed_up = 0
    
    for filename in os.listdir(CONVERSATION_HISTORY_DIR):
        if filename.endswith('.json'):
            src = os.path.join(CONVERSATION_HISTORY_DIR, filename)
            dst = os.path.join(BACKUP_DIR, filename)
            shutil.copy2(src, dst)
            files_backed_up += 1
    
    logging.info(f"Backed up {files_backed_up} files to {BACKUP_DIR}")
    return BACKUP_DIR

def parse_legacy_python_repr(content: str) -> List[Dict[str, Any]]:
    """Parse legacy Python string representation of messages."""
    messages = []
    
    # Try to find messages list in the string
    messages_match = re.search(r"messages\s*[:=]\s*\[([\s\S]*?)\]\s*,?", content)
    if messages_match:
        inner = messages_match.group(1)
        # Match message objects
        msg_regex = r"(AIMessage|HumanMessage|ChatMessage|SystemMessage)\s*\(([^)]*)\)"
        for match in re.finditer(msg_regex, inner):
            kind = match.group(1)
            body = match.group(2)
            
            # Extract content and other fields
            content_match = re.search(r"content\s*=\s*(?:r?\"([^\"]*)\"|r?\'([^\']*)\')", body)
            id_match = re.search(r"id\s*=\s*(?:r?\"([^\"]*)\"|r?\'([^\']*)\')", body)
            
            if content_match:
                content = content_match.group(1) or content_match.group(2) or ''
                msg_type = 'user' if 'Human' in kind else 'assistant' if 'AI' in kind else 'system'
                msg_id = id_match.group(1) or id_match.group(2) if id_match else str(time.time())
                
                messages.append({
                    'id': msg_id,
                    'type': msg_type,
                    'content': content,
                    'timestamp': time.time(),
                    'metadata': {}
                })
    
    return messages

def convert_to_new_format(content: str, thread_id: str) -> Dict[str, Any]:
    """Convert various formats to the new standardized format."""
    try:
        # First try parsing as JSON
        data = json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try parsing as Python string representation
        try:
            # Check if it's a string representation of a dict
            if content.strip().startswith('"') and content.strip().endswith('"'):
                content = content.strip('"').replace('\\"', '"').replace('\\\\', '\\')
            
            # Try to evaluate as Python literal
            data = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            # If all else fails, try to parse as legacy format
            messages = parse_legacy_python_repr(content)
            return {
                'messages': messages,
                'thread_id': thread_id,
                'timestamp': time.time()
            }
    
    # If we have parsed data, convert it to the new format
    if isinstance(data, list):
        # If it's a list, assume it's a list of messages
        messages = []
        for msg in data:
            if isinstance(msg, dict):
                msg_type = msg.get('type', '')
                if 'human' in msg_type.lower():
                    msg_type = 'user'
                elif 'ai' in msg_type.lower():
                    msg_type = 'assistant'
                else:
                    msg_type = 'system'
                
                messages.append({
                    'id': msg.get('id', str(time.time())),
                    'type': msg_type,
                    'content': msg.get('content', ''),
                    'timestamp': msg.get('timestamp', time.time()),
                    'metadata': msg.get('metadata', {})
                })
    elif isinstance(data, dict):
        # If it's a dict, look for messages key
        if 'messages' in data:
            return {
                'messages': [
                    {
                        'id': msg.get('id', str(time.time())),
                        'type': msg.get('type', 'system'),
                        'content': msg.get('content', ''),
                        'timestamp': msg.get('timestamp', time.time()),
                        'metadata': msg.get('metadata', {})
                    }
                    for msg in data['messages']
                ],
                'thread_id': data.get('thread_id', thread_id),
                'final_response': data.get('final_response'),
                'timestamp': data.get('timestamp', time.time())
            }
        else:
            # Single message in dict form
            messages = [{
                'id': data.get('id', str(time.time())),
                'type': data.get('type', 'system'),
                'content': data.get('content', ''),
                'timestamp': data.get('timestamp', time.time()),
                'metadata': data.get('metadata', {})
            }]
    else:
        messages = []
    
    return {
        'messages': messages,
        'thread_id': thread_id,
        'timestamp': time.time()
    }

def validate_converted_data(data: Dict[str, Any]) -> bool:
    """Validate that converted data meets the new format requirements."""
    try:
        if not isinstance(data, dict):
            return False
        
        required_fields = ['messages', 'thread_id', 'timestamp']
        if not all(field in data for field in required_fields):
            return False
        
        if not isinstance(data['messages'], list):
            return False
        
        for msg in data['messages']:
            if not isinstance(msg, dict):
                return False
            
            required_msg_fields = ['id', 'type', 'content', 'timestamp', 'metadata']
            if not all(field in msg for field in required_msg_fields):
                return False
            
            if msg['type'] not in ['user', 'assistant', 'system']:
                return False
            
            if not isinstance(msg['metadata'], dict):
                return False
        
        return True
    except Exception as e:
        logging.error(f"Validation error: {e}")
        return False

def migrate_conversations():
    """Migrate all conversation files to the new format."""
    try:
        # Create backup first
        backup_dir = create_backup()
        logging.info(f"Created backup in {backup_dir}")
        
        # Process each conversation file
        success_count = 0
        error_count = 0
        
        for filename in os.listdir(CONVERSATION_HISTORY_DIR):
            if not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(CONVERSATION_HISTORY_DIR, filename)
            thread_id = filename[:-5]  # Remove .json extension
            
            try:
                # Read the original file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert to new format
                converted_data = convert_to_new_format(content, thread_id)
                
                # Validate the converted data
                if not validate_converted_data(converted_data):
                    raise ValueError(f"Converted data for {filename} failed validation")
                
                # Write the converted data back
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                logging.info(f"Successfully migrated {filename}")
                
            except Exception as e:
                error_count += 1
                logging.error(f"Failed to migrate {filename}: {e}")
        
        logging.info(f"Migration complete. Successes: {success_count}, Failures: {error_count}")
        logging.info(f"Backup of original files is available in {backup_dir}")
        
    except Exception as e:
        logging.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    try:
        migrate_conversations()
    except Exception as e:
        logging.error(f"Migration script failed: {e}")
        raise