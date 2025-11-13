"""
One-time migration script to populate missing titles in user_threads table
by reading from conversation history files.
"""
import os
import json
from sqlalchemy.orm import Session
from database import SessionLocal
from models import UserThread

CONVERSATION_HISTORY_DIR = "conversation_history"

def migrate_titles():
    """Update all user_threads with missing titles using conversation history"""
    db: Session = SessionLocal()
    
    try:
        # Get all threads with missing titles (including the string "None")
        threads_without_titles = db.query(UserThread).filter(
            (UserThread.title == None) | 
            (UserThread.title == "") | 
            (UserThread.title == "Untitled Conversation") |
            (UserThread.title == "None")
        ).all()
        
        print(f"Found {len(threads_without_titles)} threads without proper titles")
        
        updated_count = 0
        for thread in threads_without_titles:
            # Try to read conversation history file
            history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread.thread_id}.json")
            
            if os.path.exists(history_path):
                try:
                    with open(history_path, "r", encoding="utf-8") as f:
                        history_data = json.load(f)
                    
                    # Get original prompt or first user message
                    original_prompt = history_data.get("original_prompt", "")
                    
                    # If no original_prompt, try to get from messages
                    if not original_prompt:
                        messages = history_data.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("type") == "human":
                                original_prompt = msg.get("content", "")
                                break
                    
                    if original_prompt:
                        # Generate title (first 50 chars)
                        title = original_prompt[:50] + "..." if len(original_prompt) > 50 else original_prompt
                        thread.title = title
                        updated_count += 1
                        print(f"✓ Updated {thread.thread_id}: {title}")
                    else:
                        print(f"✗ No message found for {thread.thread_id}")
                        
                except Exception as e:
                    print(f"✗ Error reading history for {thread.thread_id}: {e}")
            else:
                print(f"✗ History file not found for {thread.thread_id}")
        
        # Commit all changes
        db.commit()
        print(f"\n✅ Successfully updated {updated_count} conversation titles")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Starting title migration...")
    migrate_titles()
    print("Migration complete!")
