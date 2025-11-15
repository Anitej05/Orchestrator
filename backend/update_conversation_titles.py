"""
Update existing conversations with AI-generated titles
Run this to fix conversations that have None or empty titles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database import SessionLocal
from models import UserThread
from orchestrator.graph import generate_conversation_title
import json

CONVERSATION_HISTORY_DIR = "conversation_history"

def update_titles():
    db = SessionLocal()
    try:
        # Get all threads with None or empty titles
        threads = db.query(UserThread).filter(
            (UserThread.title == None) | 
            (UserThread.title == '') | 
            (UserThread.title == 'None')
        ).all()
        
        print(f"Found {len(threads)} conversations without titles")
        
        updated = 0
        for thread in threads:
            try:
                # Load conversation history
                history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread.thread_id}.json")
                
                if not os.path.exists(history_path):
                    print(f"  ⚠️  No history file for {thread.thread_id}, skipping")
                    continue
                
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # Get original prompt and messages
                state = history.get('state', {})
                original_prompt = state.get('original_prompt', '')
                messages = state.get('messages', [])
                
                if not original_prompt and not messages:
                    print(f"  ⚠️  No content for {thread.thread_id}, setting as 'Untitled'")
                    thread.title = "Untitled Conversation"
                    updated += 1
                    continue
                
                # Generate title
                print(f"  🔄 Generating title for {thread.thread_id}...")
                title = generate_conversation_title(original_prompt, messages)
                
                if not title:
                    # Fallback to prompt
                    if original_prompt:
                        title = original_prompt[:50] + "..." if len(original_prompt) > 50 else original_prompt
                    else:
                        title = "Untitled Conversation"
                
                thread.title = title
                print(f"  ✅ Set title: '{title}'")
                updated += 1
                
            except Exception as e:
                print(f"  ❌ Error updating {thread.thread_id}: {e}")
                continue
        
        # Commit all changes
        db.commit()
        print(f"\n✅ Updated {updated} conversation titles")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🔧 Updating conversation titles...\n")
    update_titles()
