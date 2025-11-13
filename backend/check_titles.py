"""
Quick check to see what titles are in the database
"""
from database import SessionLocal
from models import UserThread

db = SessionLocal()

try:
    # Get first 10 threads
    threads = db.query(UserThread).limit(10).all()
    
    print(f"\nChecking {len(threads)} threads:\n")
    for thread in threads:
        print(f"Thread: {thread.thread_id[:20]}...")
        print(f"  Title: '{thread.title}'")
        print(f"  User: {thread.user_id}")
        print(f"  Created: {thread.created_at}")
        print()
        
finally:
    db.close()
