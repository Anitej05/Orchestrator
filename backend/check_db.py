"""Quick database check script"""
from database import SessionLocal
from models import UserThread

db = SessionLocal()
try:
    total_count = db.query(UserThread).count()
    print(f"Total UserThread records: {total_count}")
    print("\nLast 10 threads (most recent first):")
    print("-" * 80)
    
    threads = db.query(UserThread).order_by(UserThread.updated_at.desc()).limit(10).all()
    for t in threads:
        thread_id_short = t.thread_id[:30] + "..." if len(t.thread_id) > 30 else t.thread_id
        title_short = (t.title[:40] + "...") if t.title and len(t.title) > 40 else (t.title or "No title")
        print(f"{thread_id_short:<35} | {title_short:<45} | {t.updated_at}")
finally:
    db.close()
