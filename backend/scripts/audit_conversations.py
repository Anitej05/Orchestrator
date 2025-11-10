#!/usr/bin/env python3
"""
TIER 2 FIX: Audit script to find orphaned and corrupted conversations.

This script identifies consistency issues between the file system and database:
- Orphaned files (exist in file system, not in database)
- Orphaned records (exist in database, file doesn't exist)
- Owner mismatches (DB owner != file owner)
- Corrupted JSON files

Usage:
    python backend/scripts/audit_conversations.py

Output files:
    - audit_orphaned_files.txt: Files without DB records
    - audit_orphaned_records.txt: DB records without files
    - audit_owner_mismatches.txt: Owner inconsistencies
    - audit_corrupted_files.txt: Files with JSON errors
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path so we can import from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import SessionLocal
from models import UserThread

CONVERSATION_DIR = "conversation_history"
OUTPUT_DIR = "audit_results"

def ensure_output_dir():
    """Create output directory for audit results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_file_threads():
    """Get all thread IDs from file system."""
    threads = set()
    if not os.path.exists(CONVERSATION_DIR):
        print(f"⚠️ Conversation directory not found: {CONVERSATION_DIR}")
        return threads
    
    for filename in os.listdir(CONVERSATION_DIR):
        if filename.endswith(".json"):
            thread_id = filename.replace(".json", "")
            threads.add(thread_id)
    
    return threads

def get_all_db_threads():
    """Get all thread IDs from database."""
    session = SessionLocal()
    try:
        records = session.query(UserThread).all()
        threads = {r.thread_id: r.user_id for r in records}
        return threads
    finally:
        session.close()

def audit_orphaned_files(file_threads, db_threads):
    """Find files without corresponding DB records."""
    orphaned = file_threads - set(db_threads.keys())
    
    print(f"\n{'='*70}")
    print(f"ORPHANED FILES (exist in file system, not in database)")
    print(f"{'='*70}")
    
    if orphaned:
        print(f"⚠️  Found {len(orphaned)} orphaned files:")
        with open(os.path.join(OUTPUT_DIR, "audit_orphaned_files.txt"), "w") as f:
            for thread_id in sorted(orphaned):
                file_path = os.path.join(CONVERSATION_DIR, f"{thread_id}.json")
                try:
                    size_kb = os.path.getsize(file_path) / 1024
                    with open(file_path) as fh:
                        conversation = json.load(fh)
                    owner = conversation.get("owner", "unknown")
                    print(f"  - {thread_id} ({size_kb:.1f}KB, owner={owner})")
                    f.write(f"{thread_id}|{owner}|{size_kb:.1f}\n")
                except Exception as e:
                    print(f"  - {thread_id} (ERROR: {e})")
                    f.write(f"{thread_id}|ERROR|{e}\n")
    else:
        print("✓ No orphaned files found!")
    
    return orphaned

def audit_orphaned_records(file_threads, db_threads):
    """Find DB records without corresponding files."""
    orphaned_ids = set(db_threads.keys()) - file_threads
    
    print(f"\n{'='*70}")
    print(f"ORPHANED DATABASE RECORDS (exist in database, file not found)")
    print(f"{'='*70}")
    
    if orphaned_ids:
        print(f"⚠️  Found {len(orphaned_ids)} orphaned records:")
        with open(os.path.join(OUTPUT_DIR, "audit_orphaned_records.txt"), "w") as f:
            for thread_id in sorted(orphaned_ids):
                owner_id = db_threads[thread_id]
                print(f"  - {thread_id} (owner: {owner_id})")
                f.write(f"{thread_id}|{owner_id}\n")
    else:
        print("✓ No orphaned records found!")
    
    return orphaned_ids

def audit_owner_mismatches(file_threads, db_threads):
    """Find owner mismatches between file and database."""
    mismatches = []
    
    print(f"\n{'='*70}")
    print(f"OWNER MISMATCHES (DB owner != file owner)")
    print(f"{'='*70}")
    
    with open(os.path.join(OUTPUT_DIR, "audit_owner_mismatches.txt"), "w") as f:
        for thread_id in file_threads:
            if thread_id not in db_threads:
                continue  # Skip orphaned files
            
            db_owner = db_threads[thread_id]
            file_path = os.path.join(CONVERSATION_DIR, f"{thread_id}.json")
            
            try:
                with open(file_path) as fh:
                    conversation = json.load(fh)
                file_owner = conversation.get("owner")
                
                if file_owner and file_owner != db_owner:
                    mismatches.append({
                        "thread_id": thread_id,
                        "db_owner": db_owner,
                        "file_owner": file_owner
                    })
                    print(f"  ✗ {thread_id}: DB={db_owner}, File={file_owner}")
                    f.write(f"{thread_id}|{db_owner}|{file_owner}\n")
                elif not file_owner:
                    print(f"  ⚠️ {thread_id}: File missing owner field (DB has {db_owner})")
                    f.write(f"{thread_id}|{db_owner}|MISSING\n")
                    
            except Exception as e:
                print(f"  ✗ {thread_id}: Error reading file: {e}")
                f.write(f"{thread_id}|{db_owner}|ERROR: {e}\n")
    
    if not mismatches:
        print("✓ No owner mismatches found!")
    else:
        print(f"\n⚠️  Found {len(mismatches)} owner mismatches")
    
    return mismatches

def audit_corrupted_files(file_threads):
    """Find files with JSON corruption or other errors."""
    corrupted = []
    
    print(f"\n{'='*70}")
    print(f"CORRUPTED FILES (JSON parse errors)")
    print(f"{'='*70}")
    
    with open(os.path.join(OUTPUT_DIR, "audit_corrupted_files.txt"), "w") as f:
        for thread_id in file_threads:
            file_path = os.path.join(CONVERSATION_DIR, f"{thread_id}.json")
            
            try:
                with open(file_path) as fh:
                    json.load(fh)
            except json.JSONDecodeError as e:
                corrupted.append(thread_id)
                print(f"  ✗ {thread_id}: {e}")
                f.write(f"{thread_id}|{e}\n")
            except Exception as e:
                corrupted.append(thread_id)
                print(f"  ✗ {thread_id}: {e}")
                f.write(f"{thread_id}|{e}\n")
    
    if not corrupted:
        print("✓ No corrupted files found!")
    else:
        print(f"\n⚠️  Found {len(corrupted)} corrupted files")
    
    return corrupted

def generate_summary(file_threads, db_threads, orphaned_files, orphaned_records, mismatches, corrupted):
    """Generate summary report."""
    total_issues = len(orphaned_files) + len(orphaned_records) + len(mismatches) + len(corrupted)
    
    print(f"\n{'='*70}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"Total files: {len(file_threads)}")
    print(f"Total DB records: {len(db_threads)}")
    print(f"\nIssues found:")
    print(f"  - Orphaned files: {len(orphaned_files)}")
    print(f"  - Orphaned records: {len(orphaned_records)}")
    print(f"  - Owner mismatches: {len(mismatches)}")
    print(f"  - Corrupted files: {len(corrupted)}")
    print(f"\nTotal issues: {total_issues}")
    
    if total_issues == 0:
        print("\n✓ All conversations are consistent! No fixes needed.")
    else:
        print(f"\n⚠️  Found {total_issues} issues that need fixing.")
        print(f"Run 'python backend/scripts/fix_conversations.py' to fix them.")
    
    # Save summary
    with open(os.path.join(OUTPUT_DIR, "audit_summary.txt"), "w") as f:
        f.write(f"Audit Summary - {datetime.now().isoformat()}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total files: {len(file_threads)}\n")
        f.write(f"Total DB records: {len(db_threads)}\n")
        f.write(f"\nOrphaned files: {len(orphaned_files)}\n")
        f.write(f"Orphaned records: {len(orphaned_records)}\n")
        f.write(f"Owner mismatches: {len(mismatches)}\n")
        f.write(f"Corrupted files: {len(corrupted)}\n")
        f.write(f"\nTotal issues: {total_issues}\n")

def main():
    """Run complete audit."""
    print(f"\n{'='*70}")
    print(f"CONVERSATION AUDIT - TIER 2 FIX")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Location: {os.path.abspath(CONVERSATION_DIR)}")
    
    # Create output directory
    ensure_output_dir()
    
    # Get all threads
    file_threads = get_all_file_threads()
    db_threads = get_all_db_threads()
    
    print(f"\nScanning...")
    print(f"  Files: {len(file_threads)}")
    print(f"  Database records: {len(db_threads)}")
    
    # Run audits
    orphaned_files = audit_orphaned_files(file_threads, db_threads)
    orphaned_records = audit_orphaned_records(file_threads, db_threads)
    mismatches = audit_owner_mismatches(file_threads, db_threads)
    corrupted = audit_corrupted_files(file_threads)
    
    # Generate summary
    generate_summary(file_threads, db_threads, orphaned_files, orphaned_records, mismatches, corrupted)
    
    print(f"\n✓ Audit complete. Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
