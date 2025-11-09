#!/usr/bin/env python3
"""
TIER 3 FIX: Fix script to recover and repair corrupted conversations.

This script repairs issues found by audit_conversations.py:
- Recovers orphaned files by creating DB records if owner can be extracted
- Archives orphaned files without owners
- Removes orphaned DB records
- Fixes owner mismatches (prefers DB as source of truth)

Usage:
    python backend/scripts/fix_conversations.py

WARNING: This script modifies files and database. Make backups first!
"""

import os
import sys
import json
import shutil
from datetime import datetime

# Add parent directory to path so we can import from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import SessionLocal
from models import UserThread

CONVERSATION_DIR = "conversation_history"
ARCHIVE_DIR = "conversation_history/_archive"
AUDIT_DIR = "audit_results"

def load_audit_results(filename):
    """Load results from audit output files."""
    filepath = os.path.join(AUDIT_DIR, filename)
    results = []
    
    if not os.path.exists(filepath):
        return results
    
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(line.split("|"))
    
    return results

def fix_orphaned_files():
    """Recover orphaned files by creating DB records or archiving."""
    print(f"\n{'='*70}")
    print(f"FIXING ORPHANED FILES")
    print(f"{'='*70}")
    
    orphaned = load_audit_results("audit_orphaned_files.txt")
    session = SessionLocal()
    recovered = 0
    archived = 0
    errors = 0
    
    try:
        for row in orphaned:
            if len(row) < 3:
                continue
            
            thread_id = row[0]
            owner = row[1] if row[1] != "unknown" else None
            
            file_path = os.path.join(CONVERSATION_DIR, f"{thread_id}.json")
            
            if not os.path.exists(file_path):
                print(f"  ⚠️ File missing: {thread_id}")
                continue
            
            try:
                # Try to extract owner from file if not already known
                if not owner:
                    with open(file_path) as f:
                        conversation = json.load(f)
                    owner = conversation.get("owner")
                
                if owner and owner != "unknown":
                    # Create DB record for this orphaned file
                    existing = session.query(UserThread).filter_by(
                        user_id=owner,
                        thread_id=thread_id
                    ).first()
                    
                    if not existing:
                        user_thread = UserThread(user_id=owner, thread_id=thread_id)
                        session.add(user_thread)
                        session.commit()
                        print(f"  ✓ Recovered: {thread_id} → owner={owner}")
                        recovered += 1
                    else:
                        print(f"  ℹ  Already exists: {thread_id}")
                else:
                    # Can't identify owner - archive the file
                    os.makedirs(os.path.join(ARCHIVE_DIR, "orphaned"), exist_ok=True)
                    archive_path = os.path.join(ARCHIVE_DIR, "orphaned", f"{thread_id}.json")
                    shutil.copy2(file_path, archive_path)
                    os.remove(file_path)
                    print(f"  📦 Archived: {thread_id}")
                    archived += 1
                    
            except Exception as e:
                print(f"  ✗ Error processing {thread_id}: {e}")
                errors += 1
    
    finally:
        session.close()
    
    print(f"\nResults: {recovered} recovered, {archived} archived, {errors} errors")
    return recovered, archived, errors

def fix_orphaned_records():
    """Remove orphaned DB records."""
    print(f"\n{'='*70}")
    print(f"FIXING ORPHANED DATABASE RECORDS")
    print(f"{'='*70}")
    
    orphaned = load_audit_results("audit_orphaned_records.txt")
    session = SessionLocal()
    deleted = 0
    errors = 0
    
    try:
        for row in orphaned:
            if len(row) < 2:
                continue
            
            thread_id = row[0]
            
            try:
                record = session.query(UserThread).filter_by(thread_id=thread_id).first()
                if record:
                    session.delete(record)
                    session.commit()
                    print(f"  ✓ Deleted orphaned record: {thread_id}")
                    deleted += 1
            except Exception as e:
                print(f"  ✗ Error deleting {thread_id}: {e}")
                errors += 1
    
    finally:
        session.close()
    
    print(f"\nResults: {deleted} deleted, {errors} errors")
    return deleted, errors

def fix_owner_mismatches():
    """Fix owner mismatches (prefer DB as source of truth)."""
    print(f"\n{'='*70}")
    print(f"FIXING OWNER MISMATCHES")
    print(f"{'='*70}")
    
    mismatches = load_audit_results("audit_owner_mismatches.txt")
    fixed = 0
    errors = 0
    
    for row in mismatches:
        if len(row) < 3:
            continue
        
        thread_id = row[0]
        db_owner = row[1]
        
        file_path = os.path.join(CONVERSATION_DIR, f"{thread_id}.json")
        
        if not os.path.exists(file_path):
            print(f"  ⚠️ File missing: {thread_id}")
            continue
        
        try:
            # Read file
            with open(file_path) as f:
                conversation = json.load(f)
            
            # Update owner to match DB
            if conversation.get("owner") != db_owner:
                conversation["owner"] = db_owner
                conversation["owner_fixed_at"] = datetime.now().isoformat()
                
                # Write back (atomic)
                temp_path = file_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(conversation, f, ensure_ascii=False, indent=2)
                os.replace(temp_path, file_path)
                
                print(f"  ✓ Fixed: {thread_id} → owner={db_owner}")
                fixed += 1
            else:
                print(f"  ℹ  Already correct: {thread_id}")
                
        except Exception as e:
            print(f"  ✗ Error fixing {thread_id}: {e}")
            errors += 1
    
    print(f"\nResults: {fixed} fixed, {errors} errors")
    return fixed, errors

def main():
    """Run complete fix."""
    print(f"\n{'='*70}")
    print(f"CONVERSATION FIX - TIER 3 FIX")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"\n⚠️  WARNING: This script modifies files and database!")
    print(f"Make sure you have backups before continuing.\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    if response != "yes":
        print("Cancelled.")
        return
    
    # Run fixes
    rec, arch, rec_err = fix_orphaned_files()
    del_cnt, del_err = fix_orphaned_records()
    fixed, fix_err = fix_owner_mismatches()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"FIX SUMMARY")
    print(f"{'='*70}")
    print(f"Orphaned files recovered: {rec}")
    print(f"Orphaned files archived: {arch}")
    print(f"Orphaned records deleted: {del_cnt}")
    print(f"Owner mismatches fixed: {fixed}")
    print(f"\nTotal errors: {rec_err + del_err + fix_err}")
    
    if rec_err + del_err + fix_err == 0:
        print(f"\n✓ All fixes applied successfully!")
    else:
        print(f"\n⚠️  Some errors occurred. Review the output above.")
    
    print(f"\n✓ Fixes complete. Audit again to verify: python backend/scripts/audit_conversations.py")

if __name__ == "__main__":
    main()
