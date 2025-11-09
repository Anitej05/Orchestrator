#!/usr/bin/env python3
"""
Cleanup corrupted file with encoding issues.
This script handles the one remaining corrupted file.
"""

import os
import sys
import shutil
from datetime import datetime

CORRUPTED_FILE = "conversation_history/9d59a584-8fee-4166-bf4b-523f2065d9ed.json"
ARCHIVE_DIR = "conversation_history/_archive/corrupted"

def cleanup_corrupted_file():
    """Handle the one corrupted file with encoding issues."""
    
    print(f"\n{'='*70}")
    print(f"CLEANUP: Corrupted File Handler")
    print(f"{'='*70}")
    print(f"\nFile: {CORRUPTED_FILE}")
    print(f"Issue: UTF-8 encoding error (binary data in conversation)")
    
    if not os.path.exists(CORRUPTED_FILE):
        print(f"✓ File already removed or archived")
        return
    
    # Get file info
    file_size = os.path.getsize(CORRUPTED_FILE) / 1024
    print(f"Size: {file_size:.1f} KB")
    
    print(f"\n⚠️  This file is corrupted (cannot be parsed).")
    print(f"Options:")
    print(f"  1. Archive it (recommended) - Keep in _archive/ for manual inspection")
    print(f"  2. Delete it - Remove file completely")
    print(f"  3. Leave it - Do nothing (not recommended)")
    
    response = input(f"\nAction (archive/delete/leave): ").strip().lower()
    
    if response == "archive":
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        archive_path = os.path.join(ARCHIVE_DIR, "9d59a584-8fee-4166-bf4b-523f2065d9ed.json")
        shutil.copy2(CORRUPTED_FILE, archive_path)
        os.remove(CORRUPTED_FILE)
        print(f"✓ Archived to: {archive_path}")
        print(f"✓ Original file removed")
        
    elif response == "delete":
        os.remove(CORRUPTED_FILE)
        print(f"✓ Deleted file: {CORRUPTED_FILE}")
        
    elif response == "leave":
        print(f"⚠️  File left as-is. This will continue to show in audits.")
    else:
        print(f"✗ Invalid response")

if __name__ == "__main__":
    cleanup_corrupted_file()
