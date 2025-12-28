"""Script to read log file and filter out base64 content"""
import re
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

log_file = "logs/vnr_syllabus_test_20251226_073655.log"

# Patterns to filter out base64 content
base64_pattern = re.compile(r'[A-Za-z0-9+/]{100,}[=]{0,2}')  # Long base64 strings

with open(log_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        # Skip lines with very long sequences (likely base64)
        if len(line) > 500:
            # Check if it's mostly base64
            if base64_pattern.search(line):
                print(f"[LINE {line_num}: <base64 image data skipped - {len(line)} chars>]")
                continue
        
        # Skip lines that contain obvious base64 markers
        if 'data:image' in line or 'base64,' in line:
            print(f"[LINE {line_num}: <base64 image data skipped>]")
            continue
            
        print(line.rstrip())
