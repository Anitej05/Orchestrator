"""
Quick test of LLM-based vision analysis
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.browser_automation_agent import analyze_task_for_vision

# Test a few key cases
test_cases = [
    ("Take a screenshot of google.com", False),
    ("Check if the login button is visible", True),
    ("Navigate to github.com", False),
    ("Verify the layout looks correct", True),
    ("Find the red button on the page", True),
]

print("Testing LLM-based vision analysis...\n")

for task, expected in test_cases:
    print(f"Task: '{task}'")
    result = analyze_task_for_vision(task)
    status = "✓" if result == expected else "✗"
    print(f"{status} Result: {'Vision' if result else 'Text-only'} (Expected: {'Vision' if expected else 'Text-only'})")
    print()

print("Test complete!")
