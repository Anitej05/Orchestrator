"""
Compare LLM-based analysis vs keyword-based analysis
Shows why LLM is better at understanding context
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.browser_automation_agent import analyze_task_for_vision

# Edge cases where keywords might fail but LLM understands context
test_cases = [
    # Simple cases - both should get right
    ("Take a screenshot of google.com", False, "Simple screenshot"),
    ("Check if button is visible", True, "Visual verification"),
    
    # Context-dependent cases - LLM should understand better
    ("Go to the page and verify everything looks good", True, "Implicit visual check"),
    ("Navigate to the site and make sure it loaded properly", True, "Implicit visual verification"),
    ("Screenshot the page and tell me what you see", True, "Requires analyzing screenshot"),
    ("Just take a quick screenshot", False, "Just capture, no analysis"),
    
    # Ambiguous cases - LLM should use reasoning
    ("Check the website", False, "Ambiguous - could mean many things"),
    ("Look at the homepage", True, "Visual inspection implied"),
    ("Find the submit button", False, "Can use DOM/text"),
    ("Find the button that says Submit", False, "Text-based search"),
    ("Find the blue submit button", True, "Color-based, needs vision"),
]

print("=" * 70)
print("LLM-based Vision Analysis Test")
print("=" * 70)
print("\nTesting edge cases where context matters...\n")

passed = 0
total = len(test_cases)

for task, expected, description in test_cases:
    result = analyze_task_for_vision(task)
    status = "PASS" if result == expected else "FAIL"
    
    if result == expected:
        passed += 1
    
    print(f"[{status}] {description}")
    print(f"  Task: '{task}'")
    print(f"  Expected: {'Vision' if expected else 'Text-only'}, Got: {'Vision' if result else 'Text-only'}")
    print()

print("=" * 70)
print(f"Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
print("=" * 70)
