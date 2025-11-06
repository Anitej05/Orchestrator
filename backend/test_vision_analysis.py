"""
Test the vision analysis function to verify it correctly identifies
when vision is needed vs when text-only is sufficient
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.browser_automation_agent import analyze_task_for_vision

def test_vision_analysis():
    """Test various task types to verify correct vision/text-only classification"""
    
    print("=" * 70)
    print("Vision Analysis Test")
    print("=" * 70)
    
    # Test cases: (task, expected_needs_vision, reason)
    test_cases = [
        # Simple tasks - should use text-only
        ("Take a screenshot of google.com", False, "Simple screenshot capture"),
        ("Navigate to github.com", False, "Simple navigation"),
        ("Go to weather.com", False, "Simple URL visit"),
        ("Open https://example.com", False, "Direct URL open"),
        ("Screenshot of reddit.com", False, "Basic screenshot"),
        
        # Text extraction - should use text-only
        ("Extract text from the page", False, "Text extraction"),
        ("Get the title of the page", False, "Simple text retrieval"),
        ("Read the content from the article", False, "Text reading"),
        ("Search for python on github", False, "Search operation"),
        
        # Visual verification - should use vision
        ("Check if the button is visible", True, "Visual verification"),
        ("Verify the layout looks correct", True, "Layout verification"),
        ("See if the image loaded properly", True, "Image verification"),
        ("Look at the design and tell me if it matches", True, "Design comparison"),
        ("Check the appearance of the homepage", True, "Appearance check"),
        
        # Complex visual tasks - should use vision
        ("Find the red button on the page", True, "Color-based identification"),
        ("Identify the login button", True, "Visual identification"),
        ("Compare the two screenshots", True, "Visual comparison"),
        ("Tell me what the page looks like", True, "Visual description"),
        ("Verify the CAPTCHA is displayed", True, "CAPTCHA detection"),
        
        # Ambiguous cases - defaults to text-only for efficiency
        ("Check the website", False, "Ambiguous - defaults to text-only"),
        ("Visit the page and tell me about it", False, "General task - text-only"),
    ]
    
    passed = 0
    failed = 0
    
    print("\nRunning test cases...\n")
    
    for task, expected_vision, reason in test_cases:
        result = analyze_task_for_vision(task)
        status = "✓ PASS" if result == expected_vision else "✗ FAIL"
        
        if result == expected_vision:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | Task: '{task}'")
        print(f"       | Expected: {'Vision' if expected_vision else 'Text-only'}, "
              f"Got: {'Vision' if result else 'Text-only'}")
        print(f"       | Reason: {reason}")
        print()
    
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All tests passed! Vision analysis is working correctly.")
    else:
        print(f"\n✗ {failed} test(s) failed. Review the logic in analyze_task_for_vision()")
    
    print("\n" + "=" * 70)
    print("Vision Analysis Strategy:")
    print("=" * 70)
    print("✓ Text-only for: Simple navigation, screenshots, text extraction")
    print("✓ Vision for: Visual verification, layout checks, image recognition")
    print("✓ Default: Text-only (more cost-efficient, can fallback to vision)")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = test_vision_analysis()
    sys.exit(0 if success else 1)
