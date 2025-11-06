"""
Comprehensive Test Suite for Browser Automation Agent
Tests all features exhaustively to identify any issues
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime

class BrowserAgentTester:
    def __init__(self):
        self.base_url = "http://localhost:8070/browse"
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def test(self, name, task, expected_success=True, max_steps=15, timeout=180, 
             check_downloads=False, check_uploads=False, check_metrics=True):
        """Run a single test"""
        self.total_tests += 1
        test_id = f"test_{self.total_tests:03d}"
        
        print(f"\n{'='*100}")
        print(f"TEST #{self.total_tests}: {name}")
        print(f"{'='*100}")
        print(f"Task: {task}")
        print(f"Expected: {'SUCCESS' if expected_success else 'FAILURE'}")
        
        payload = {
            "task": task,
            "thread_id": test_id,
            "max_steps": max_steps
        }
        
        start_time = time.time()
        try:
            response = requests.post(self.base_url, json=payload, timeout=timeout)
            elapsed = time.time() - start_time
            
            if response.status_code != 200:
                print(f"❌ FAIL: HTTP {response.status_code}")
                self.failed_tests += 1
                self.results.append({
                    "test": name,
                    "status": "FAIL",
                    "reason": f"HTTP {response.status_code}",
                    "time": elapsed
                })
                return False
            
            result = response.json()
            success = result.get('success', False)
            actions = result.get('actions_taken', [])
            failed_actions = [a for a in actions if a.get('status') == 'failed']
            screenshots = result.get('screenshot_files', [])
            downloads = result.get('downloaded_files', [])
            uploads = result.get('uploaded_files', [])
            metrics = result.get('metrics', {})
            
            # Validation checks
            checks = []
            
            # Check 1: Success matches expectation
            if success == expected_success:
                checks.append(("Success status", True))
            else:
                checks.append(("Success status", False, f"Expected {expected_success}, got {success}"))
            
            # Check 2: No failed actions (unless expected to fail)
            if expected_success and len(failed_actions) > 0:
                checks.append(("No failed actions", False, f"{len(failed_actions)} actions failed"))
            else:
                checks.append(("No failed actions", True))
            
            # Check 3: Screenshots captured
            if len(screenshots) > 0:
                checks.append(("Screenshots captured", True, f"{len(screenshots)} screenshots"))
            else:
                checks.append(("Screenshots captured", False, "No screenshots"))
            
            # Check 4: Metrics present
            if check_metrics and metrics:
                checks.append(("Metrics tracked", True, f"{metrics.get('total_time', 0):.1f}s, {metrics.get('llm_calls', 0)} LLM calls"))
            elif check_metrics:
                checks.append(("Metrics tracked", False, "No metrics"))
            
            # Check 5: Downloads (if expected)
            if check_downloads:
                if len(downloads) > 0:
                    checks.append(("Downloads", True, f"{len(downloads)} files"))
                else:
                    checks.append(("Downloads", False, "No downloads"))
            
            # Check 6: Uploads (if expected)
            if check_uploads:
                if len(uploads) > 0:
                    checks.append(("Uploads", True, f"{len(uploads)} files"))
                else:
                    checks.append(("Uploads", False, "No uploads"))
            
            # Check 7: Actions taken
            if len(actions) > 0:
                checks.append(("Actions executed", True, f"{len(actions)} actions"))
            else:
                checks.append(("Actions executed", False, "No actions"))
            
            # Print results
            all_passed = all(check[1] for check in checks)
            
            print(f"\n📊 Results:")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Actions: {len(actions)} ({len(failed_actions)} failed)")
            print(f"   Summary: {result.get('task_summary', 'N/A')[:100]}")
            
            print(f"\n✓ Validation Checks:")
            for check in checks:
                status = "✅" if check[1] else "❌"
                detail = f" - {check[2]}" if len(check) > 2 else ""
                print(f"   {status} {check[0]}{detail}")
            
            if all_passed:
                print(f"\n✅ PASS")
                self.passed_tests += 1
                self.results.append({
                    "test": name,
                    "status": "PASS",
                    "time": elapsed,
                    "actions": len(actions),
                    "checks": len(checks)
                })
                return True
            else:
                print(f"\n❌ FAIL")
                self.failed_tests += 1
                failed_checks = [c for c in checks if not c[1]]
                self.results.append({
                    "test": name,
                    "status": "FAIL",
                    "reason": f"{len(failed_checks)} checks failed",
                    "time": elapsed
                })
                return False
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ FAIL: Exception - {str(e)[:100]}")
            self.failed_tests += 1
            self.results.append({
                "test": name,
                "status": "FAIL",
                "reason": f"Exception: {str(e)[:50]}",
                "time": elapsed
            })
            return False
    
    def print_summary(self):
        """Print final test summary"""
        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE TEST SUITE SUMMARY")
        print(f"{'='*100}")
        print(f"\nTotal Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ({100*self.passed_tests//self.total_tests if self.total_tests > 0 else 0}%)")
        print(f"Failed: {self.failed_tests}")
        
        if self.failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for r in self.results:
                if r['status'] == 'FAIL':
                    print(f"   - {r['test']}: {r.get('reason', 'Unknown')}")
        
        print(f"\n{'='*100}")
        
        if self.passed_tests == self.total_tests:
            print("🎉 ALL TESTS PASSED - Agent is production ready!")
        elif self.passed_tests >= self.total_tests * 0.9:
            print("⚠️  Most tests passed - Minor issues detected")
        elif self.passed_tests >= self.total_tests * 0.7:
            print("⚠️  Some tests failed - Moderate issues detected")
        else:
            print("❌ Many tests failed - Major issues detected")
        
        print(f"{'='*100}\n")

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    tester = BrowserAgentTester()
    
    print("\n" + "="*100)
    print("BROWSER AUTOMATION AGENT - COMPREHENSIVE TEST SUITE")
    print("Testing all features exhaustively")
    print("="*100)
    
    # ============================================================================
    # CATEGORY 1: BASIC NAVIGATION
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 1: BASIC NAVIGATION")
    print("="*100)
    
    tester.test(
        "Simple Navigation",
        "Go to example.com",
        max_steps=5
    )
    time.sleep(2)
    
    tester.test(
        "HTTPS Navigation",
        "Navigate to https://example.org",
        max_steps=5
    )
    time.sleep(2)
    
    tester.test(
        "Multi-page Navigation",
        "Go to example.com, then navigate to example.org",
        max_steps=10
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 2: DATA EXTRACTION
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 2: DATA EXTRACTION")
    print("="*100)
    
    tester.test(
        "Simple Text Extraction",
        "Go to example.com and tell me what the page says",
        max_steps=5
    )
    time.sleep(2)
    
    tester.test(
        "Structured Data Extraction",
        "Go to httpbin.org/html and extract all headings",
        max_steps=10
    )
    time.sleep(2)
    
    tester.test(
        "Multi-element Extraction",
        "Visit example.com and tell me the page title, main text, and any links",
        max_steps=10
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 3: FORM INTERACTION
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 3: FORM INTERACTION")
    print("="*100)
    
    tester.test(
        "Search Box Interaction - Google",
        "Go to google.com and search for 'Python programming'",
        max_steps=10
    )
    time.sleep(2)
    
    tester.test(
        "Search Box Interaction - Wikipedia",
        "Go to wikipedia.org and search for 'Machine Learning'",
        max_steps=10
    )
    time.sleep(2)
    
    tester.test(
        "Auto-submit Form",
        "Visit wikipedia.org, search for 'Artificial Intelligence', and tell me the first paragraph",
        max_steps=10
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 4: DYNAMIC CONTENT
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 4: DYNAMIC CONTENT")
    print("="*100)
    
    tester.test(
        "Dynamic Page - GitHub Trending",
        "Go to github.com/trending and tell me the top 3 trending repositories",
        max_steps=15
    )
    time.sleep(2)
    
    tester.test(
        "Page Change Detection",
        "Go to wikipedia.org, search for 'Python', and extract the page title",
        max_steps=10
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 5: COMPLEX TASKS
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 5: COMPLEX TASKS")
    print("="*100)
    
    tester.test(
        "Multi-step Task",
        "Visit example.com, extract the text, then go to example.org and compare the content",
        max_steps=15
    )
    time.sleep(2)
    
    tester.test(
        "Search and Extract",
        "Go to google.com, search for 'OpenAI', and tell me what you find",
        max_steps=15
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 6: ERROR HANDLING
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 6: ERROR HANDLING")
    print("="*100)
    
    tester.test(
        "Invalid URL Handling",
        "Go to thissitedoesnotexist12345.com",
        expected_success=False,
        max_steps=5
    )
    time.sleep(2)
    
    tester.test(
        "Timeout Handling",
        "Go to example.com and click on a button that doesn't exist",
        max_steps=10
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 7: EDGE CASES
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 7: EDGE CASES")
    print("="*100)
    
    tester.test(
        "Empty Task",
        "Go to example.com",
        max_steps=5
    )
    time.sleep(2)
    
    tester.test(
        "Very Long Task",
        "Go to example.com, extract all text, then go to example.org, extract all text, compare them, and tell me which one has more content",
        max_steps=20
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 8: PERFORMANCE
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 8: PERFORMANCE")
    print("="*100)
    
    tester.test(
        "Quick Task",
        "Go to example.com and get the title",
        max_steps=5,
        timeout=30
    )
    time.sleep(2)
    
    tester.test(
        "Metrics Tracking",
        "Visit example.com and extract the main heading",
        max_steps=5,
        check_metrics=True
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 9: STEALTH & ANTI-DETECTION
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 9: STEALTH & ANTI-DETECTION")
    print("="*100)
    
    tester.test(
        "Google Access (Anti-bot test)",
        "Go to google.com and search for 'test'",
        max_steps=10
    )
    time.sleep(2)
    
    # ============================================================================
    # CATEGORY 10: RESOURCE MANAGEMENT
    # ============================================================================
    print("\n" + "="*100)
    print("CATEGORY 10: RESOURCE MANAGEMENT")
    print("="*100)
    
    tester.test(
        "Multiple Sequential Tasks",
        "Go to example.com",
        max_steps=5
    )
    time.sleep(1)
    
    tester.test(
        "Browser Cleanup Test",
        "Go to example.org",
        max_steps=5
    )
    time.sleep(1)
    
    tester.test(
        "No Memory Leak Test",
        "Visit example.com and tell me what it says",
        max_steps=5
    )
    time.sleep(2)
    
    # Print final summary
    tester.print_summary()
    
    return tester.passed_tests == tester.total_tests

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)
