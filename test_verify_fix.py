"""Quick verification that all direct dispatch capabilities return real content."""
import requests
import json

BASE = "http://localhost:8070/execute"

tests = [
    ("solve_problem", "What is the sum of the first 10 prime numbers? Show your work."),
    ("analyze", "Analyze the pros and cons of microservices vs monolithic architecture."),
    ("research", "What are the key differences between REST and GraphQL APIs?"),
    ("creative_write", "Write a short 4-line poem about debugging code at midnight."),
    ("generate_code", "Write a Python function that checks if a string is a valid palindrome."),
]

all_pass = True
for action, prompt in tests:
    print(f"\n--- {action} ---")
    r = requests.post(BASE, json={"prompt": prompt, "action": action}, timeout=60)
    d = r.json()
    result = d.get("result")
    has_content = result is not None and len(str(result)) > 10
    status = d["status"]
    result_len = len(str(result)) if result else 0
    
    if has_content:
        print(f"PASS | Status: {status} | Result length: {result_len}")
        print(f"Preview: {str(result)[:200]}...")
    else:
        print(f"FAIL | Status: {status} | Result: {result}")
        print(f"Data field: {str(d.get('data'))[:200]}")
        all_pass = False

print("\n" + "="*50)
if all_pass:
    print("ALL TESTS PASSED - Bug is fixed!")
else:
    print("SOME TESTS FAILED - Bug still present!")
