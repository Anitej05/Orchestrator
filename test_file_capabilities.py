"""Verification tests for Universal Agent file system capabilities."""
import requests
import json

BASE = "http://localhost:8070"

# 1. Check capabilities list
print("="*60)
print("1. CAPABILITIES CHECK")
print("="*60)
r = requests.get(f"{BASE}/capabilities", timeout=10)
caps = r.json()
cap_names = [c["name"] for c in caps.get("capabilities", [])]
expected_file_caps = ["read_file", "write_file", "list_directory", "search_files", "manage_file"]
for cap in expected_file_caps:
    status = "FOUND" if cap in cap_names else "MISSING"
    print(f"  {cap}: {status}")
print(f"\nTotal capabilities: {len(cap_names)}")
print(f"All file caps present: {all(c in cap_names for c in expected_file_caps)}")

# 2. Test read_file
print("\n" + "="*60)
print("2. READ_FILE TEST")
print("="*60)
r = requests.post(f"{BASE}/execute", json={
    "prompt": "d:/Internship/Orbimesh/README.md",
    "action": "read_file"
}, timeout=30)
d = r.json()
result = d.get("result")
has_content = result is not None and len(str(result)) > 10
print(f"Status: {d['status']} | Has content: {has_content} | Length: {len(str(result)) if result else 0}")
if has_content:
    print(f"Preview: {str(result)[:150]}...")
else:
    print(f"Result: {result}")
    print(f"Data: {str(d.get('data'))[:200]}")

# 3. Test list_directory
print("\n" + "="*60)
print("3. LIST_DIRECTORY TEST")
print("="*60)
r = requests.post(f"{BASE}/execute", json={
    "prompt": "d:/Internship/Orbimesh/backend/agents",
    "action": "list_directory"
}, timeout=30)
d = r.json()
result = d.get("result")
has_content = result is not None and len(str(result)) > 10
print(f"Status: {d['status']} | Has content: {has_content} | Length: {len(str(result)) if result else 0}")
if has_content:
    print(f"Preview: {str(result)[:200]}...")

# 4. Test search_files
print("\n" + "="*60)
print("4. SEARCH_FILES TEST")
print("="*60)
r = requests.post(f"{BASE}/execute", json={
    "prompt": "*.py",
    "action": "search_files",
    "payload": {"directory": "d:/Internship/Orbimesh/backend/agents/universal_agent"}
}, timeout=30)
d = r.json()
result = d.get("result")
has_content = result is not None and len(str(result)) > 10
print(f"Status: {d['status']} | Has content: {has_content} | Length: {len(str(result)) if result else 0}")
if has_content:
    print(f"Preview: {str(result)[:200]}...")

# 5. Test manage_file (info)
print("\n" + "="*60)
print("5. MANAGE_FILE (info) TEST")
print("="*60)
r = requests.post(f"{BASE}/execute", json={
    "prompt": "d:/Internship/Orbimesh/README.md",
    "action": "manage_file",
    "payload": {"operation": "info"}
}, timeout=30)
d = r.json()
result = d.get("result")
has_content = result is not None and len(str(result)) > 5
print(f"Status: {d['status']} | Has content: {has_content} | Length: {len(str(result)) if result else 0}")
if has_content:
    print(f"Preview: {str(result)[:200]}...")

# 6. Test write_file
print("\n" + "="*60)
print("6. WRITE_FILE TEST")
print("="*60)
r = requests.post(f"{BASE}/execute", json={
    "action": "write_file",
    "payload": {
        "file_path": "d:/Internship/Orbimesh/storage/universal_agent/test_output.txt",
        "content": "Hello from Universal Agent! File write test successful."
    }
}, timeout=30)
d = r.json()
result = d.get("result")
has_content = result is not None and len(str(result)) > 5
print(f"Status: {d['status']} | Has content: {has_content}")
if has_content:
    print(f"Result: {str(result)[:200]}")

# 7. Run original capability tests too
print("\n" + "="*60)
print("7. ORIGINAL CAPABILITIES (smoke test)")
print("="*60)
for action, prompt in [("solve_problem", "What is 2+2?"), ("analyze", "Compare Python and Java briefly.")]:
    r = requests.post(f"{BASE}/execute", json={"prompt": prompt, "action": action}, timeout=30)
    d = r.json()
    result = d.get("result")
    status = "PASS" if result and len(str(result)) > 10 else "FAIL"
    print(f"  {action}: {status} ({len(str(result)) if result else 0} chars)")

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)
