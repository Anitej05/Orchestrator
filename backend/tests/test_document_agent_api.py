import requests
import json
import os
import sys
import io
from pathlib import Path

# Fix Unicode issues on Windows consoles
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:8070"
STORAGE_DIR = Path("storage/documents")
SAMPLE_DOC = STORAGE_DIR / "agi_architecture_v3.docx"

def log_response(name, resp):
    print(f"\n--- {name} ---")
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(f"Response Body:\n{json.dumps(data, indent=2)}")
        return data
    except:
        print(f"Response Body (Raw):\n{resp.text}")
        return None

def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    log_response("HEALTH", resp)
    assert resp.status_code == 200

def test_metrics():
    resp = requests.get(f"{BASE_URL}/metrics")
    log_response("METRICS", resp)
    assert resp.status_code == 200

def test_upload():
    if not SAMPLE_DOC.exists():
        print(f"Skipping upload test: {SAMPLE_DOC} not found")
        return None
    
    with open(SAMPLE_DOC, "rb") as f:
        files = {"files": (SAMPLE_DOC.name, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        resp = requests.post(f"{BASE_URL}/upload", files=files)
    
    data = log_response("UPLOAD", resp)
    assert resp.status_code == 200
    return data["files"][0]["file_path"]

def test_create():
    payload = {
        "title": "Test AI Report",
        "file_name": "test_ai_report_v2.docx",
        "content": "This is a detailed test report about AGIs and their architectural layers. It discusses sensors, processors, and effectors.",
        "format": "docx",
        "author": "Antigravity"
    }
    resp = requests.post(f"{BASE_URL}/create", json=payload)
    data = log_response("CREATE", resp)
    assert resp.status_code == 200
    return data["file_path"]

def test_analyze(file_path):
    payload = {
        "query": "Summarize the key architectural layers in 3 points.",
        "file_path": file_path
    }
    resp = requests.post(f"{BASE_URL}/analyze", json=payload)
    data = log_response("ANALYZE", resp)
    assert resp.status_code == 200
    return data

def test_display(file_path):
    payload = {
        "file_path": file_path
    }
    resp = requests.post(f"{BASE_URL}/display", json=payload)
    log_response("DISPLAY", resp)
    assert resp.status_code == 200

def test_edit(file_path):
    payload = {
        "file_path": file_path,
        "instruction": "Add a paragraph about the importance of 'Safety and Ethics' in AI development.",
        "auto_approve": True
    }
    resp = requests.post(f"{BASE_URL}/edit", json=payload)
    data = log_response("EDIT", resp)
    assert resp.status_code == 200
    return data

def test_extract(file_path):
    payload = {
        "file_path": file_path,
        "extraction_type": "text"
    }
    resp = requests.post(f"{BASE_URL}/extract", json=payload)
    data = log_response("EXTRACT", resp)
    assert resp.status_code == 200
    return data

def test_versions(file_path):
    payload = {
        "file_path": file_path
    }
    resp = requests.post(f"{BASE_URL}/versions", json=payload)
    log_response("VERSIONS", resp)
    assert resp.status_code == 200

def test_undo_redo(file_path):
    payload = {
        "file_path": file_path,
        "action": "undo"
    }
    resp = requests.post(f"{BASE_URL}/undo-redo", json=payload)
    log_response("UNDO", resp)
    assert resp.status_code == 200

def test_execute_analyze(file_path):
    payload = {
        "type": "execute",
        "action": "/analyze",
        "payload": {
            "query": "What is the main topic?",
            "file_path": file_path
        }
    }
    resp = requests.post(f"{BASE_URL}/execute", json=payload)
    log_response("EXECUTE - ANALYZE", resp)
    assert resp.status_code == 200

def main():
    print("=== Document Agent COMPREHENSIVE Response Test ===")
    try:
        test_health()
        
        # 1. Create a fresh document for consistent testing
        created_path = test_create()
        print(f"Targeting: {created_path}")
        
        # 2. Test analysis on the created content
        test_analyze(created_path)
        
        # 3. Test display
        test_display(created_path)
        
        # 4. Test extraction
        test_extract(created_path)
        
        # 5. Test editing
        test_edit(created_path)
        
        # 6. Test version control
        test_versions(created_path)
        test_undo_redo(created_path)
        
        # 7. Test Orchestrator flow
        test_execute_analyze(created_path)
        
        print("\n✅ ALL COMPREHENSIVE TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
