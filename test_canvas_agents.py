"""
Canvas Integration Test Script
Tests spreadsheet, document, and universal agents with real tasks.
Checks if canvas_display is properly generated in API responses.
"""

import urllib.request
import json
import time
import sys

BASE = "http://localhost:8000"

def post_chat(prompt, files=None):
    """Send a chat request and return the response."""
    payload = {"prompt": prompt, "files": files or []}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def check_canvas(response, test_name):
    """Check if canvas data is present in the response."""
    has_canvas = response.get("has_canvas", False)
    canvas_type = response.get("canvas_type")
    canvas_data = response.get("canvas_data")
    canvas_registry = response.get("canvas_registry")
    error = response.get("error")

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    if error:
        print(f"  ERROR: {error}")
        return False

    # Check final_response
    final_resp = response.get("final_response", "")
    if final_resp:
        print(f"  Response: {str(final_resp)[:150]}...")
    
    # Check task_agent_pairs
    pairs = response.get("task_agent_pairs", [])
    for pair in pairs:
        agent = pair.get("primary_agent", {}).get("agent_name", "?")
        task = pair.get("task_name", "?")
        result = pair.get("result", {})
        print(f"  Agent: {agent} | Task: {task}")
        
        # Check for canvas in result
        if isinstance(result, dict):
            cd = result.get("canvas_display")
            if cd:
                print(f"  >> CANVAS IN RESULT: type={cd.get('canvas_type')}, title={cd.get('canvas_title')}")
                if cd.get('canvas_data'):
                    keys = list(cd['canvas_data'].keys())
                    print(f"     canvas_data keys: {keys}")

    # Top-level canvas check
    print(f"  has_canvas: {has_canvas}")
    print(f"  canvas_type: {canvas_type}")
    if canvas_data:
        print(f"  canvas_data keys: {list(canvas_data.keys()) if isinstance(canvas_data, dict) else 'raw'}")
    
    # Registry check
    if canvas_registry:
        canvases = canvas_registry.get("canvases", {})
        active = canvas_registry.get("active_canvas_id")
        print(f"  Registry: {len(canvases)} canvases, active={active}")
        for cid, entry in canvases.items():
            print(f"    [{cid}] type={entry.get('canvas_type')}, agent={entry.get('source_agent')}")
    
    success = has_canvas or bool(canvas_registry and canvas_registry.get("canvases"))
    print(f"  RESULT: {'PASS - Canvas found' if success else 'FAIL - No canvas'}")
    return success


def main():
    print("Canvas Integration Tests")
    print("=" * 60)
    
    results = {}
    
    # ---------------------------------------------------------------
    # TEST 1: Spreadsheet Agent — Create a spreadsheet
    # ---------------------------------------------------------------
    print("\n[1/4] Testing Spreadsheet Agent...")
    resp = post_chat(
        "Create a spreadsheet with employee data: "
        "Name, Department, Salary columns. "
        "Add 5 employees from Engineering, Marketing, and Sales departments."
    )
    results["Spreadsheet: Create"] = check_canvas(resp, "Spreadsheet Agent - Create Data")

    time.sleep(2)

    # ---------------------------------------------------------------
    # TEST 2: Document Agent — Create a document
    # ---------------------------------------------------------------
    print("\n[2/4] Testing Document Agent...")
    resp = post_chat(
        "Create a markdown document titled 'Project Status Report' with sections: "
        "Executive Summary, Key Milestones, Risks and Mitigation, Next Steps."
    )
    results["Document: Create"] = check_canvas(resp, "Document Agent - Create Document")

    time.sleep(2)

    # ---------------------------------------------------------------
    # TEST 3: Universal Agent — Code generation
    # ---------------------------------------------------------------
    print("\n[3/4] Testing Universal Agent (code)...")
    resp = post_chat(
        "Write a Python function that calculates the fibonacci sequence up to n terms "
        "and returns the result as a list."
    )
    results["Universal: Code"] = check_canvas(resp, "Universal Agent - Code Generation")

    time.sleep(2)

    # ---------------------------------------------------------------
    # TEST 4: Universal Agent — Analysis
    # ---------------------------------------------------------------
    print("\n[4/4] Testing Universal Agent (analysis)...")
    resp = post_chat(
        "Analyze the pros and cons of microservices vs monolithic architecture. "
        "Provide a detailed comparison covering scalability, deployment, complexity, "
        "team organization, and testing strategies."
    )
    results["Universal: Analysis"] = check_canvas(resp, "Universal Agent - Analysis")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} | {name}")
    print(f"\n  {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
