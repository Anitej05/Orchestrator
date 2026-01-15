
import requests
import json
import time

URL = "http://localhost:8040/execute"
PROMPT = "Can you download the attachment attached to this mail: Purchase Order PO-2026-156 - Urgent Supplier Confirmation Needed"

def test_download():
    print(f"Testing Mail Agent at {URL}")
    print(f"Prompt: {PROMPT}")
    
    payload = {
        "prompt": PROMPT,
        "action": None
    }
    
    try:
        start_time = time.time()
        response = requests.post(URL, json=payload, timeout=120)
        duration = time.time() - start_time
        
        print(f"Request completed in {duration:.2f}s")
        print(f"Status Code: {response.status_code}")
        
        try:
            data = response.json()
            print("Response Data:")
            print(json.dumps(data, indent=2))
            
            # Check for success
            if data.get("status") == "complete":
                result = data.get("result", {})
                results_list = result.get("results", [])
                
                # Check execution steps
                files_downloaded = 0
                for step in results_list:
                    step_name = step.get("step")
                    step_res = step.get("result", {})
                    print(f"Step '{step_name}': {step_res.get('success')}")
                    
                    if step_name == "download_attachments":
                        if step_res.get("success"):
                            files = step_res.get("files", [])
                            files_downloaded = len(files)
                            print(f"Files downloaded: {files_downloaded}")
                            if files_downloaded > 0:
                                print("✅ SUCCESS: Attachment downloaded!")
                            else:
                                print("❌ FAILURE: Success=True but 0 files downloaded.")
                        else:
                            print(f"❌ FAILURE: Step failed with error: {step_res.get('error')}")
                            
        except json.JSONDecodeError:
            print("Failed to decode JSON response")
            print(response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_download()
