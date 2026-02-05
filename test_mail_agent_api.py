import requests
import json
import time

BASE_URL = "http://localhost:8040"

def test_health():
    print("Testing /health")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 20)

def test_search(query):
    print(f"Testing /search: {query}")
    payload = {"query": query, "max_results": 10, "user_id": "me"}
    response = requests.post(f"{BASE_URL}/search", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()
    count = len(data.get("result", {}).get("messages", []))
    print(f"Found {count} messages")
    print("-" * 20)
    return data

def test_get_message(message_id):
    print(f"Testing /get_message: {message_id}")
    # SCHEMA FIX: Must include 'operation'
    payload = {
        "operation": "get_message",
        "parameters": {"message_id": message_id}
    }
    response = requests.post(f"{BASE_URL}/get_message", json=payload)
    print(f"Status: {response.status_code}")
    res_data = response.json()
    if response.status_code == 200:
        subject = res_data.get("result", {}).get("subject", "No Subject")
        print(f"Subject: {subject}")
    else:
        print(f"Error: {res_data}")
    print("-" * 20)
    return res_data

def test_summarize(message_ids):
    print(f"Testing /summarize_emails for {len(message_ids)} IDs")
    payload = {"message_ids": message_ids, "user_id": "me"}
    response = requests.post(f"{BASE_URL}/summarize_emails", json=payload)
    print(f"Status: {response.status_code}")
    summary = response.json().get("result", {}).get("summary", "")
    print(f"Summary length: {len(summary)}")
    print("-" * 20)

def test_extract_actions(message_ids):
    print(f"Testing /extract_action_items for {len(message_ids)} IDs")
    payload = {"message_ids": message_ids, "user_id": "me"}
    response = requests.post(f"{BASE_URL}/extract_action_items", json=payload)
    print(f"Status: {response.status_code}")
    actions = response.json().get("result", {}).get("actions", [])
    print(f"Extracted {len(actions)} actions")
    print("-" * 20)

def test_manage_emails(message_ids, action):
    print(f"Testing /manage_emails: {action} for {len(message_ids)} IDs")
    payload = {"message_ids": message_ids, "action": action, "user_id": "me"}
    response = requests.post(f"{BASE_URL}/manage_emails", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Success: {response.json().get('success')}")
    print("-" * 20)

def test_download_attachments(message_id):
    print(f"Testing /download_attachments: {message_id}")
    payload = {"message_id": message_id, "user_id": "me"}
    response = requests.post(f"{BASE_URL}/download_attachments", json=payload)
    print(f"Status: {response.status_code}")
    files = response.json().get("result", {}).get("files", [])
    print(f"Downloaded {len(files)} files")
    print("-" * 20)

def test_send_and_reply():
    print("Testing /send_email and /draft_reply sequence")
    # 1. Send
    send_payload = {
        "to": ["anitej473@gmail.com"],
        "subject": "Final End-to-End Mail Agent Test",
        "body": "Verifying all fixes are applied.",
        "user_id": "me"
    }
    send_res = requests.post(f"{BASE_URL}/send_email", json=send_payload).json()
    if not send_res.get("success"):
        print("Send failed!")
        return
    
    msg_id = send_res["result"]["message_id"]
    print(f"Sent email ID: {msg_id}")
    
    # 2. Draft Reply
    reply_payload = {
        "message_id": msg_id,
        "intent": "Confirm that the fix for JSON parsing is working.",
        "user_id": "me"
    }
    reply_res = requests.post(f"{BASE_URL}/draft_reply", json=reply_payload).json()
    print(f"Draft success: {reply_res.get('success')}")
    if reply_res.get("success"):
        print(f"Draft Subject: {reply_res['result']['subject']}")
    print("-" * 20)

if __name__ == "__main__":
    test_health()
    
    # Search for context
    search_data = test_search("subject:test")
    messages = search_data.get("result", {}).get("messages", [])
    
    if messages:
        # Test Details on the first one
        mid = messages[0]["id"]
        test_get_message(mid)
        
        # Find one with attachments for download test
        msg_with_attachments = None
        for m in messages:
             # Some search results include attachment info in snippet or snippet might be empty if we didn't fetch full
             # We'll just try to download from all until we find one or finish the list
             pass
        
        # Test Text Analysis
        test_summarize([mid])
        test_extract_actions([mid])
        
        # Test Management
        test_manage_emails([mid], "star")
        test_manage_emails([mid], "unstar")
        
        # Test Attachments (try the first few)
        for i in range(min(3, len(messages))):
             test_download_attachments(messages[i]["id"])
    
    # Test Full Flow
    test_send_and_reply()
    
    print("=== TEST COMPLETE ===")
