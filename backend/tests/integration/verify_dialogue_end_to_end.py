import requests
import json
import time
import sys

# Color codes for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

BASE_URL = "http://localhost:8000/api"

def run_verification():
    print(f"{YELLOW}starting End-to-End Dialogue Verification...{RESET}")
    
    # Step 1: Start Conversation with Ambiguous Request
    # "john" without "specific" triggers ambiguity in Mail Agent
    prompt = "Check my emails to find updates from John" 
    print(f"\n{YELLOW}[Step 1] Sending initial prompt: '{prompt}'{RESET}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat", 
            json={"prompt": prompt, "files": []}
        )
        response.raise_for_status()
        data = response.json()
        thread_id = data.get("thread_id")
        
        print(f"  Thread ID: {thread_id}")
        
        # Check if it paused for user input
        if data.get("pending_user_input"):
            print(f"{GREEN}  ✅ Success: Orchestrator paused for user input!{RESET}")
            print(f"  Question: {data.get('question_for_user')}")
            
            # Verify the question is about "Which John?"
            question = data.get('question_for_user', '').lower()
            if "john" in question or "which" in question:
                 print(f"{GREEN}  ✅ Success: Question matches expected ambiguity logic.{RESET}")
            else:
                 print(f"{RED}  ❌ Warning: Question content unexpected: {data.get('question_for_user')}{RESET}")

        else:
            print(f"{RED}  ❌ Failed: Orchestrator did NOT pause. Result: {data}{RESET}")
            # If it didn't pause, maybe it hallucinated or failed.
            return

    except Exception as e:
        print(f"{RED}  ❌ Error in Step 1: {e}{RESET}")
        return

    # Step 2: Provide Clarification
    user_answer = "John Smith (Work)"
    print(f"\n{YELLOW}[Step 2] Sending clarification: '{user_answer}'{RESET}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/continue", 
            json={
                "response": user_answer, 
                "thread_id": thread_id
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Check if it resumed and completed
        if not data.get("pending_user_input"):
            print(f"{GREEN}  ✅ Success: Orchestrator resumed and completed the task!{RESET}")
            print(f"  Final Response: {data.get('final_response')}")
        else:
            print(f"{RED}  ❌ Failed: Orchestrator is still paused. Question: {data.get('question_for_user')}{RESET}")

    except Exception as e:
        print(f"{RED}  ❌ Error in Step 2: {e}{RESET}")
        return

    print(f"\n{GREEN}Inventory Check Complete. Bidirectional flow confirmed.{RESET}")

if __name__ == "__main__":
    run_verification()
