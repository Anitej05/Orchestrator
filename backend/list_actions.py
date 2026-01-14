
import os
from composio import Composio, Action
from dotenv import load_dotenv

load_dotenv()

def list_all_actions():
    api_key = os.getenv("COMPOSIO_API_KEY")
    client = Composio(api_key=api_key)
    
    print("Searching for upload related actions...")
    # This might be slow or not possible directly on Action enum
    # but let's try to see if we can find anything with 'UPLOAD' in name
    
    count = 0
    for action in Action:
        if "UPLOAD" in action.name or "FILE" in action.name:
            print(f"Action: {action.name}")
            count += 1
            if count > 50: break
    
    print(f"Total relevant actions found: {count}")

if __name__ == "__main__":
    list_all_actions()
