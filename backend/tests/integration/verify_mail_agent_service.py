
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import json

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
from backend.agents.mail_agent.agent import send_email
from backend.agents.mail_agent.schemas import SendEmailRequest

async def test_agent_send_with_native_attachment():
    print("Starting Mail Agent Logic Test (Native Attachment)")
    
    # 1. Create Dummy Attachment
    test_file = Path("test_upload_service.txt")
    with open(test_file, "w") as f:
        f.write("This is a test file for Mail Agent logic verification.")
    print(f"Created test file: {test_file.absolute()}")

    try:
        # 2. Define Request Object
        recipient = "anitej473@gmail.com"
        subject = "Mail Agent Logic Test - Native Attachment"
        body = "This email was sent via the Mail Agent send_email function directly using a native file attachment."
        
        request = SendEmailRequest(
            to=[recipient],
            subject=subject,
            body=body,
            attachment_paths=[str(test_file.absolute())],
            user_id="me"
        )
        
        print(f"Calling send_email with payload: {request.model_dump()}")
        
        # 3. Call Logic
        result = await send_email(request)
        
        print(f"Result: {result}")
        
        if result.success:
            msg = "SUCCESS: Mail Agent successfully handled the send request!"
            print(msg)
            with open("agent_test_result.txt", "w") as f: f.write(msg)
        else:
            msg = f"FAILURE: Agent returned error: {result.error}"
            print(msg)
            with open("agent_test_result.txt", "w") as f: f.write(msg)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = f"EXCEPTION during agent test: {e}"
        print(msg)
        with open("agent_test_result.txt", "w") as f: f.write(msg)
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
            print("Cleaned up test file")

if __name__ == "__main__":
    asyncio.run(test_agent_send_with_native_attachment())
