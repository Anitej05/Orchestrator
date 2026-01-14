
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Load env vars
load_dotenv()

# Import the client
# Note: The file defines 'GmailClient' but usually it's used as MailClient or similar.
# Based on the file path 'agents/mail_agent/client.py', and previous view, the class is GmailClient.
try:
    from backend.agents.mail_agent.client import GmailClient
except ImportError:
    # Try relative import if running from root
    sys.path.append(os.getcwd())
    from agents.mail_agent.client import GmailClient

async def test_native_upload():
    try:
        print("🚀 Starting Native Upload E2E Test")
        
        # 1. Initialize Client
        try:
            client = GmailClient()
            print("✅ Client initialized")
        except Exception as e:
            msg = f"❌ Client init failed: {e}"
            print(msg)
            with open("test_result.txt", "w") as f: f.write(msg)
            return

        # 2. Create Dummy Attachment
        test_file = Path("test_upload_native.txt")
        with open(test_file, "w") as f:
            f.write("This is a test file for native upload verification via Composio SDK 0.10.6+")
        print(f"✅ Created test file: {test_file.absolute()}")

        # 3. Define Email Params
        recipient = "annaborbimesh@gmail.com"
        subject = "E2E Native Upload Test"
        body = "This email should contain a native file attachment (test_upload_native.txt)."
        
        # 4. Call send_email_with_attachments
        # We pass the absolute path to be sure
        attachment_paths = [str(test_file.absolute())]
        
        print(f"📧 Sending email to {recipient} with attachment {attachment_paths}...")
        
        result = await client.send_email_with_attachments(
            to=[recipient],
            subject=subject,
            body=body,
            attachment_file_ids=[], 
            attachment_paths=attachment_paths
        )
        
        print(f"🔍 Result: {result}")
        
        if result.get("success"):
            msg = "✅ Email sent successfully!"
            print(msg)
            with open("test_result.txt", "w") as f: f.write(msg)
        else:
            msg = f"❌ Email failed: {result.get('error')}"
            print(msg)
            with open("test_result.txt", "w") as f: f.write(msg)
            
    except Exception as e:
        msg = f"❌ Exception during send: {e}"
        print(msg)
        with open("test_result.txt", "w") as f: f.write(msg)
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'test_file' in locals() and test_file.exists():
            test_file.unlink()
            print("🧹 Cleaned up test file")

if __name__ == "__main__":
    try:
        asyncio.run(test_native_upload())
    except Exception as e:
        with open("test_result.txt", "w") as f: f.write(f"❌ Crash in main: {e}")
