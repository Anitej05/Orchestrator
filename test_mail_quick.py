"""
Quick test to verify mail agent can start and respond to health checks
"""

import subprocess
import time
import sys
from pathlib import Path
import httpx

print("Testing Mail Agent Startup")
print("="*60)

# Start the agent
cmd = [
    sys.executable, '-m', 'uvicorn',
    'agents.mail_agent.__init__:app',
    '--host', '0.0.0.0',
    '--port', '8042',
    '--log-level', 'info',
]

print(f"\n1. Starting Mail Agent on port 8042...")
process = subprocess.Popen(
    cmd,
    cwd='backend',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
print(f"   PID: {process.pid}")

# Wait for startup
print(f"\n2. Waiting 15 seconds for startup...")
time.sleep(15)

# Check if running
if process.poll() is not None:
    print(f"   [FAIL] Process exited early!")
    stdout, stderr = process.communicate()
    print(f"\n   STDOUT:\n{stdout.decode()[-1000:]}")
    print(f"\n   STDERR:\n{stderr.decode()[-1000:]}")
else:
    print(f"   [OK] Process still running")
    
    # Try health check with longer timeout
    print(f"\n3. Testing health endpoint...")
    try:
        response = httpx.get('http://localhost:8042/health', timeout=10)
        print(f"   [OK] Status: {response.status_code}")
        print(f"   [OK] Response: {response.text}")
    except Exception as e:
        print(f"   [FAIL] {e}")

# Cleanup
print(f"\n4. Stopping agent...")
process.terminate()
try:
    process.wait(timeout=5)
    print(f"   [OK] Stopped")
except:
    process.kill()
    print(f"   [OK] Killed")

print("\n" + "="*60)
print("Test complete")
