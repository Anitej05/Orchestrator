"""
Simple Mail Agent test - check if it can start and respond
"""

import subprocess
import time
import sys
from pathlib import Path
import os

# Set PYTHONPATH
os.environ['PYTHONPATH'] = str(Path('backend').parent.absolute())

print("Starting Mail Agent...")
cmd = [
    sys.executable, '-m', 'uvicorn',
    'agents.mail_agent.__init__:app',
    '--host', '0.0.0.0',
    '--port', '8044',
]

process = subprocess.Popen(
    cmd,
    cwd='backend',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=os.environ.copy(),
)

print(f"PID: {process.pid}")
print("Waiting 20 seconds...")
time.sleep(20)

if process.poll() is not None:
    print(f"Process exited with code {process.returncode}")
    output = process.stdout.read().decode()
    print(f"\nOutput:\n{output}")
else:
    print("Process still running")
    # Try curl
    import urllib.request
    try:
        response = urllib.request.urlopen('http://localhost:8044/health', timeout=5)
        print(f"Health check: {response.read().decode()}")
    except Exception as e:
        print(f"Health check failed: {e}")

process.terminate()
