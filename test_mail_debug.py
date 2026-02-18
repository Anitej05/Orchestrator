"""
Debug Mail Agent binding
"""

import subprocess
import time
import sys
from pathlib import Path
import socket

print("Debug Mail Agent Binding")
print("="*60)

# Set PYTHONPATH
import os
os.environ['PYTHONPATH'] = str(Path('backend').parent.absolute())
print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

# Start the agent
cmd = [
    sys.executable, '-m', 'uvicorn',
    'agents.mail_agent.__init__:app',
    '--host', '0.0.0.0',
    '--port', '8043',
]

print(f"\n1. Starting Mail Agent...")
process = subprocess.Popen(
    cmd,
    cwd='backend',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=os.environ.copy(),
)

print(f"   PID: {process.pid}")

# Read output for 20 seconds
print(f"\n2. Reading startup output (20s)...")
start_time = time.time()
output_lines = []

while time.time() - start_time < 20:
    import select
    import sys
    
    # Check if process is still running
    if process.poll() is not None:
        print(f"   Process exited with code {process.returncode}")
        break
    
    # Try to read output
    import io
    line = process.stdout.readline()
    if line:
        line_str = line.decode().strip()
        output_lines.append(line_str)
        print(f"   > {line_str}")
    else:
        time.sleep(0.1)

# Check port binding
print(f"\n3. Checking if port 8043 is bound...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
result = sock.connect_ex(('localhost', 8043))
if result == 0:
    print(f"   [OK] Port 8043 is bound")
else:
    print(f"   [FAIL] Port 8043 not accessible (error {result})")
sock.close()

# Cleanup
print(f"\n4. Stopping...")
process.terminate()
try:
    process.wait(timeout=5)
except:
    process.kill()

print("\n" + "="*60)
