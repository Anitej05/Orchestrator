import requests
import time

time.sleep(3)
r = requests.post('http://localhost:8000/api/chat', json={
    'prompt': 'Go to example.com and tell me the heading',
    'thread_id': f'quick_{int(time.time())}',
    'planning_mode': False
}, timeout=120)

print(f'Status: {r.status_code}')
result = r.json()
print(f'Has canvas: {result.get("has_canvas")}')
print(f'Browser view: {"browser_view" in result}')
print(f'Plan view: {"plan_view" in result}')
response = result.get('final_response', '')
if response:
    print(f'Response: {response[:150]}...')
else:
    print('No response')
