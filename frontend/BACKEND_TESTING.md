# Backend Endpoint Tester

This is a simple testing utility to check what endpoints are available on the backend.

## Testing Script

```bash
#!/bin/bash

echo "Testing backend endpoints..."

# Test health endpoint
echo "Testing /api/health:"
curl -s -w "Status: %{http_code}\n" http://localhost:8000/api/health || echo "Failed to connect"

echo ""

# Test basic chat endpoint
echo "Testing /chat:"
curl -s -w "Status: %{http_code}\n" -X POST -H "Content-Type: application/json" -d '{"prompt": "test"}' http://localhost:8000/chat || echo "Failed to connect"

echo ""

# Test interactive chat endpoint
echo "Testing /api/chat:"
curl -s -w "Status: %{http_code}\n" -X POST -H "Content-Type: application/json" -d '{"prompt": "test", "max_results": 5}' http://localhost:8000/api/chat || echo "Failed to connect"

echo ""

# Test WebSocket (this will just check if the port responds)
echo "Testing WebSocket /ws/chat:"
curl -s -w "Status: %{http_code}\n" http://localhost:8000/ws/chat || echo "Failed to connect"

echo ""

# Test agents endpoint
echo "Testing /api/agents/all:"
curl -s -w "Status: %{http_code}\n" http://localhost:8000/api/agents/all || echo "Failed to connect"
```

## PowerShell Version

```powershell
Write-Host "Testing backend endpoints..."

# Test health endpoint
Write-Host "Testing /api/health:"
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -Method GET
    Write-Host "Status: $($response.StatusCode)"
} catch {
    Write-Host "Failed: $($_.Exception.Message)"
}

Write-Host ""

# Test basic chat endpoint
Write-Host "Testing /chat:"
try {
    $body = @{prompt = "test"} | ConvertTo-Json
    $response = Invoke-WebRequest -Uri "http://localhost:8000/chat" -Method POST -Body $body -ContentType "application/json"
    Write-Host "Status: $($response.StatusCode)"
} catch {
    Write-Host "Failed: $($_.Exception.Message)"
}

Write-Host ""

# Test interactive chat endpoint
Write-Host "Testing /api/chat:"
try {
    $body = @{prompt = "test"; max_results = 5} | ConvertTo-Json
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/chat" -Method POST -Body $body -ContentType "application/json"
    Write-Host "Status: $($response.StatusCode)"
} catch {
    Write-Host "Failed: $($_.Exception.Message)"
}

Write-Host ""

# Test agents endpoint
Write-Host "Testing /api/agents/all:"
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/agents/all" -Method GET
    Write-Host "Status: $($response.StatusCode)"
} catch {
    Write-Host "Failed: $($_.Exception.Message)"
}
```

## JavaScript Browser Console Test

```javascript
// Test endpoints from browser console
async function testEndpoints() {
    const endpoints = [
        { method: 'GET', url: 'http://localhost:8000/api/health' },
        { method: 'POST', url: 'http://localhost:8000/chat', body: { prompt: 'test' } },
        { method: 'POST', url: 'http://localhost:8000/api/chat', body: { prompt: 'test', max_results: 5 } },
        { method: 'GET', url: 'http://localhost:8000/api/agents/all' }
    ];

    for (const endpoint of endpoints) {
        try {
            const options = {
                method: endpoint.method,
                headers: { 'Content-Type': 'application/json' }
            };
            
            if (endpoint.body) {
                options.body = JSON.stringify(endpoint.body);
            }
            
            const response = await fetch(endpoint.url, options);
            console.log(`${endpoint.method} ${endpoint.url}: Status ${response.status}`);
            
            if (response.ok) {
                const data = await response.json();
                console.log('Response:', data);
            }
        } catch (error) {
            console.log(`${endpoint.method} ${endpoint.url}: Failed - ${error.message}`);
        }
    }
}

// Run the test
testEndpoints();
```

## Current Expected Behavior

Based on the frontend implementation, here's what each mode expects:

### Classic Mode
- Uses existing working endpoints
- Should work with current backend

### Interactive Mode  
- Tries `/api/chat` first, falls back to `/chat`
- Gracefully degrades if interactive features unavailable
- Shows user-friendly error messages

### Real-time Mode
- Attempts WebSocket connection to `ws://localhost:8000/ws/chat`
- Falls back gracefully if WebSocket unavailable
- Shows connection status

## Error Handling

The frontend now includes:
- Graceful degradation for missing endpoints
- User-friendly error messages
- Fallback to working functionality
- Clear status indicators
