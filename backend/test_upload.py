import httpx
import asyncio

async def test_upload():
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open('storage/spreadsheets/sample_data.csv', 'rb') as f:
                files = {'file': ('sample_data.csv', f, 'text/csv')}
                response = await client.post('http://localhost:8041/upload', files=files)
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_upload())
