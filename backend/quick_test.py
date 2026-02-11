import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        # Check Spreadsheet Agent on 9000
        try:
            resp = await client.post('http://localhost:9000/execute', json={'prompt': 'test'})
            print(f'Spreadsheet (9000): {resp.status_code}')
        except Exception as e:
            print(f'Spreadsheet (9000) fail: {e}')

        # Check Browser Automation Agent on 8090
        try:
            resp = await client.post('http://localhost:8090/execute', json={'prompt': 'test'})
            print(f'Browser (8090): {resp.status_code}')
        except Exception as e:
            print(f'Browser (8090) fail: {e}')

if __name__ == "__main__":
    asyncio.run(test())
