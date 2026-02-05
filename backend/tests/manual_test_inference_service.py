
import asyncio
import os
import sys
import io
import time
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from backend.services.inference_service import inference_service, ProviderType, InferencePriority

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class TestSchema(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    slogan: str = Field(description="A short catchy slogan for them")

async def test_provider_connectivity():
    print("\n=== 1. Testing Provider Connectivity ===")
    providers = [ProviderType.CEREBRAS, ProviderType.GROQ, ProviderType.NVIDIA]
    
    for p in providers:
        print(f"\nScanning {p.value}...")
        try:
            start = time.time()
            response = await inference_service.generate(
                messages=[HumanMessage(content="Say 'Hello' and nothing else.")],
                provider=p,
                max_tokens=10,
                use_cache=False
            )
            duration = time.time() - start
            print(f"✅ {p.value}: Success ({duration:.2f}s) -> '{response}'")
        except Exception as e:
            print(f"❌ {p.value}: Failed ({e})")

async def test_caching():
    print("\n=== 2. Testing Caching ===")
    prompt = [HumanMessage(content="Write a unique 10-word sentence about lemons.")]
    
    print("Request 1 (Cold Cache)...")
    start1 = time.time()
    res1 = await inference_service.generate(messages=prompt, provider=ProviderType.CEREBRAS)
    dur1 = time.time() - start1
    print(f"Result 1: {res1[:30]}... ({dur1:.2f}s)")
    
    print("Request 2 (Warm Cache)...")
    start2 = time.time()
    res2 = await inference_service.generate(messages=prompt, provider=ProviderType.CEREBRAS)
    dur2 = time.time() - start2
    print(f"Result 2: {res2[:30]}... ({dur2:.2f}s)")
    
    if dur2 < dur1 and res1 == res2:
        print("✅ Caching is WORKING (Time saved)")
    else:
        print("⚠️ Caching metric unclear (Network variance or cache miss)")

async def test_structured_output():
    print("\n=== 3. Testing Structured Output ===")
    prompt = [HumanMessage(content="Generate a fictional profile for a space pirate.")]
    
    try:
        result = await inference_service.generate_structured(
            messages=prompt,
            schema=TestSchema,
            provider=ProviderType.CEREBRAS
        )
        print(f"✅ Structured Output Success:\n{result.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"❌ Structured Output Failed: {e}")

async def main():
    print("Starting Comprehensive Inference Service Test...")
    
    # Check Environment Variables first
    keys = ["CEREBRAS_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY"]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        print(f"⚠️ WARNING: Missing API Keys: {missing}. Some tests will fail.")
    
    await test_provider_connectivity()
    await test_caching()
    await test_structured_output()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
