"""
Test script to verify tool registry integration with orchestrator.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_tool_registry():
    """Test that tools are registered and can be discovered."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Tool Registry Initialization")
    print("="*60 + "\n")
    
    try:
        from orchestrator.tool_registry import (
            initialize_tools,
            get_all_tool_capabilities,
            get_tool_for_capability,
            is_tool_capable,
            execute_tool
        )
        
        # Initialize tools
        print("📦 Initializing tools...")
        initialize_tools()
        
        # Get all capabilities
        capabilities = get_all_tool_capabilities()
        print(f"\n✅ Registered {len(capabilities)} tool capabilities:")
        for cap in capabilities[:10]:  # Show first 10
            print(f"  • {cap}")
        if len(capabilities) > 10:
            print(f"  ... and {len(capabilities) - 10} more")
        
        print("\n" + "="*60)
        print("🧪 TEST 2: Tool Capability Check")
        print("="*60 + "\n")
        
        # Test specific capabilities
        test_capabilities = [
            "get stock quote",
            "search news",
            "search wikipedia",
            "web search",
            "analyze image",
            "edit document"  # This should NOT be handled by tools (agent only)
        ]
        
        for cap in test_capabilities:
            is_capable = is_tool_capable(cap)
            tool = get_tool_for_capability(cap)
            status = "✅ TOOL" if is_capable else "❌ AGENT NEEDED"
            tool_name = tool.name if tool else "N/A"
            print(f"{status}: '{cap}' -> {tool_name}")
        
        print("\n" + "="*60)
        print("🧪 TEST 3: Tool Execution")
        print("="*60 + "\n")
        
        # Test stock quote
        print("📈 Testing stock quote tool...")
        result = await execute_tool("get stock quote", {"ticker": "AAPL"})
        if result.get('success'):
            price_data = result.get('result', {})
            print(f"✅ Success: AAPL price = ${price_data.get('price', 'N/A')}")
            print(f"   Tool used: {result.get('tool_name')}")
        else:
            print(f"❌ Failed: {result.get('error')}")
        
        # Test news search
        print("\n📰 Testing news search tool...")
        result = await execute_tool("search news", {"query": "Apple", "page_size": 3})
        if result.get('success'):
            articles = result.get('result', {}).get('articles', [])
            print(f"✅ Success: Found {len(articles)} articles")
            if articles:
                print(f"   First article: {articles[0].get('title', 'N/A')[:60]}...")
        else:
            print(f"❌ Failed: {result.get('error')}")
        
        # Test Wikipedia
        print("\n📚 Testing Wikipedia search tool...")
        result = await execute_tool("search wikipedia", {"query": "Python programming"})
        if result.get('success'):
            results = result.get('result', {}).get('results', [])
            print(f"✅ Success: Found {len(results)} Wikipedia pages")
            if results:
                print(f"   First result: {results[0]}")
        else:
            print(f"❌ Failed: {result.get('error')}")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tool_registry())
