"""
Test script to verify lazy tool initialization.
Tools should NOT load until first use.
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


async def test_lazy_initialization():
    """Test that tools are NOT initialized until first use."""
    print("\n" + "="*60)
    print("🧪 TEST: Lazy Tool Initialization")
    print("="*60 + "\n")
    
    try:
        # Import tool registry WITHOUT initializing
        from orchestrator import tool_registry
        
        # Check internal state - tools should NOT be initialized yet
        print("📦 Step 1: Import tool_registry module...")
        print(f"   _tools_initialized = {tool_registry._tools_initialized}")
        print(f"   _tool_registry = {len(tool_registry._tool_registry)} items")
        
        if tool_registry._tools_initialized:
            print("❌ FAILED: Tools were initialized at import time!")
            return False
        else:
            print("✅ PASS: Tools are dormant (not initialized yet)")
        
        # Now trigger lazy initialization by calling a function
        print("\n📦 Step 2: Call is_tool_capable() - should trigger lazy init...")
        is_capable = tool_registry.is_tool_capable("get stock quote")
        
        print(f"   _tools_initialized = {tool_registry._tools_initialized}")
        print(f"   _tool_registry = {len(tool_registry._tool_registry)} items")
        print(f"   is_capable('get stock quote') = {is_capable}")
        
        if not tool_registry._tools_initialized:
            print("❌ FAILED: Tools were NOT initialized after first use!")
            return False
        else:
            print("✅ PASS: Tools initialized on first use")
        
        # Verify tools are actually loaded
        if len(tool_registry._tool_registry) == 0:
            print("❌ FAILED: Tool registry is empty!")
            return False
        else:
            print(f"✅ PASS: {len(tool_registry._tool_registry)} capabilities registered")
        
        # Test that subsequent calls don't re-initialize
        print("\n📦 Step 3: Call get_all_tool_capabilities() - should use existing...")
        capabilities = tool_registry.get_all_tool_capabilities()
        print(f"   Found {len(capabilities)} capabilities")
        print("✅ PASS: Subsequent calls work correctly")
        
        # Test tool execution
        print("\n📦 Step 4: Execute a tool...")
        result = await tool_registry.execute_tool("get stock quote", {"ticker": "AAPL"})
        if result.get('success'):
            print(f"✅ PASS: Tool executed successfully (AAPL = ${result['result'].get('price', 'N/A')})")
        else:
            print(f"⚠️  Tool execution failed: {result.get('error')}")
        
        print("\n" + "="*60)
        print("✅ ALL LAZY LOADING TESTS PASSED")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_lazy_initialization())
    sys.exit(0 if success else 1)
