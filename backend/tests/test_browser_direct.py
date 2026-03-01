"""
Standalone Browser Agent Test (Non-Headless)

Tests the conversational memory architecture directly without the orchestrator.
Runs the browser agent with a visible browser so you can watch it work.

Usage:
    python backend/tests/test_browser_direct.py
"""

import asyncio
import sys
import os
import logging
import time

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging to see conversation manager activity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-7s %(name)-40s %(message)s',
    datefmt='%H:%M:%S'
)

# Boost conversation manager logging to see multi-turn activity
logging.getLogger('backend.agents.browser_agent.conversation_manager').setLevel(logging.DEBUG)
logging.getLogger('backend.agents.browser_agent.llm').setLevel(logging.DEBUG)

# Quiet noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('playwright').setLevel(logging.WARNING)


async def run_direct_test():
    """Run browser agent directly in non-headless mode."""
    
    from backend.agents.browser_agent.agent import BrowserAgent
    
    task = (
        "Go to amazon.in and search for Samsung Galaxy S25 Ultra. "
        "Find the cheapest option from the top results. "
        "Go to its page and extract: RAM, storage, display size, and battery capacity."
    )
    
    print("\n" + "=" * 70)
    print("🧪 DIRECT BROWSER AGENT TEST (Non-Headless)")
    print("=" * 70)
    print(f"📝 Task: {task}")
    print("🖥️  Mode: Non-headless (visible browser)")
    print("🧠 Architecture: Conversational multi-turn")
    print("=" * 70 + "\n")
    
    # Create agent in non-headless mode
    agent = BrowserAgent(task=task, headless=False)
    
    start = time.time()
    
    try:
        result = await agent.run()
        duration = time.time() - start
        
        print("\n" + "=" * 70)
        print("📊 RESULTS")
        print("=" * 70)
        print(f"✅ Success: {result.success}")
        print(f"⏱️  Duration: {duration:.1f}s")
        print(f"📝 Summary: {result.task_summary}")
        
        if result.extracted_data:
            print(f"\n📦 Extracted Data ({len(result.extracted_data)} items):")
            for item in result.extracted_data:
                if isinstance(item, dict):
                    info = item.get('structured_info', item)
                    if isinstance(info, dict):
                        key = info.get('key', '?')
                        value = info.get('value', '?')
                        verified = "✅" if info.get('verified') else "⚠️"
                        print(f"  {verified} {key}: {value}")
                    else:
                        print(f"  • {info}")
                else:
                    print(f"  • {item}")
        
        if result.error:
            print(f"\n❌ Error: {result.error}")
        
        # Print conversation stats
        try:
            stats = agent.llm.conversation.get_stats()
            print("\n🧠 Conversation Stats:")
            print(f"  Turns: {stats['total_turns']}")
            print(f"  Tokens: {stats['total_tokens']} ({stats['budget_used_pct']:.0f}% budget)")
            print(f"  Has summary: {stats['has_summary']}")
            if stats['has_summary']:
                print(f"  Summary tokens: {stats['summary_tokens']}")
            print(f"  Data inventory: {stats['data_inventory_size']} items")
            
            # Print the conversation data inventory
            if agent.llm.conversation.data_inventory:
                print("\n📦 Conversation Data Inventory:")
                for k, v in agent.llm.conversation.data_inventory.items():
                    print(f"  • {k}: {v}")
        except Exception as e:
            print(f"\n⚠️ Could not get conversation stats: {e}")
        
        # Print agent metrics
        print("\n📈 Agent Metrics:")
        print(f"  Actions: {agent.metrics['actions']['total']} total, "
              f"{agent.metrics['actions']['successful']} success, "
              f"{agent.metrics['actions']['failed']} failed")
        print(f"  LLM calls: {agent.metrics['llm_calls']['total']}")
        print(f"  Steps: {result.steps_taken if hasattr(result, 'steps_taken') else '?'}")
        
        print("\n" + "=" * 70)
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        duration = time.time() - start
        print(f"⏱️  Ran for: {duration:.1f}s")
        
        # Still print conversation stats
        try:
            stats = agent.llm.conversation.get_stats()
            print("\n🧠 Conversation Stats at interruption:")
            print(f"  Turns: {stats['total_turns']}")
            print(f"  Tokens: {stats['total_tokens']} ({stats['budget_used_pct']:.0f}% budget)")
            print(f"  Has summary: {stats['has_summary']}")
        except:
            pass
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_direct_test())
