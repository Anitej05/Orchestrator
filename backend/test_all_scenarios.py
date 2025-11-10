"""
Test All Complex Scenarios - Run multiple complex tasks in sequence

This script runs several complex tasks to thoroughly test all agent capabilities.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from agents.browser_automation_agent import BrowserAgent


# Define test scenarios
SCENARIOS = [
    {
        'id': 'scenario_1_simple_search',
        'name': 'Simple Product Search',
        'task': 'Go to Best Buy and search for "laptops". Extract the first 3 product names and prices.',
        'max_steps': 10,
        'difficulty': 'Easy'
    },
    {
        'id': 'scenario_2_filtering',
        'name': 'Search with Filtering',
        'task': 'Go to Best Buy, search for "laptops", and find 2 laptops with at least 4-star ratings. Extract model and price.',
        'max_steps': 15,
        'difficulty': 'Medium'
    },
    {
        'id': 'scenario_3_sorting',
        'name': 'Search with Sorting',
        'task': 'Go to Best Buy, search for "gaming laptops", sort by "Price: Low to High", and extract the 3 cheapest laptops.',
        'max_steps': 15,
        'difficulty': 'Medium'
    },
    {
        'id': 'scenario_4_multi_site',
        'name': 'Multi-Site Comparison',
        'task': 'Search for "wireless mouse" on Best Buy and get the top 2 products. Then go to Newegg and search for the same.',
        'max_steps': 20,
        'difficulty': 'Hard'
    },
    {
        'id': 'scenario_5_full_complex',
        'name': 'Full Complex Task',
        'task': '''Go to Best Buy and search for 'gaming laptops'. 
Sort results by 'Price: Low to High'. 
Find the 3 cheapest laptops that have at least 4-star ratings. 
For each laptop, extract: model name, price, rating, RAM, storage, and GPU. 
If you encounter any CAPTCHA or security check, solve it.''',
        'max_steps': 25,
        'difficulty': 'Very Hard'
    }
]


async def run_scenario(scenario: dict, scenario_num: int, total_scenarios: int):
    """Run a single test scenario"""
    
    print()
    print("=" * 80)
    print(f"SCENARIO {scenario_num}/{total_scenarios}: {scenario['name']}")
    print(f"Difficulty: {scenario['difficulty']}")
    print("=" * 80)
    print()
    print(f"📋 Task: {scenario['task']}")
    print()
    
    start_time = datetime.now()
    
    try:
        async with BrowserAgent(
            task=scenario['task'],
            max_steps=scenario['max_steps'],
            headless=False,
            enable_streaming=True
        ) as agent:
            
            print(f"✅ Agent initialized (ID: {agent.task_id})")
            
            # Run the task
            result = await agent.run()
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Display results
            print()
            print("-" * 80)
            print(f"✅ Success: {result.get('success', False)}")
            print(f"⏱️  Duration: {duration:.2f}s")
            print(f"📝 Steps: {len(result.get('actions_taken', []))}")
            print(f"📋 Summary: {result.get('task_summary', 'N/A')[:100]}...")
            
            # Show extracted data count
            extracted = result.get('extracted_data', {})
            if extracted:
                items = extracted.get('structured_items', [])
                if items:
                    print(f"📦 Extracted: {len(items)} items")
            
            print("-" * 80)
            
            # Save scenario results
            results_file = Path("backend/logs") / f"scenario_{scenario['id']}_{agent.task_id}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump({
                    'scenario': scenario,
                    'task_id': agent.task_id,
                    'success': result.get('success', False),
                    'duration_seconds': duration,
                    'steps_taken': len(result.get('actions_taken', [])),
                    'actions': result.get('actions_taken', []),
                    'extracted_data': extracted,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            return {
                'scenario': scenario['name'],
                'success': result.get('success', False),
                'duration': duration,
                'steps': len(result.get('actions_taken', [])),
                'items_extracted': len(extracted.get('structured_items', [])) if extracted else 0
            }
            
    except Exception as e:
        print(f"\n❌ Scenario failed: {e}")
        return {
            'scenario': scenario['name'],
            'success': False,
            'duration': (datetime.now() - start_time).total_seconds(),
            'error': str(e)
        }


async def run_all_scenarios(selected_scenarios=None):
    """Run all test scenarios"""
    
    print()
    print("=" * 80)
    print("🚀 COMPLEX TASK TESTING - ALL SCENARIOS")
    print("=" * 80)
    print()
    
    # Use selected scenarios or all
    scenarios_to_run = selected_scenarios or SCENARIOS
    
    print(f"📋 Running {len(scenarios_to_run)} scenarios:")
    for i, scenario in enumerate(scenarios_to_run, 1):
        print(f"   {i}. {scenario['name']} ({scenario['difficulty']})")
    print()
    
    input("Press Enter to start testing...")
    print()
    
    # Run each scenario
    results = []
    for i, scenario in enumerate(scenarios_to_run, 1):
        result = await run_scenario(scenario, i, len(scenarios_to_run))
        results.append(result)
        
        # Pause between scenarios
        if i < len(scenarios_to_run):
            print()
            print("⏸️  Pausing 5 seconds before next scenario...")
            await asyncio.sleep(5)
    
    # Display summary
    print()
    print("=" * 80)
    print("📊 TESTING SUMMARY")
    print("=" * 80)
    print()
    
    successful = sum(1 for r in results if r.get('success'))
    total = len(results)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    print(f"✅ Success Rate: {successful}/{total} ({success_rate:.1f}%)")
    print()
    
    print("Results by Scenario:")
    for i, result in enumerate(results, 1):
        status = "✅" if result.get('success') else "❌"
        scenario_name = result['scenario']
        duration = result.get('duration', 0)
        steps = result.get('steps', 0)
        items = result.get('items_extracted', 0)
        
        print(f"{i}. {status} {scenario_name}")
        print(f"   Duration: {duration:.1f}s | Steps: {steps} | Items: {items}")
        
        if not result.get('success') and 'error' in result:
            print(f"   Error: {result['error'][:60]}...")
    
    print()
    print("=" * 80)
    
    # Save summary
    summary_file = Path("backend/logs") / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_scenarios': total,
            'successful': successful,
            'success_rate': success_rate,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"💾 Summary saved to: {summary_file}")
    print()
    
    return results


def select_scenarios():
    """Interactive scenario selection"""
    
    print()
    print("=" * 80)
    print("SCENARIO SELECTION")
    print("=" * 80)
    print()
    print("Available scenarios:")
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"{i}. {scenario['name']} ({scenario['difficulty']})")
        print(f"   {scenario['task'][:70]}...")
        print()
    
    print("Options:")
    print("  - Enter scenario numbers (e.g., '1,2,3' or '1-3')")
    print("  - Enter 'all' to run all scenarios")
    print("  - Enter 'easy' for easy scenarios only")
    print("  - Enter 'hard' for hard scenarios only")
    print()
    
    choice = input("Select scenarios: ").strip().lower()
    
    if choice == 'all':
        return SCENARIOS
    elif choice == 'easy':
        return [s for s in SCENARIOS if s['difficulty'] in ['Easy', 'Medium']]
    elif choice == 'hard':
        return [s for s in SCENARIOS if s['difficulty'] in ['Hard', 'Very Hard']]
    else:
        # Parse numbers
        try:
            if '-' in choice:
                start, end = map(int, choice.split('-'))
                indices = range(start - 1, end)
            else:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
            
            return [SCENARIOS[i] for i in indices if 0 <= i < len(SCENARIOS)]
        except:
            print("Invalid selection, running all scenarios")
            return SCENARIOS


if __name__ == "__main__":
    print()
    print("🤖 Browser Agent - Complex Task Testing Suite")
    print()
    
    # Check environment
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = ['CEREBRAS_API_KEY', 'GROQ_API_KEY', 'NVIDIA_API_KEY']
    available_keys = [key for key in required_keys if os.getenv(key)]
    
    if not available_keys:
        print("❌ ERROR: No API keys found!")
        sys.exit(1)
    
    print(f"✅ API Keys: {', '.join(available_keys)}")
    
    if os.getenv('OLLAMA_API_KEY'):
        print("✅ Vision: Enabled")
    else:
        print("⚠️  Vision: Disabled")
    
    # Select scenarios
    selected = select_scenarios()
    
    # Run scenarios
    results = asyncio.run(run_all_scenarios(selected))
    
    # Exit with appropriate code
    successful = sum(1 for r in results if r.get('success'))
    sys.exit(0 if successful == len(results) else 1)
