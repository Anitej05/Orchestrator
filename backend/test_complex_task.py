"""
Complex Task Test - E-commerce Product Hunt with Multi-Site Comparison

This test exercises ALL agent capabilities:
- Multi-site navigation
- Search and sorting
- Filtering by criteria
- Structured data extraction
- Image saving
- CAPTCHA/security handling with vision
- Cross-site price comparison
- Dynamic replanning
- All SOTA improvements
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.browser_automation_agent import BrowserAgent


async def run_complex_task():
    """Run the ultimate complex task test"""
    
    print("=" * 80)
    print("🚀 COMPLEX TASK TEST - E-COMMERCE PRODUCT HUNT")
    print("=" * 80)
    print()
    
    # The ultimate complex task
    task = """Go to Best Buy and search for 'gaming laptops'. 
Sort results by 'Price: Low to High'. 
Find the 3 cheapest laptops that have at least 4-star ratings. 
For each laptop, extract: model name, price, rating, RAM, storage, and GPU. 
Save the product image for each laptop. 
If you encounter any CAPTCHA or security check, solve it. 
Then go to Newegg and check if any of these same laptop models are available for a lower price."""
    
    print(f"📋 TASK:")
    print(f"   {task}")
    print()
    print("⏱️  Starting task execution...")
    print()
    
    start_time = datetime.now()
    
    try:
        # Initialize agent with visible browser and extended steps
        async with BrowserAgent(
            task=task,
            max_steps=25,  # Complex task needs more steps
            headless=False,  # Visible for monitoring
            enable_streaming=True
        ) as agent:
            
            print("✅ Browser agent initialized")
            print(f"   Task ID: {agent.task_id}")
            print(f"   Max steps: {agent.max_steps}")
            print()
            
            # Run the task
            result = await agent.run()
            
            # Calculate execution time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print()
            print("=" * 80)
            print("📊 TASK EXECUTION COMPLETE")
            print("=" * 80)
            print()
            
            # Display results
            print(f"✅ Success: {result.get('success', False)}")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            print(f"📝 Steps taken: {len(result.get('actions_taken', []))}")
            print()
            
            # Display task summary
            print("📋 TASK SUMMARY:")
            print(f"   {result.get('task_summary', 'No summary available')}")
            print()
            
            # Display actions taken
            actions = result.get('actions_taken', [])
            if actions:
                print(f"🎬 ACTIONS EXECUTED ({len(actions)}):")
                for i, action in enumerate(actions, 1):
                    action_type = action.get('action', 'unknown')
                    status = "✅" if action.get('success') else "❌"
                    reasoning = action.get('reasoning', '')[:60]
                    print(f"   {i}. {status} {action_type}: {reasoning}...")
                print()
            
            # Display extracted data
            extracted = result.get('extracted_data', {})
            if extracted:
                print("📦 EXTRACTED DATA:")
                
                # Check for structured items (products)
                items = extracted.get('structured_items', [])
                if items:
                    print(f"   Found {len(items)} products:")
                    for i, item in enumerate(items, 1):
                        print(f"\n   Product {i}:")
                        for key, value in item.items():
                            if key != 'position' and value:
                                print(f"      {key}: {value}")
                
                # Check for vision analysis
                vision = extracted.get('vision_analysis')
                if vision:
                    print(f"\n   Vision Analysis:")
                    print(f"      {vision[:200]}...")
                
                # Check extraction type
                extraction_type = extracted.get('extraction_type')
                if extraction_type:
                    print(f"\n   Extraction Method: {extraction_type}")
                
                print()
            
            # Display screenshots
            screenshots = result.get('screenshot_files', [])
            if screenshots:
                print(f"📸 SCREENSHOTS CAPTURED ({len(screenshots)}):")
                for screenshot in screenshots:
                    print(f"   - {screenshot}")
                print()
            
            # Display downloads (saved images)
            if hasattr(agent, 'downloads') and agent.downloads:
                print(f"📥 IMAGES SAVED ({len(agent.downloads)}):")
                for download in agent.downloads:
                    print(f"   - {download}")
                print()
            
            # Display metrics
            if hasattr(agent, 'metrics'):
                metrics = agent.metrics
                print("📈 PERFORMANCE METRICS:")
                print(f"   Total time: {metrics.get('total_time', 0):.2f}s")
                print(f"   LLM calls: {metrics.get('llm_calls', 0)}")
                print(f"   Page loads: {metrics.get('page_loads', 0)}")
                print(f"   Screenshots: {metrics.get('screenshots_taken', 0)}")
                print()
            
            # Display SOTA improvements usage
            print("🚀 SOTA IMPROVEMENTS UTILIZED:")
            
            # Check if context optimization was used
            if hasattr(agent, 'context_optimizer'):
                print("   ✅ ContextOptimizer - Token reduction")
            
            # Check if selector strategy was used
            if hasattr(agent, 'selector_strategy'):
                print("   ✅ SelectorStrategy - Multi-strategy element selection")
            
            # Check if page stabilizer was used
            if hasattr(agent, 'page_stabilizer'):
                print("   ✅ PageStabilizer - Overlay handling & stability")
            
            # Check if dynamic planner was used
            if hasattr(agent, 'dynamic_planner'):
                print("   ✅ DynamicPlanner - Adaptive replanning")
            
            # Check if vision optimizer was used
            if hasattr(agent, 'vision_optimizer'):
                print("   ✅ VisionOptimizer - Smart vision usage")
            
            print()
            
            # Check for failures
            if hasattr(agent, 'actions_failed') and agent.actions_failed:
                print(f"⚠️  FAILED ACTIONS ({len(agent.actions_failed)}):")
                for i, failed in enumerate(agent.actions_failed[-5:], 1):  # Last 5
                    action_type = failed.get('action', 'unknown')
                    error = failed.get('error', 'Unknown error')[:60]
                    print(f"   {i}. {action_type}: {error}...")
                print()
            
            # Check if task was completed
            if result.get('success'):
                print("🎉 TASK COMPLETED SUCCESSFULLY!")
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ TASK FAILED: {error}")
            
            print()
            print("=" * 80)
            
            # Save detailed results to file
            results_file = Path("backend/logs") / f"complex_task_results_{agent.task_id}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump({
                    'task': task,
                    'task_id': agent.task_id,
                    'success': result.get('success', False),
                    'duration_seconds': duration,
                    'actions_taken': actions,
                    'extracted_data': extracted,
                    'screenshots': screenshots,
                    'downloads': agent.downloads if hasattr(agent, 'downloads') else [],
                    'metrics': agent.metrics if hasattr(agent, 'metrics') else {},
                    'task_plan': agent.task_plan if hasattr(agent, 'task_plan') else [],
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"💾 Detailed results saved to: {results_file}")
            print()
            
            return result
            
    except KeyboardInterrupt:
        print("\n⚠️  Task interrupted by user")
        return {'success': False, 'error': 'Interrupted by user'}
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


async def run_alternative_tasks():
    """Run alternative complex tasks for comparison"""
    
    alternative_tasks = [
        {
            'name': 'Multi-Site Product Comparison',
            'task': """Compare wireless headphones under $100 on Amazon and eBay. 
Find the top 3 products from each site. Extract: title, price, rating, and number of reviews. 
Save product images and analyze them to describe their appearance.""",
            'max_steps': 20
        },
        {
            'name': 'Real Estate Visual Assessment',
            'task': """Find 3 houses for sale in Seattle under $800k on Zillow. 
Extract: address, price, bedrooms, bathrooms, square footage. 
Save the main property photo for each and analyze it to describe the house's exterior style.""",
            'max_steps': 20
        },
        {
            'name': 'Job Market Intelligence',
            'task': """Search for 'Python Developer' jobs on LinkedIn and Indeed. 
Get the top 3 from each site including: job title, company, location, and salary range. 
Identify which skills appear most frequently.""",
            'max_steps': 20
        }
    ]
    
    print("\n" + "=" * 80)
    print("🎯 ALTERNATIVE COMPLEX TASKS AVAILABLE")
    print("=" * 80)
    print()
    
    for i, task_info in enumerate(alternative_tasks, 1):
        print(f"{i}. {task_info['name']}")
        print(f"   Task: {task_info['task'][:100]}...")
        print()
    
    print("To run an alternative task, modify the script and uncomment the desired task.")
    print()


if __name__ == "__main__":
    print()
    print("🤖 Browser Agent - Complex Task Test")
    print()
    
    # Check environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = ['CEREBRAS_API_KEY', 'GROQ_API_KEY', 'NVIDIA_API_KEY']
    available_keys = [key for key in required_keys if os.getenv(key)]
    
    if not available_keys:
        print("❌ ERROR: No API keys found!")
        print("   Please set at least one of: CEREBRAS_API_KEY, GROQ_API_KEY, NVIDIA_API_KEY")
        sys.exit(1)
    
    print(f"✅ API Keys configured: {', '.join(available_keys)}")
    
    # Check for vision
    if os.getenv('OLLAMA_API_KEY'):
        print("✅ Vision enabled (Ollama)")
    else:
        print("⚠️  Vision disabled (set OLLAMA_API_KEY for vision capabilities)")
    
    print()
    
    # Run the main complex task
    result = asyncio.run(run_complex_task())
    
    # Show alternative tasks
    asyncio.run(run_alternative_tasks())
    
    # Exit with appropriate code
    sys.exit(0 if result.get('success') else 1)
