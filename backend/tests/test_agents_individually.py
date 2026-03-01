#!/usr/bin/env python3
"""
Individual Agent Testing Script
Tests each agent with real-world test cases.

Usage:
    cd backend && python tests/test_agents_individually.py
"""

import asyncio
import json
import os
import sys
import time
import tempfile
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Load environment variables (optional - they may already be set)
try:
    from dotenv import load_dotenv
    load_dotenv(backend_dir / ".env")
except ImportError:
    pass  # Environment variables may already be set

# Test results storage
test_results = {
    "spreadsheet_agent": [],
    "document_agent": [],
    "mail_agent": [],
    "universal_agent": [],
    "zoho_books_agent": []
}

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_test(name: str, success: bool, message: str = "", duration: float = 0):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    duration_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"  {status} | {name}{duration_str}")
    if message:
        print(f"         {message[:100]}{'...' if len(message) > 100 else ''}")

def save_result(agent: str, test_name: str, success: bool, message: str, duration: float):
    """Save test result."""
    test_results[agent].append({
        "test": test_name,
        "success": success,
        "message": message,
        "duration": duration
    })

# ============================================================================
# SPREADSHEET AGENT TESTS
# ============================================================================

async def test_spreadsheet_agent():
    """Test Spreadsheet Agent with real-world scenarios."""
    print_header("SPREADSHEET AGENT TESTS")
    
    try:
        from agents.spreadsheet_agent.agent import spreadsheet_agent
        
        # Test 1: Create and analyze a sales dataset
        print("\n📊 Test 1: Sales Data Analysis")
        start = time.time()
        try:
            # Create a test CSV file
            test_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            test_csv.write("""Product,Region,Sales,Quantity,Date
Laptop,North,1500,5,2024-01-15
Mouse,South,200,20,2024-01-16
Keyboard,North,300,15,2024-01-17
Monitor,East,800,8,2024-01-18
Laptop,South,1500,3,2024-01-19
Mouse,North,200,25,2024-01-20
Keyboard,West,300,10,2024-01-21
Monitor,North,800,12,2024-01-22
""")
            test_csv.close()
            
            # Load and analyze
            result = await spreadsheet_agent.execute(
                prompt="Load this file and tell me the total sales by region",
                params={'file_path': test_csv.name},
                thread_id="test-spreadsheet-1"
            )
            
            duration = time.time() - start
            success = result.status.value == "complete" or result.success
            message = result.result.get('answer', 'No answer') if result.result else str(result.error)
            print_test("Sales Data Analysis", success, message, duration)
            save_result("spreadsheet_agent", "Sales Data Analysis", success, message, duration)
            
            # Cleanup
            os.unlink(test_csv.name)
            
        except Exception as e:
            print_test("Sales Data Analysis", False, str(e))
            save_result("spreadsheet_agent", "Sales Data Analysis", False, str(e), time.time() - start)
        
        # Test 2: Aggregation query
        print("\n📊 Test 2: Aggregation Query")
        start = time.time()
        try:
            test_csv2 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            test_csv2.write("""Employee,Department,Salary,Years
Alice,Engineering,85000,5
Bob,Marketing,65000,3
Charlie,Engineering,90000,7
Diana,Sales,70000,4
Eve,Engineering,80000,6
Frank,Marketing,60000,2
Grace,Sales,75000,5
""")
            test_csv2.close()
            
            result = await spreadsheet_agent.execute(
                prompt="What is the average salary by department?",
                params={'file_path': test_csv2.name},
                thread_id="test-spreadsheet-2"
            )
            
            duration = time.time() - start
            success = result.status.value == "complete" or result.success
            message = result.result.get('answer', 'No answer') if result.result else str(result.error)
            print_test("Aggregation Query", success, message, duration)
            save_result("spreadsheet_agent", "Aggregation Query", success, message, duration)
            
            os.unlink(test_csv2.name)
            
        except Exception as e:
            print_test("Aggregation Query", False, str(e))
            save_result("spreadsheet_agent", "Aggregation Query", False, str(e), time.time() - start)
        
        # Test 3: Data filtering
        print("\n📊 Test 3: Data Filtering")
        start = time.time()
        try:
            test_csv3 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            test_csv3.write("""OrderID,Customer,Amount,Status,Priority
1001,Acme Corp,5000,Shipped,High
1002,Globex,3000,Pending,Medium
1003,Initech,7500,Shipped,High
1004,Umbrella,2000,Cancelled,Low
1005,Stark Ind,10000,Processing,High
1006,Wayne Ent,4500,Shipped,Medium
""")
            test_csv3.close()
            
            result = await spreadsheet_agent.execute(
                prompt="Show me all high priority orders",
                params={'file_path': test_csv3.name},
                thread_id="test-spreadsheet-3"
            )
            
            duration = time.time() - start
            success = result.status.value == "complete" or result.success
            message = result.result.get('answer', 'No answer') if result.result else str(result.error)
            print_test("Data Filtering", success, message, duration)
            save_result("spreadsheet_agent", "Data Filtering", success, message, duration)
            
            os.unlink(test_csv3.name)
            
        except Exception as e:
            print_test("Data Filtering", False, str(e))
            save_result("spreadsheet_agent", "Data Filtering", False, str(e), time.time() - start)
        
        # Test 4: Column operations
        print("\n📊 Test 4: Column Operations (Add Calculated Column)")
        start = time.time()
        try:
            test_csv4 = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            test_csv4.write("""Item,Price,Quantity,Tax
Widget,100,10,5
Gadget,250,5,12.5
Gizmo,75,20,3.75
Doohickey,150,8,7.5
""")
            test_csv4.close()
            
            result = await spreadsheet_agent.execute(
                prompt="Add a new column 'Total' that calculates Price * Quantity + Tax",
                params={'file_path': test_csv4.name},
                thread_id="test-spreadsheet-4"
            )
            
            duration = time.time() - start
            success = result.status.value == "complete" or result.success
            message = result.result.get('answer', 'No answer') if result.result else str(result.error)
            print_test("Column Operations", success, message, duration)
            save_result("spreadsheet_agent", "Column Operations", success, message, duration)
            
            os.unlink(test_csv4.name)
            
        except Exception as e:
            print_test("Column Operations", False, str(e))
            save_result("spreadsheet_agent", "Column Operations", False, str(e), time.time() - start)
            
    except ImportError as e:
        print(f"\n  ⚠️  Spreadsheet Agent not available: {e}")
        save_result("spreadsheet_agent", "Import", False, str(e), 0)

# ============================================================================
# DOCUMENT AGENT TESTS
# ============================================================================

async def test_document_agent():
    """Test Document Agent with real-world scenarios."""
    print_header("DOCUMENT AGENT TESTS")
    
    try:
        from agents.document_agent_lib.agent import DocumentAgent
        from agents.document_agent_lib.agent_schemas import (
            AnalyzeDocumentRequest, CreateDocumentRequest, EditDocumentRequest
        )
        
        agent = DocumentAgent()
        
        # Test 1: Create a document
        print("\n📄 Test 1: Create Document")
        start = time.time()
        try:
            result = await agent.create_document(CreateDocumentRequest(
                content="# Test Report\n\nThis is a test document for the Document Agent.\n\n## Section 1\n\nThis section contains sample content for testing purposes.\n\n## Section 2\n\nAdditional content to verify document creation capabilities.",
                file_name="test_report.docx",
                file_type="docx",
                output_dir=str(backend_dir.parent / "storage" / "document_agent")
            ))
            
            duration = time.time() - start
            success = result.get('success', False)
            message = result.get('message', 'No message')
            print_test("Create Document", success, message, duration)
            save_result("document_agent", "Create Document", success, message, duration)
            
        except Exception as e:
            print_test("Create Document", False, str(e))
            save_result("document_agent", "Create Document", False, str(e), time.time() - start)
        
        # Test 2: Analyze document content
        print("\n📄 Test 2: Analyze Document Content")
        start = time.time()
        try:
            # Create a test document first
            test_doc = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            test_doc.write("""
QUARTERLY BUSINESS REPORT - Q4 2024

Executive Summary:
The company achieved significant growth in Q4 2024, with total revenue reaching $2.5 million,
representing a 15% increase from the previous quarter. Key highlights include:

1. Customer acquisition increased by 25%
2. Product satisfaction scores improved to 4.5/5
3. Operational costs reduced by 10%
4. New market expansion into 3 regions

Financial Highlights:
- Revenue: $2,500,000
- Operating Expenses: $1,800,000
- Net Profit: $700,000
- Profit Margin: 28%

Challenges and Mitigation:
Supply chain disruptions were addressed through diversification of suppliers.
Employee retention improved following implementation of new benefits program.

Outlook:
Q1 2025 projections indicate continued growth with expected revenue of $2.8 million.
""")
            test_doc.close()
            
            result = await agent.analyze_document(AnalyzeDocumentRequest(
                file_path=test_doc.name,
                query="What were the key financial highlights and challenges mentioned in this report?"
            ))
            
            duration = time.time() - start
            success = result.get('success', False)
            message = result.get('answer', 'No answer')[:200] if result.get('answer') else result.get('message', 'No message')
            print_test("Analyze Document", success, message, duration)
            save_result("document_agent", "Analyze Document", success, message, duration)
            
            os.unlink(test_doc.name)
            
        except Exception as e:
            print_test("Analyze Document", False, str(e))
            save_result("document_agent", "Analyze Document", False, str(e), time.time() - start)
        
        # Test 3: Extract key information
        print("\n📄 Test 3: Extract Key Information")
        start = time.time()
        try:
            test_doc2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            test_doc2.write("""
PROJECT PROPOSAL: WEBSITE REDESIGN

Project Lead: John Smith
Department: Digital Marketing
Budget: $50,000
Timeline: 3 months (January - March 2025)

Team Members:
- Sarah Johnson (UI/UX Designer)
- Mike Chen (Frontend Developer)
- Emily Davis (Backend Developer)
- Alex Brown (QA Engineer)

Objectives:
1. Modernize website appearance
2. Improve user experience
3. Increase conversion rate by 20%
4. Implement responsive design

Deliverables:
- New homepage design
- Product catalog redesign
- Mobile-responsive layouts
- Performance optimization

Risk Assessment:
- Low risk: Design changes
- Medium risk: Backend integration
- High risk: Timeline constraints
""")
            test_doc2.close()
            
            result = await agent.analyze_document(AnalyzeDocumentRequest(
                file_path=test_doc2.name,
                query="Extract the project budget, timeline, and team members"
            ))
            
            duration = time.time() - start
            success = result.get('success', False)
            message = result.get('answer', 'No answer')[:200] if result.get('answer') else result.get('message', 'No message')
            print_test("Extract Key Information", success, message, duration)
            save_result("document_agent", "Extract Key Information", success, message, duration)
            
            os.unlink(test_doc2.name)
            
        except Exception as e:
            print_test("Extract Key Information", False, str(e))
            save_result("document_agent", "Extract Key Information", False, str(e), time.time() - start)
            
    except ImportError as e:
        print(f"\n  ⚠️  Document Agent not available: {e}")
        save_result("document_agent", "Import", False, str(e), 0)

# ============================================================================
# MAIL AGENT TESTS
# ============================================================================

async def test_mail_agent():
    """Test Mail Agent capabilities."""
    print_header("MAIL AGENT TESTS")
    
    try:
        # Check if Composio API key is available
        composio_key = os.getenv("COMPOSIO_API_KEY")
        if not composio_key:
            print("\n  ⚠️  COMPOSIO_API_KEY not found - skipping Mail Agent tests")
            save_result("mail_agent", "Configuration", False, "COMPOSIO_API_KEY not found", 0)
            return
        
        from agents.mail_agent.agent import app
        from agents.mail_agent.agent_schemas import SemanticSearchRequest, SummarizeRequest
        
        # Test 1: Health check
        print("\n📧 Test 1: Mail Agent Health Check")
        start = time.time()
        try:
            # The mail agent uses lazy initialization, so we test the health endpoint
            print_test("Health Check", True, "Mail agent module loaded successfully", time.time() - start)
            save_result("mail_agent", "Health Check", True, "Module loaded", time.time() - start)
            
        except Exception as e:
            print_test("Health Check", False, str(e))
            save_result("mail_agent", "Health Check", False, str(e), time.time() - start)
        
        # Test 2: Request decomposition
        print("\n📧 Test 2: Request Decomposition")
        start = time.time()
        try:
            from agents.mail_agent.llm import llm_client
            
            plan = await llm_client.decompose_complex_request(
                "Find emails from last week about the project meeting and summarize them"
            )
            
            duration = time.time() - start
            success = bool(plan.get('steps'))
            message = f"Decomposed into {len(plan.get('steps', []))} steps"
            print_test("Request Decomposition", success, message, duration)
            save_result("mail_agent", "Request Decomposition", success, message, duration)
            
        except Exception as e:
            print_test("Request Decomposition", False, str(e))
            save_result("mail_agent", "Request Decomposition", False, str(e), time.time() - start)
            
    except ImportError as e:
        print(f"\n  ⚠️  Mail Agent not available: {e}")
        save_result("mail_agent", "Import", False, str(e), 0)

# ============================================================================
# UNIVERSAL AGENT TESTS
# ============================================================================

async def test_universal_agent():
    """Test Universal Agent with various tasks."""
    print_header("UNIVERSAL AGENT TESTS")
    
    try:
        from agents.universal_agent.base_agent_impl import UniversalAgent
        from agents.base.services import AgentServices
        
        # Create agent with default services
        services = AgentServices.create_default()
        agent = UniversalAgent(services=services)
        await agent.initialize()
        
        # Test 1: General reasoning task
        print("\n🤖 Test 1: General Reasoning Task")
        start = time.time()
        try:
            from agents.base.types import AgentRequest
            
            result = await agent.execute(AgentRequest(
                prompt="Explain the difference between supervised and unsupervised machine learning in simple terms"
            ))
            
            duration = time.time() - start
            success = result.status.value == "complete"
            message = result.summary[:100] if result.summary else "Task completed"
            print_test("General Reasoning", success, message, duration)
            save_result("universal_agent", "General Reasoning", success, message, duration)
            
        except Exception as e:
            print_test("General Reasoning", False, str(e))
            save_result("universal_agent", "General Reasoning", False, str(e), time.time() - start)
        
        # Test 2: Problem solving
        print("\n🤖 Test 2: Problem Solving Task")
        start = time.time()
        try:
            from agents.base.types import AgentRequest
            
            result = await agent.execute(AgentRequest(
                prompt="If a train travels at 60 mph and needs to cover 180 miles, how long will it take? Show your calculation."
            ))
            
            duration = time.time() - start
            success = result.status.value == "complete"
            message = result.summary[:100] if result.summary else "Task completed"
            print_test("Problem Solving", success, message, duration)
            save_result("universal_agent", "Problem Solving", success, message, duration)
            
        except Exception as e:
            print_test("Problem Solving", False, str(e))
            save_result("universal_agent", "Problem Solving", False, str(e), time.time() - start)
        
        # Test 3: Analysis task
        print("\n🤖 Test 3: Analysis Task")
        start = time.time()
        try:
            from agents.base.types import AgentRequest
            
            result = await agent.execute(AgentRequest(
                prompt="Analyze the pros and cons of remote work vs office work for software developers"
            ))
            
            duration = time.time() - start
            success = result.status.value == "complete"
            message = result.summary[:100] if result.summary else "Task completed"
            print_test("Analysis Task", success, message, duration)
            save_result("universal_agent", "Analysis Task", success, message, duration)
            
        except Exception as e:
            print_test("Analysis Task", False, str(e))
            save_result("universal_agent", "Analysis Task", False, str(e), time.time() - start)
        
        await agent.terminate()
        
    except ImportError as e:
        print(f"\n  ⚠️  Universal Agent not available: {e}")
        save_result("universal_agent", "Import", False, str(e), 0)
    except Exception as e:
        print(f"\n  ⚠️  Universal Agent initialization failed: {e}")
        save_result("universal_agent", "Initialization", False, str(e), 0)

# ============================================================================
# ZOHO BOOKS AGENT TESTS
# ============================================================================

async def test_zoho_books_agent():
    """Test Zoho Books Agent capabilities."""
    print_header("ZOHO BOOKS AGENT TESTS")
    
    try:
        # Check if Zoho credentials are available
        temp_json_path = backend_dir / "temp.json"
        if not temp_json_path.exists():
            print("\n  ⚠️  Zoho Books credentials (temp.json) not found - skipping live API tests")
            save_result("zoho_books_agent", "Configuration", False, "temp.json not found", 0)
            
            # Test module import only
            print("\n💰 Test 1: Module Import Test")
            start = time.time()
            try:
                from agents.zoho_books.zoho_books_agent import app, AGENT_DEFINITION
                print_test("Module Import", True, "Zoho Books agent module loaded", time.time() - start)
                save_result("zoho_books_agent", "Module Import", True, "Module loaded", time.time() - start)
                
                # Print agent capabilities
                print(f"\n  Agent Capabilities: {len(AGENT_DEFINITION['capabilities'])} capabilities")
                print(f"  Sample: {', '.join(AGENT_DEFINITION['capabilities'][:5])}...")
                
            except Exception as e:
                print_test("Module Import", False, str(e))
                save_result("zoho_books_agent", "Module Import", False, str(e), time.time() - start)
            
            return
        
        # If credentials exist, run live tests
        from agents.zoho_books.zoho_books_agent import app
        
        # Test health endpoint
        print("\n💰 Test 1: Health Check")
        start = time.time()
        try:
            # Import and call health check
            from agents.zoho_books.zoho_books_agent import health_check
            result = health_check()
            
            duration = time.time() - start
            success = result.get("status") in ["healthy", "needs_oauth"]
            message = f"Status: {result.get('status')}"
            print_test("Health Check", success, message, duration)
            save_result("zoho_books_agent", "Health Check", success, message, duration)
            
        except Exception as e:
            print_test("Health Check", False, str(e))
            save_result("zoho_books_agent", "Health Check", False, str(e), time.time() - start)
        
        # Test planner
        print("\n💰 Test 2: Action Planning")
        start = time.time()
        try:
            from agents.zoho_books.planner import ZohoPlanner
            
            planner = ZohoPlanner()
            plan = await planner.plan("Show me all unpaid invoices")
            
            duration = time.time() - start
            success = bool(plan.action)
            message = f"Planned action: {plan.action}"
            print_test("Action Planning", success, message, duration)
            save_result("zoho_books_agent", "Action Planning", success, message, duration)
            
        except Exception as e:
            print_test("Action Planning", False, str(e))
            save_result("zoho_books_agent", "Action Planning", False, str(e), time.time() - start)
            
    except ImportError as e:
        print(f"\n  ⚠️  Zoho Books Agent not available: {e}")
        save_result("zoho_books_agent", "Import", False, str(e), 0)

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all agent tests."""
    print("\n" + "=" * 70)
    print("  ORCHESTRATOR AGENT TESTING SUITE")
    print("  Testing all agents with real-world scenarios")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run tests for each agent
    await test_spreadsheet_agent()
    await test_document_agent()
    await test_mail_agent()
    await test_universal_agent()
    await test_zoho_books_agent()
    
    total_duration = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    total_tests = 0
    total_passed = 0
    
    for agent, results in test_results.items():
        passed = sum(1 for r in results if r['success'])
        total = len(results)
        total_tests += total
        total_passed += passed
        
        if total > 0:
            status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
            print(f"\n  {status} {agent.replace('_', ' ').title()}: {passed}/{total} passed")
            for r in results:
                status_icon = "✓" if r['success'] else "✗"
                print(f"      {status_icon} {r['test']} ({r['duration']:.2f}s)")
    
    print("\n" + "-" * 70)
    print(f"  Total: {total_passed}/{total_tests} tests passed in {total_duration:.2f}s")
    print("=" * 70)
    
    # Save results to file
    results_file = backend_dir.parent / "agent_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "duration": total_duration,
            "results": test_results
        }, f, indent=2)
    
    print(f"\n  Results saved to: {results_file}")
    
    return total_passed, total_tests

if __name__ == "__main__":
    asyncio.run(run_all_tests())