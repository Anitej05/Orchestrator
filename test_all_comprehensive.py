import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

async def test_agent_sequence(agent_class, name, prompts):
    print(f"\n{'='*50}\nTesting {name} Comprehensively\n{'='*50}")
    try:
        agent = agent_class()
        await agent.initialize()
        print(f"[{name}] Initialized successfully.")
        
        from backend.agents.base.types import AgentRequest
        
        for p in prompts:
            print(f"\n--- [{name}] Prompt: '{p}' ---")
            request = AgentRequest(
                prompt=p,
                thread_id=f"test_comprehensive_{name.replace(' ', '_')}"
            )
            response = await agent.execute(request)
            print(f"[{name}] Status: {response.status}")
            if response.status == "error":
                print(f"[{name}] Error: {response.error_message}")
            else:
                print(f"[{name}] Summary: {getattr(response, 'summary', 'N/A')}")
            
            # Throttle to avoid rate limits
            await asyncio.sleep(15) # Wait 15 seconds between prompts
            
    except Exception as e:
        import traceback
        print(f"[{name}] Exception during test sequence:")
        traceback.print_exc()

async def main():
    from backend.agents.pdf_agent.base_agent_impl import PDFAgent
    from backend.agents.document_agent_lib.base_agent_impl import DocumentAgent
    from backend.agents.ppt_agent.base_agent_impl import PPTAgent
    from backend.agents.spreadsheet_agent.base_agent_impl import SpreadsheetAgent
    
    storage_dir = os.path.join(os.path.dirname(__file__), "storage")
    os.makedirs(storage_dir, exist_ok=True)
    
    pdf_path = os.path.join(storage_dir, "comp_test.pdf")
    docx_path = os.path.join(storage_dir, "comp_test.docx")
    pptx_path = os.path.join(storage_dir, "comp_test.pptx")
    
    # 1. PDF Tests
    pdf_prompts = [
        f"Create a PDF with the title 'Comprehensive PDF' and content 'Page 1 text'. Save as {pdf_path}",
        f"Extract text from {pdf_path}",
        f"Extract metadata from {pdf_path}",
        f"Rotate pages in {pdf_path} by 90 degrees"
    ]
    await test_agent_sequence(PDFAgent, "PDF Agent", pdf_prompts)
    await asyncio.sleep(15)
    
    # 2. DOCX Tests
    docx_prompts = [
        f"Create a new document titled 'Comprehensive Doc' with the content 'Initial body text' saved as {docx_path}",
        f"Edit {docx_path} to add a paragraph with text 'Here is a new paragraph'",
        f"Extract data from {docx_path} for 'summary'",
        f"Get version history of {docx_path}"
    ]
    await test_agent_sequence(DocumentAgent, "Document Agent", docx_prompts)
    await asyncio.sleep(15)
    
    # 3. PPTX Tests
    pptx_prompts = [
        f"Create a new presentation with 1 slide. Title is 'Comp Test', content is 'Hello'. Save as {pptx_path}",
        f"Read and analyze presentation {pptx_path}",
        f"Edit {pptx_path} by adding a new slide with title 'Slide 2'",
        f"Extract all text from {pptx_path}"
    ]
    await test_agent_sequence(PPTAgent, "PPT Agent", pptx_prompts)
    await asyncio.sleep(15)
    
    # 4. XLSX Tests
    csv_path = os.path.join(storage_dir, "comp_test_data.csv")
    with open(csv_path, "w") as f:
        f.write("Department,Sales,Employees\nSales,100,5\nEngineering,500,20\nMarketing,50,2\n")
        
    xlsx_prompts = [
        f"Load {csv_path}",
        "Process the loaded data to filter for departments with Employees > 4",
        "Export the processed data as comp_test_export.xlsx"
    ]
    await test_agent_sequence(SpreadsheetAgent, "Spreadsheet Agent", xlsx_prompts)

if __name__ == "__main__":
    asyncio.run(main())
