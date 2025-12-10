"""
Example usage of the Document Analysis Agent with Natural Language Instructions

This demonstrates how to:
1. Display existing documents in canvas
2. Create new documents
3. Edit documents using natural language instructions
4. Analyze documents with RAG
5. Use undo/redo functionality
"""

import httpx
import asyncio

DOCUMENT_AGENT_URL = "http://localhost:8070"

async def example_display_document():
    """Example: Display an existing document in canvas"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/display",
            json={
                "file_path": "backend/storage/documents/sample.docx"
            }
        )
        result = response.json()
        print("Display Result:", result)
        return result

async def example_create_document():
    """Example: Create a new document"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/create",
            json={
                "content": "This is a sample document.\n\nIt has multiple paragraphs.\n\nCreated by the document agent.",
                "file_name": "sample_report.docx",
                "file_type": "docx"
            }
        )
        create_result = response.json()
        print("Create Result:", create_result)
        return create_result

async def example_edit_with_natural_language():
    """Example: Edit documents using natural language instructions"""
    async with httpx.AsyncClient() as client:
        
        # Example 1: Format subheadings
        print("\n=== Example 1: Format subheadings ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/edit",
            json={
                "file_path": "backend/storage/documents/report.docx",
                "instruction": "Make all subheadings red and bold"
            }
        )
        print("Result:", response.json())
        
        # Example 2: Add content
        print("\n=== Example 2: Add bullet list ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/edit",
            json={
                "file_path": "backend/storage/documents/report.docx",
                "instruction": "Add a bullet list with three key findings: increased revenue, improved customer satisfaction, and reduced costs"
            }
        )
        print("Result:", response.json())
        
        # Example 3: Format specific text
        print("\n=== Example 3: Highlight important terms ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/edit",
            json={
                "file_path": "backend/storage/documents/report.docx",
                "instruction": "Highlight all instances of 'important' in yellow"
            }
        )
        print("Result:", response.json())
        
        # Example 4: Add table
        print("\n=== Example 4: Add sales table ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/edit",
            json={
                "file_path": "backend/storage/documents/report.docx",
                "instruction": "Add a table with 3 columns showing Q1, Q2, Q3 sales data with values $10k, $12k, $15k"
            }
        )
        print("Result:", response.json())
        
        # Example 5: Change section formatting
        print("\n=== Example 5: Format conclusion ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/edit",
            json={
                "file_path": "backend/storage/documents/report.docx",
                "instruction": "Make the conclusion section italic and add a page break before it"
            }
        )
        print("Result:", response.json())
        
        # Example 6: Add header/footer
        print("\n=== Example 6: Add header ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/edit",
            json={
                "file_path": "backend/storage/documents/report.docx",
                "instruction": "Add a centered header with the text 'Quarterly Sales Report 2024'"
            }
        )
        print("Result:", response.json())
        
        # Example 7: Complex formatting
        print("\n=== Example 7: Complex formatting ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/edit",
            json={
                "file_path": "backend/storage/documents/report.docx",
                "instruction": "Change all headings to blue, make them size 16, and add a yellow highlight to any text that says 'critical'"
            }
        )
        print("Result:", response.json())

async def example_undo_redo():
    """Example: Use undo/redo functionality"""
    async with httpx.AsyncClient() as client:
        
        # Make an edit
        print("\n=== Making an edit ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/edit",
            json={
                "file_path": "backend/storage/documents/report.docx",
                "instruction": "Make all headings green"
            }
        )
        print("Edit Result:", response.json())
        
        # Undo the edit
        print("\n=== Undoing the edit ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/undo",
            json={
                "file_path": "backend/storage/documents/report.docx"
            }
        )
        print("Undo Result:", response.json())
        
        # Redo the edit
        print("\n=== Redoing the edit ===")
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/redo",
            json={
                "file_path": "backend/storage/documents/report.docx"
            }
        )
        print("Redo Result:", response.json())

async def example_version_history():
    """Example: Get version history"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/history",
            json={
                "file_path": "backend/storage/documents/report.docx"
            }
        )
        result = response.json()
        print("Version History:", result)
        return result

async def example_analyze_document():
    """Example: Analyze document with RAG"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DOCUMENT_AGENT_URL}/analyze",
            json={
                "vector_store_path": "backend/storage/documents/report_faiss_index",
                "query": "What are the main conclusions of this report?"
            }
        )
        result = response.json()
        print("Analysis Result:", result)
        return result

async def run_all_examples():
    """Run all examples"""
    print("=" * 60)
    print("Document Agent Examples - Natural Language Editing")
    print("=" * 60)
    
    # Display document
    print("\n### Example: Display Document ###")
    await example_display_document()
    
    # Create document
    print("\n### Example: Create Document ###")
    await example_create_document()
    
    # Edit with natural language
    print("\n### Example: Edit with Natural Language ###")
    await example_edit_with_natural_language()
    
    # Undo/Redo
    print("\n### Example: Undo/Redo ###")
    await example_undo_redo()
    
    # Version history
    print("\n### Example: Version History ###")
    await example_version_history()
    
    # Analyze document
    print("\n### Example: Analyze Document ###")
    await example_analyze_document()

if __name__ == "__main__":
    asyncio.run(run_all_examples())
