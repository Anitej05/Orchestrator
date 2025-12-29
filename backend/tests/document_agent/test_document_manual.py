import logging
import sys
from pathlib import Path
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Get workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))

from agents.document_agent.agent import DocumentAgent
from agents.document_agent.schemas import (
    AnalyzeDocumentRequest,
    EditDocumentRequest,
    CreateDocumentRequest,
    DisplayDocumentRequest
)

def test_analyze_document():
    """Test 1: Analyze a document with a query."""
    print("\n\n" + "="*60)
    print("📄 TEST 1: Analyze Document")
    print("="*60 + "\n")
    
    # Put your test file in: backend/tests/test_data/
    # Example: backend/tests/test_data/sample_resume.pdf
    test_file = "backend/tests/test_data/sample_resume.pdf"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("📁 Please add test files to: backend/tests/test_data/")
        print("   Supported formats: PDF, DOCX, TXT")
        return
    
    agent = DocumentAgent()
    
    request = AnalyzeDocumentRequest(
        file_path=test_file,
        query="What are the key skills mentioned in this document?",
        thread_id="test-session-1"
    )
    
    try:
        result = agent.analyze_document(request)
        print(f"✅ Analysis Result: {result.get('answer', 'No answer')[:200]}...")
        print(f"📊 Success: {result.get('success', False)}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_edit_document():
    """Test 2: Edit a document with natural language instruction."""
    print("\n\n" + "="*60)
    print("✏️  TEST 2: Edit Document")
    print("="*60 + "\n")
    
    # Put your test file in: backend/tests/test_data/
    test_file = "backend/tests/test_data/sample_document.docx"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("📁 Please add a DOCX file to: backend/tests/test_data/")
        return
    
    agent = DocumentAgent()
    
    request = EditDocumentRequest(
        file_path=test_file,
        instruction="Add a professional summary section at the beginning highlighting key achievements",
        thread_id="test-session-2"
    )
    
    try:
        print("🔄 Editing document...")
        result = agent.edit_document(request)
        
        if result.get('success'):
            print(f"✅ SUCCESS: {result.get('message', 'Document edited')}")
            print(f"📄 File: {result.get('file_path', 'N/A')}")
            print(f"↩️  Can undo: {result.get('can_undo', False)}")
        else:
            print(f"❌ FAILED: {result.get('message', 'Unknown error')}")
            if 'error' in result:
                print(f"   Error details: {result['error']}")
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


def test_create_document():
    """Test 3: Create a new document from content."""
    print("\n\n" + "="*60)
    print("📝 TEST 3: Create New Document")
    print("="*60 + "\n")
    
    agent = DocumentAgent()
    
    content = """
# Project Proposal

## Executive Summary
This document outlines a comprehensive project proposal for implementing 
an AI-powered document management system.

## Key Objectives
1. Streamline document workflows
2. Enable natural language document queries
3. Automate document generation

## Timeline
- Phase 1: Q1 2025
- Phase 2: Q2 2025
- Phase 3: Q3 2025
"""
    
    request = CreateDocumentRequest(
        content=content,
        file_name="test_proposal.docx",
        output_dir=str(WORKSPACE_ROOT / "storage" / "documents"),
        thread_id="test-session-3"
    )
    
    try:
        print("🔄 Creating document...")
        result = agent.create_document(request)
        
        if result.get('success'):
            print(f"✅ SUCCESS: {result.get('message', 'Document created')}")
            file_path = result.get('file_path')
            print(f"📄 File: {file_path}")
            
            # Verify file exists
            if file_path and Path(file_path).exists():
                size = Path(file_path).stat().st_size
                print(f"✅ Verified - File exists ({size:,} bytes)")
                print(f"📂 Location: storage/documents/{Path(file_path).name}")
            else:
                print(f"⚠️  WARNING: File path returned but file not found")
        else:
            print(f"❌ FAILED: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all document agent tests."""
    print("\n\n" + "🔵"*30)
    print("🚀 DOCUMENT AGENT MANUAL TESTS")
    print("🔵"*30 + "\n")
    
    # Run tests
    test_analyze_document()
    test_edit_document()
    test_create_document()
    
    print("\n\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
