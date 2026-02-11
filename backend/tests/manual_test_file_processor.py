
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import io
import asyncio
from datetime import datetime

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set path for imports
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.schemas import FileObject
# We import FileProcessor but will patch its dependencies
from backend.services.file_processor import FileProcessor

class TestFileProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FileProcessor(cache_enabled=True)
        # Mock CMS
        self.processor.cms = AsyncMock()
        self.processor.cms.register_content.return_value = MagicMock(id="cms-123")
        
    @patch('services.file_processor.RecursiveCharacterTextSplitter')
    @patch('services.file_processor.FAISS')
    @patch('orchestrator.nodes.utils.get_hf_embeddings')
    async def test_process_document(self, mock_get_embeddings, mock_faiss, mock_splitter):
        print("\n=== Testing Document Processing ===")
        
        # Mock dependencies
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content " * 100
        
        # Mock _load_document
        with patch.object(self.processor, '_load_document', return_value=[mock_doc]) as mock_load:
            # Mock split (must trigger vector store creation)
            mock_splitter.return_value.split_documents.return_value = ["chunk1", "chunk2"]
            
            # Mock FAISS
            mock_vector_store = MagicMock()
            mock_faiss.from_documents.return_value = mock_vector_store
            
            file_obj = FileObject(
                file_path="/tmp/test.txt",
                file_name="test.txt",
                file_type="document"
            )
            
            # Run
            result = await self.processor.process_document(file_obj, auto_display=False)
            
            print(f"Result: {result}")
            
            # Verify
            self.assertIn('vector_store_path', result)
            self.assertEqual(result['chunks_count'], 2)
            self.assertEqual(result['content_id'], "cms-123")
            
            # Verify calls
            mock_load.assert_called_with("/tmp/test.txt")
            mock_faiss.from_documents.assert_called()
            mock_vector_store.save_local.assert_called()
            print("✅ Document processing logic verified")

    @patch('httpx.AsyncClient')
    async def test_process_spreadsheet(self, mock_client_cls):
        print("\n=== Testing Spreadsheet Processing ===")
        
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "file_id": "spread-123",
                "canvas_display": {"type": "table"}
            }
        }
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client
        
        file_obj = FileObject(
            file_path="/tmp/data.xlsx",
            file_name="data.xlsx",
            file_type="spreadsheet"
        )
        
        # Ensure file check passes (mock os.path.exists)
        with patch('os.path.exists', return_value=True):
             with patch('builtins.open', mock_open(read_data=b"fake data")):
                result = await self.processor.process_spreadsheet(file_obj)
        
        print(f"Result: {result}")
        self.assertEqual(result['file_id'], "spread-123")
        print("✅ Spreadsheet processing logic verified")

def mock_open(read_data=b''):
    file_mock = MagicMock()
    file_mock.read.return_value = read_data
    file_mock.__enter__.return_value = file_mock
    return lambda *args, **kwargs: file_mock

async def main():
    # Run async tests manually
    test = TestFileProcessor()
    test.setUp()
    
    # Needs patches applied manually if calling methods directly, 
    # but unittest.main() does discovery.
    # To run async tests with unittest, we usually need IsolatedAsyncioTestCase
    # IF available (Python 3.8+).
    # Since we are writing a manual script, let's just run logic.
    pass

if __name__ == "__main__":
    # Helper to run async tests in unittest
    # Or we can just just use IsolatedAsyncioTestCase if python version supports
    # Let's try standard unittest with async helper if needed, 
    # or just use unittest.IsolatedAsyncioTestCase (Standard in 3.8+)
    from unittest import IsolatedAsyncioTestCase
    
    class AsyncTest(IsolatedAsyncioTestCase, TestFileProcessor):
        pass
        
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
