"""
Async File Processing Service

Handles file preprocessing (PDF parsing, vector store creation) 
asynchronously without blocking the orchestration graph.
"""

import os
import asyncio
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import httpx

from schemas import FileObject

logger = logging.getLogger(__name__)

# In-memory cache for processed files (use Redis in production)
_file_cache: Dict[str, Dict] = {}


class FileProcessor:
    """Async file processor with caching"""
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for cache key"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key from file path and content hash"""
        file_hash = self._compute_file_hash(file_path)
        return f"{os.path.basename(file_path)}:{file_hash}"
    
    async def process_document(
        self, 
        file_obj: FileObject,
        auto_display: bool = True
    ) -> Dict:
        """
        Process a single document file asynchronously.
        
        Returns dict with:
        - vector_store_path: Path to FAISS index
        - canvas_display: Canvas display object (if auto_display=True)
        - chunks_count: Number of text chunks
        - processing_time: Time taken in seconds
        - cached: Whether result was from cache
        """
        start_time = datetime.now()
        file_path = file_obj.file_path
        
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(file_path)
            if cache_key in _file_cache:
                logger.info(f"📦 Cache HIT for {file_obj.file_name}")
                cached_result = _file_cache[cache_key].copy()
                cached_result['cached'] = True
                cached_result['processing_time'] = 0.0
                return cached_result
        
        logger.info(f"⚡ Processing document async: {file_obj.file_name}")
        
        # Load document (runs in thread pool to avoid blocking)
        documents = await asyncio.to_thread(self._load_document, file_path)
        
        if not documents or sum(len(d.page_content) for d in documents) == 0:
            logger.error(f"❌ Document {file_obj.file_name} is empty")
            return {
                'error': 'Empty document',
                'cached': False,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Split into chunks (CPU-intensive, run in thread pool)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        texts = await asyncio.to_thread(text_splitter.split_documents, documents)
        logger.info(f"📄 Split into {len(texts)} chunks")
        
        # Create vector store (runs in thread pool)
        # Import here to avoid circular dependency
        from orchestrator.graph import get_hf_embeddings
        
        vector_store = await asyncio.to_thread(
            FAISS.from_documents,
            texts,
            get_hf_embeddings()
        )
        
        # Save vector store
        index_path = f"storage/vector_store/{os.path.basename(file_path)}.faiss"
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        await asyncio.to_thread(vector_store.save_local, index_path)
        
        result = {
            'vector_store_path': index_path,
            'chunks_count': len(texts),
            'total_chars': sum(len(d.page_content) for d in documents),
            'cached': False
        }
        
        # Auto-display if requested
        if auto_display:
            try:
                canvas_display = await self._display_document(file_path, file_obj.file_name)
                if canvas_display:
                    result['canvas_display'] = canvas_display
            except Exception as e:
                logger.error(f"Failed to display {file_obj.file_name}: {e}")
        
        # Cache the result
        if self.cache_enabled:
            cache_key = self._get_cache_key(file_path)
            _file_cache[cache_key] = result.copy()
            logger.info(f"📦 Cached result for {file_obj.file_name}")
        
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Processed {file_obj.file_name} in {result['processing_time']:.2f}s")
        
        return result
    
    def _load_document(self, file_path: str):
        """Load document synchronously (called in thread pool)"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        return loader.load()
    
    async def _display_document(self, file_path: str, file_name: str) -> Optional[Dict]:
        """Call document agent to display in canvas"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8070/display",
                    json={"file_path": file_path}
                )
                
                if response.status_code == 200:
                    display_result = response.json()
                    canvas_display = display_result.get("canvas_display")
                    if canvas_display:
                        logger.info(f"✅ Canvas display ready: {file_name}")
                        return canvas_display
                else:
                    logger.warning(f"Display failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Display error for {file_name}: {e}")
        
        return None
    
    async def process_spreadsheet(self, file_obj: FileObject) -> Dict:
        """Process spreadsheet file (upload to agent)"""
        start_time = datetime.now()
        
        # Check if already has file_id (skip re-upload)
        existing_file_id = file_obj.file_id or file_obj.content_id
        if existing_file_id:
            logger.info(f"📊 Reusing existing file_id: {existing_file_id}")
            return {
                'file_id': existing_file_id,
                'cached': True,
                'processing_time': 0.0
            }
        
        file_path = file_obj.file_path
        if not os.path.exists(file_path):
            return {'error': 'File not found'}
        
        try:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_obj.file_name)
            mime_type = mime_type or 'application/octet-stream'
            
            file_content = await asyncio.to_thread(
                lambda: open(file_path, 'rb').read()
            )
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                files = {"file": (file_obj.file_name, file_content, mime_type)}
                response = await client.post(
                    "http://localhost:8041/upload",
                    files=files
                )
                
                if response.status_code == 200:
                    upload_result = response.json()
                    result_data = upload_result.get('result', {})
                    
                    result = {
                        'file_id': result_data.get('file_id'),
                        'canvas_display': result_data.get('canvas_display'),
                        'cached': False,
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    }
                    
                    logger.info(f"✅ Spreadsheet uploaded: {file_obj.file_name}")
                    return result
                else:
                    logger.error(f"Upload failed: {response.status_code}")
                    return {'error': f'Upload failed: {response.status_code}'}
                    
        except Exception as e:
            logger.error(f"Spreadsheet processing error: {e}")
            return {'error': str(e)}
    
    async def process_files_batch(
        self,
        file_objects: List[FileObject],
        max_concurrent: int = 3
    ) -> List[Dict]:
        """
        Process multiple files concurrently with limit.
        
        Returns list of results in same order as input.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_obj: FileObject):
            async with semaphore:
                if file_obj.file_type == 'document':
                    return await self.process_document(file_obj)
                elif file_obj.file_type == 'spreadsheet':
                    return await self.process_spreadsheet(file_obj)
                else:
                    # Images don't need processing
                    return {'type': 'image', 'processing_time': 0.0}
        
        logger.info(f"⚡ Processing {len(file_objects)} files concurrently (max={max_concurrent})")
        tasks = [process_with_semaphore(fo) for fo in file_objects]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"File processing error: {result}")
                processed_results.append({
                    'error': str(result),
                    'file_name': file_objects[i].file_name
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def clear_cache(self):
        """Clear the file processing cache"""
        _file_cache.clear()
        logger.info("🗑️ File cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_files': len(_file_cache),
            'cache_keys': list(_file_cache.keys())
        }


# Global instance
file_processor = FileProcessor(cache_enabled=True)
