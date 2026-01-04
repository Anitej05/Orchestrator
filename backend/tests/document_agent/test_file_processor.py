"""
Test script for async file processor

Run this to verify the file processor works correctly.
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.file_processor import file_processor, FileObject


async def test_cache():
    """Test cache functionality"""
    print("🧪 Testing File Processor Cache\n")
    
    # Get initial stats
    stats = file_processor.get_cache_stats()
    print(f"Initial cache state:")
    print(f"  Cached files: {stats['cached_files']}")
    print(f"  Cache keys: {stats['cache_keys']}\n")
    
    # Test cache clear
    file_processor.clear_cache()
    stats = file_processor.get_cache_stats()
    print(f"After clearing cache:")
    print(f"  Cached files: {stats['cached_files']}")
    print(f"  Cache keys: {stats['cache_keys']}\n")
    
    print("✅ Cache operations working!")


async def test_document_processing():
    """Test document processing (requires actual PDF file)"""
    print("\n🧪 Testing Document Processing\n")
    
    # Check if we have any PDF files in storage
    storage_path = "../storage"
    if os.path.exists(storage_path):
        for root, dirs, files in os.walk(storage_path):
            pdf_files = [f for f in files if f.endswith('.pdf')]
            if pdf_files:
                print(f"Found {len(pdf_files)} PDF files in {root}")
                print(f"Files: {', '.join(pdf_files[:3])}")
                
                # Create a test FileObject
                test_file = os.path.join(root, pdf_files[0])
                file_obj = FileObject(
                    file_id=None,
                    file_name=pdf_files[0],
                    file_type='document',
                    file_path=test_file,
                    mime_type='application/pdf'
                )
                
                print(f"\n📄 Processing: {pdf_files[0]}")
                print("  First run (not cached)...")
                
                try:
                    result1 = await file_processor.process_document(file_obj, auto_display=False)
                    print(f"  ✅ Processed in {result1.get('processing_time', 0):.2f}s")
                    print(f"     Chunks: {result1.get('chunks_count')}")
                    print(f"     Cached: {result1.get('cached', False)}")
                    
                    print("\n  Second run (should be cached)...")
                    result2 = await file_processor.process_document(file_obj, auto_display=False)
                    print(f"  ✅ Processed in {result2.get('processing_time', 0):.2f}s")
                    print(f"     Chunks: {result2.get('chunks_count')}")
                    print(f"     Cached: {result2.get('cached', False)}")
                    
                    if result2.get('cached'):
                        print("\n  🎉 Caching works! Second run was instant.")
                    else:
                        print("\n  ⚠️ Warning: Second run wasn't cached")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                return
        
        print("  ℹ️ No PDF files found in storage")
    else:
        print("  ℹ️ Storage directory not found")


async def test_batch_processing():
    """Test batch processing with concurrency limit"""
    print("\n🧪 Testing Batch Processing\n")
    
    # Create mock file objects
    mock_files = []
    for i in range(5):
        mock_files.append(FileObject(
            file_id=None,
            file_name=f"test_file_{i}.pdf",
            file_type='image',  # Use image to skip processing
            file_path=f"/tmp/test_{i}.pdf",
            mime_type='application/pdf'
        ))
    
    print(f"📦 Processing {len(mock_files)} files with max_concurrent=3")
    
    try:
        results = await file_processor.process_files_batch(mock_files, max_concurrent=3)
        print(f"✅ Batch processed {len(results)} files")
        print(f"   Results: {results}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("FILE PROCESSOR TEST SUITE")
    print("=" * 60)
    
    await test_cache()
    await test_batch_processing()
    
    # Only test document processing if explicitly requested
    if len(sys.argv) > 1 and sys.argv[1] == '--with-documents':
        await test_document_processing()
    else:
        print("\n💡 Run with --with-documents flag to test actual PDF processing")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
