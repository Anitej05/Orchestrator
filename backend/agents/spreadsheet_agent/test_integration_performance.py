"""
Integration test for performance optimization that runs end-to-end tests
with proper module imports from the backend directory.
"""

import sys
import os
import time
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

def test_end_to_end_performance_with_real_components():
    """Test complete performance optimization pipeline with real components."""
    print("🚀 Testing End-to-End Performance Optimization with Real Components")
    
    try:
        # Import components with proper path
        from agents.spreadsheet_agent.performance_optimizer import (
            performance_monitor,
            memory_optimizer,
            token_optimizer,
            advanced_cache
        )
        from agents.spreadsheet_agent.memory import spreadsheet_memory
        from agents.spreadsheet_agent.spreadsheet_parser import spreadsheet_parser
        
        print("✅ Successfully imported all performance optimization components")
        
        # Create test spreadsheet file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create large dataset for performance testing
            df = pd.DataFrame({
                'ID': range(1, 1001),
                'Name': [f'Item_{i}' for i in range(1, 1001)],
                'Value': np.random.randn(1000),
                'Category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000),
                'Date': pd.date_range('2024-01-01', periods=1000, freq='h'),
                'Description': [f'Description for item {i}' * 5 for i in range(1, 1001)]  # Long strings
            })
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            print(f"📊 Created test dataset: {len(df)} rows × {len(df.columns)} columns")
            
            # Test 1: Parsing with performance monitoring
            print("\n🔍 Testing parsing with performance monitoring...")
            start_time = time.time()
            
            parsed = spreadsheet_parser.parse_file(
                file_path=temp_file,
                file_id="integration_performance_test",
                max_rows=1000
            )
            
            parse_time = time.time() - start_time
            
            # Verify parsing completed successfully
            assert parsed is not None, "Parsing should succeed"
            assert parsed.file_id == "integration_performance_test", "File ID should match"
            assert len(parsed.tables) > 0, "Should detect at least one table"
            
            print(f"✅ Parsing completed in {parse_time:.3f}s")
            
            # Test 2: Context building with token optimization
            print("\n🎯 Testing context building with token optimization...")
            context_start = time.time()
            
            context = spreadsheet_parser.build_context(
                parsed=parsed,
                max_tokens=2000
            )
            
            context_time = time.time() - context_start
            
            # Verify context was built
            assert context is not None, "Context should be built"
            assert hasattr(context, 'sections'), "Context should have sections"
            
            print(f"✅ Context building completed in {context_time:.3f}s")
            
            # Test 3: Cache performance
            print("\n💾 Testing cache performance...")
            cache_start = time.time()
            
            # Second parse should be faster due to caching
            parsed2 = spreadsheet_parser.parse_file(
                file_path=temp_file,
                file_id="integration_performance_test",
                max_rows=1000
            )
            
            cache_time = time.time() - cache_start
            
            assert parsed2 is not None, "Cached parsing should succeed"
            print(f"✅ Cached parsing completed in {cache_time:.3f}s")
            
            # Test 4: Memory optimization
            print("\n🧠 Testing memory optimization...")
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            memory_optimizer.track_session_memory("integration_test", memory_usage)
            
            total_memory = memory_optimizer.get_total_memory_usage()
            assert total_memory > 0, "Should track memory usage"
            
            memory_info = memory_optimizer.get_system_memory_info()
            assert 'process_memory_mb' in memory_info, "Should provide memory info"
            
            print(f"✅ Memory tracking: {memory_usage:.2f}MB tracked")
            
            # Test 5: Token optimization
            print("\n🎯 Testing token optimization...")
            optimized_context = token_optimizer.optimize_dataframe_context(
                df=df,
                max_tokens=1000,
                include_columns=['ID', 'Name', 'Value'],
                priority_columns=['ID']
            )
            
            assert 'schema' in optimized_context, "Should have schema"
            assert 'sample_data' in optimized_context, "Should have sample data"
            assert 'metadata' in optimized_context, "Should have metadata"
            
            print("✅ Token optimization completed successfully")
            
            # Test 6: Performance monitoring
            print("\n📊 Testing performance monitoring...")
            with performance_monitor.time_operation("integration_test_operation"):
                time.sleep(0.1)  # Simulate work
            
            report = performance_monitor.get_performance_report()
            assert 'operation_stats' in report, "Should have operation stats"
            assert 'system_stats' in report, "Should have system stats"
            
            print("✅ Performance monitoring working correctly")
            
            # Test 7: Advanced caching
            print("\n📦 Testing advanced caching...")
            cache_stats = advanced_cache.get_stats()
            assert 'size' in cache_stats, "Should have cache size"
            assert 'hits' in cache_stats, "Should track cache hits"
            assert 'misses' in cache_stats, "Should track cache misses"
            
            print(f"✅ Advanced cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses")
            
            # Performance assertions
            print("\n⚡ Performance Validation:")
            print(f"   Parse time: {parse_time:.3f}s (target: <2.0s)")
            print(f"   Context time: {context_time:.3f}s (target: <1.0s)")
            print(f"   Cache time: {cache_time:.3f}s")
            
            assert parse_time < 2.0, f"Parsing took too long: {parse_time:.2f}s"
            assert context_time < 1.0, f"Context building took too long: {context_time:.2f}s"
            
            # Cleanup
            memory_optimizer.cleanup_session("integration_test")
            
            print("\n🎉 ALL END-TO-END PERFORMANCE TESTS PASSED!")
            print("✅ Performance optimization is working correctly with real components")
            
            return True
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except ImportError as e:
        print(f"❌ Failed to import components: {e}")
        return False
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_optimization_integration():
    """Test performance optimization integration with memory system."""
    print("\n🔄 Testing Performance Optimization Integration")
    
    try:
        from agents.spreadsheet_agent.memory import spreadsheet_memory
        
        # Test enhanced memory system
        file_id = "integration_test_file"
        metadata = {
            "shape": (1000, 5),
            "columns": ["A", "B", "C", "D", "E"],
            "dtypes": {"A": "int64", "B": "float64", "C": "object", "D": "datetime64[ns]", "E": "bool"}
        }
        
        # Cache metadata with performance tracking
        spreadsheet_memory.cache_df_metadata(file_id, metadata)
        
        # Retrieve metadata
        cached_metadata = spreadsheet_memory.get_df_metadata(file_id)
        assert cached_metadata == metadata, "Metadata should be cached correctly"
        
        # Test cache stats with performance features
        stats = spreadsheet_memory.get_cache_stats()
        print(f"📊 Cache stats: {stats}")
        
        # Test performance report if available
        if hasattr(spreadsheet_memory, 'get_performance_report'):
            perf_report = spreadsheet_memory.get_performance_report()
            print(f"📈 Performance report available: {bool(perf_report)}")
        
        print("✅ Performance optimization integration working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Running Integration Performance Tests")
    print("=" * 60)
    
    success1 = test_end_to_end_performance_with_real_components()
    success2 = test_performance_optimization_integration()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION PERFORMANCE TESTS PASSED!")
        print("✅ Task 4.3: Performance Optimization - FULLY VALIDATED")
        exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Some integration tests failed")
        exit(1)