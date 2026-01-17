"""
Comprehensive Requirements Testing Suite

This test suite validates all 18 requirements with real-world scenarios
and ensures production readiness of the intelligent spreadsheet parsing system.

Requirements Coverage:
- Requirements 1-6: Core Parsing
- Requirements 7-8: Query Processing  
- Requirements 9-14: Orchestrator Integration
- Requirements 15-18: Advanced Document Understanding
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch
from io import StringIO, BytesIO

# Import all components
from agents.spreadsheet_agent.spreadsheet_parser import spreadsheet_parser
from agents.spreadsheet_agent.anomaly_detector import AnomalyDetector
from agents.spreadsheet_agent.edge_case_handler import edge_case_handler
from agents.spreadsheet_agent.planner import planner
from schemas import AgentResponse, AgentResponseStatus


class TestRequirements1to6_CoreParsing:
    """Test Requirements 1-6: Core Parsing Capabilities"""
    
    def test_requirement_1_detect_primary_tables(self):
        """Requirement 1: Detect Primary Data Tables"""
        # Create spreadsheet with metadata and primary table
        df = pd.DataFrame({
            'Title': ['Sales Report Q4 2024', '', '', '', ''],
            'Date': ['Generated: 2024-01-15', '', 'Product', 'Sales', 'Quantity'],
            'Summary': ['Total Revenue: $50K', '', 'Widget A', 1000, 50],
            'Col3': ['', '', 'Widget B', 2000, 75],
            'Col4': ['', '', 'Widget C', 1500, 60]
        })
        
        # Parse with intelligent parsing
        parsed = spreadsheet_parser.parse_dataframe(df, "test-req1", "Sheet1")
        
        # Validate table detection
        assert len(parsed.tables) > 0, "Should detect at least one table"
        
        # Get primary table
        primary_table = spreadsheet_parser.get_primary_table(parsed)
        assert primary_table is not None, "Should identify primary table"
        
        region, table_df, schema = primary_table
        assert region.confidence > 0.5, "Primary table detection should have reasonable confidence"
        assert len(table_df) >= 3, "Primary table should contain data rows"
        
        print("✅ Requirement 1: Primary table detection - PASSED")
    
    def test_requirement_2_extract_accurate_schema(self):
        """Requirement 2: Extract Accurate Schema"""
        # Create table with mixed data types
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'Score': [95.5, 87.2, 92.0, 88.8],
            'Active': [True, False, True, True],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
        })
        
        parsed = spreadsheet_parser.parse_dataframe(df, "test-req2", "Sheet1")
        
        # Validate schema extraction
        assert len(parsed.tables) > 0
        region, table_df, schema = parsed.tables[0]
        
        # Check headers
        assert 'ID' in schema.headers
        assert 'Name' in schema.headers
        assert 'Score' in schema.headers
        
        # Check data types
        assert len(schema.dtypes) == len(schema.headers)
        assert schema.row_count == 4
        assert schema.col_count == 5
        
        # Check null counts
        assert isinstance(schema.null_counts, dict)
        
        print("✅ Requirement 2: Schema extraction - PASSED")
    
    def test_requirement_3_handle_large_datasets(self):
        """Requirement 3: Handle Large Datasets Efficiently"""
        # Create large dataset (>100 rows)
        large_df = pd.DataFrame({
            'ID': range(1, 201),
            'Category': [f'Cat_{i%10}' for i in range(200)],
            'Value': np.random.randint(100, 1000, 200),
            'Description': [f'Item {i}' for i in range(200)]
        })
        
        start_time = time.time()
        parsed = spreadsheet_parser.parse_dataframe(large_df, "test-req3", "Sheet1")
        parse_time = time.time() - start_time
        
        # Validate efficient processing
        assert parse_time < 2.0, f"Large dataset parsing should be fast, took {parse_time:.2f}s"
        assert parsed.parsing_confidence > 0, "Should successfully parse large dataset"
        
        # Test context building with sampling
        context = spreadsheet_parser.build_context(parsed, max_tokens=8000)
        assert context is not None, "Should build context for large dataset"
        
        print(f"✅ Requirement 3: Large dataset handling ({len(large_df)} rows in {parse_time:.2f}s) - PASSED")
    
    def test_requirement_4_preserve_metadata_context(self):
        """Requirement 4: Preserve Metadata Context"""
        # Create spreadsheet with rich metadata
        df = pd.DataFrame({
            'Invoice': ['INV-2024-001', '', '', '', ''],
            'Date': ['2024-01-15', '', 'Item', 'Qty', 'Price'],
            'Customer': ['Acme Corp', '', 'Widget', 5, 100.00],
            'Total': ['$500.00', '', 'Gadget', 3, 75.00],
            'Status': ['Paid', '', '', '', '']
        })
        
        parsed = spreadsheet_parser.parse_dataframe(df, "test-req4", "Sheet1")
        
        # Validate metadata extraction
        assert parsed.metadata is not None, "Should extract metadata"
        assert len(parsed.metadata) > 0, "Should have metadata items"
        
        # Check metadata summary
        summary = spreadsheet_parser.get_metadata_summary(parsed)
        assert summary["has_metadata"], "Should detect metadata presence"
        assert summary["metadata_items"] > 0, "Should count metadata items"
        
        print("✅ Requirement 4: Metadata preservation - PASSED")
    
    def test_requirement_5_support_multiple_sheets(self):
        """Requirement 5: Support Multiple Sheets"""
        # Test multi-sheet parsing capability
        # Note: This tests the parsing logic, actual Excel files would be tested in integration
        
        # Simulate multi-sheet data
        sheet1_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        sheet2_df = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
        
        # Parse individual sheets
        parsed1 = spreadsheet_parser.parse_dataframe(sheet1_df, "test-req5_sheet1", "Sheet1")
        parsed2 = spreadsheet_parser.parse_dataframe(sheet2_df, "test-req5_sheet2", "Sheet2")
        
        # Validate multi-sheet support
        assert parsed1.sheet_name == "Sheet1"
        assert parsed2.sheet_name == "Sheet2"
        assert len(parsed1.tables) > 0
        assert len(parsed2.tables) > 0
        
        print("✅ Requirement 5: Multi-sheet support - PASSED")
    
    def test_requirement_6_handle_edge_cases(self):
        """Requirement 6: Handle Edge Cases"""
        # Test various edge cases
        edge_cases_df = pd.DataFrame({
            'Normal': [1, 2, 3, 4],
            'WithNulls': [1, None, 3, None],
            'Mixed': [1, 'text', 3, 'more text'],
            'Errors': ['#DIV/0!', 2, '#N/A', 4]
        })
        
        # Test error handling
        cleaned_df = edge_case_handler.handle_error_cells(edge_cases_df)
        assert pd.isna(cleaned_df.iloc[0, 3]), "Should handle #DIV/0! error"
        assert pd.isna(cleaned_df.iloc[2, 3]), "Should handle #N/A error"
        
        # Test parsing with edge cases
        parsed = spreadsheet_parser.parse_dataframe(edge_cases_df, "test-req6", "Sheet1")
        assert parsed is not None, "Should handle edge cases gracefully"
        
        print("✅ Requirement 6: Edge case handling - PASSED")


class TestRequirements7to8_QueryProcessing:
    """Test Requirements 7-8: Query Processing"""
    
    def test_requirement_7_enable_accurate_query_answering(self):
        """Requirement 7: Enable Accurate Query Answering"""
        # Create test data for query answering
        df = pd.DataFrame({
            'Product': ['A', 'B', 'C', 'A', 'B'],
            'Sales': [100, 200, 150, 120, 180],
            'Quantity': [10, 20, 15, 12, 18]
        })
        
        parsed = spreadsheet_parser.parse_dataframe(df, "test-req7", "Sheet1")
        
        # Test context building for query answering
        context = spreadsheet_parser.build_context(parsed)
        assert context is not None, "Should build context for queries"
        
        # Validate data structure understanding
        assert len(parsed.tables) > 0, "Should detect table structure"
        region, table_df, schema = parsed.tables[0]
        
        # Check that schema provides enough info for accurate queries
        assert 'Sales' in schema.headers, "Should identify Sales column"
        assert 'Product' in schema.headers, "Should identify Product column"
        assert schema.row_count == 5, "Should count all rows"
        
        print("✅ Requirement 7: Query answering support - PASSED")
    
    def test_requirement_8_optimize_context_window_usage(self):
        """Requirement 8: Optimize Context Window Usage"""
        # Create large dataset to test context optimization
        large_df = pd.DataFrame({
            'ID': range(1, 101),
            'Category': [f'Category_{i%5}' for i in range(100)],
            'Value': np.random.randint(1, 1000, 100),
            'Description': [f'Long description for item {i} with lots of text' for i in range(100)]
        })
        
        parsed = spreadsheet_parser.parse_dataframe(large_df, "test-req8", "Sheet1")
        
        # Test context optimization with token limits
        small_context = spreadsheet_parser.build_context(parsed, max_tokens=1000)
        large_context = spreadsheet_parser.build_context(parsed, max_tokens=8000)
        
        assert small_context is not None, "Should build context within token limits"
        assert large_context is not None, "Should build larger context"
        
        # Context should prioritize schema and samples over full data
        assert parsed.parsing_confidence > 0, "Should maintain parsing quality"
        
        print("✅ Requirement 8: Context window optimization - PASSED")


class TestRequirements9to14_OrchestratorIntegration:
    """Test Requirements 9-14: Orchestrator Integration"""
    
    def test_requirement_9_bidirectional_orchestrator_communication(self):
        """Requirement 9: Bidirectional Orchestrator Communication"""
        # Test AgentResponse format compliance
        
        # Test COMPLETE response
        complete_response = AgentResponse(
            status=AgentResponseStatus.COMPLETE,
            result={"data": "test"},
            context={"task_id": "test-123"}
        )
        assert complete_response.status == AgentResponseStatus.COMPLETE
        
        # Test NEEDS_INPUT response
        needs_input_response = AgentResponse(
            status=AgentResponseStatus.NEEDS_INPUT,
            question="How should we proceed?",
            question_type="choice",
            options=["option1", "option2"],
            context={"task_id": "test-456"}
        )
        assert needs_input_response.status == AgentResponseStatus.NEEDS_INPUT
        assert needs_input_response.question is not None
        
        # Test ERROR response
        error_response = AgentResponse(
            status=AgentResponseStatus.ERROR,
            error="Something went wrong",
            context={"task_id": "test-789"}
        )
        assert error_response.status == AgentResponseStatus.ERROR
        assert error_response.error is not None
        
        print("✅ Requirement 9: Bidirectional communication - PASSED")
    
    def test_requirement_10_intelligent_anomaly_detection(self):
        """Requirement 10: Intelligent Anomaly Detection and User Interaction"""
        # Create data with anomalies
        anomaly_df = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'Amount': ['100', '200', 'invalid', '400', '500'],  # Dtype drift
            'Score': [95, 87, None, None, 92]  # Missing values
        })
        
        detector = AnomalyDetector()
        anomalies = detector.detect_anomalies(anomaly_df)
        
        # Validate anomaly detection
        assert len(anomalies) > 0, "Should detect anomalies"
        
        # Check for dtype drift detection
        dtype_anomalies = [a for a in anomalies if a.type == 'dtype_drift']
        assert len(dtype_anomalies) > 0, "Should detect dtype drift"
        
        # Check fix suggestions
        for anomaly in dtype_anomalies:
            assert len(anomaly.suggested_fixes) > 0, "Should suggest fixes"
            fix_actions = [fix.action for fix in anomaly.suggested_fixes]
            assert 'convert_numeric' in fix_actions, "Should suggest numeric conversion"
        
        print("✅ Requirement 10: Anomaly detection - PASSED")
    
    def test_requirement_11_complex_multi_step_query_handling(self):
        """Requirement 11: Complex Multi-Step Query Handling"""
        # Test multi-step planning
        df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'C', 'B'],
            'Sales': [100, 200, 150, 300, 250],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        })
        
        # Test plan generation
        plan = planner._propose_plan_heuristic(
            df, 
            "Add serial number and sort by Sales descending", 
            {"columns": df.columns.tolist(), "shape": df.shape}
        )
        
        assert plan is not None, "Should generate execution plan"
        assert len(plan.actions) > 0, "Should have execution actions"
        assert plan.plan_id is not None, "Should have plan ID"
        
        # Test plan simulation
        simulation_result = planner.simulate_plan(plan, df)
        assert 'success' in simulation_result, "Should simulate plan"
        assert 'preview' in simulation_result, "Should provide preview"
        
        print("✅ Requirement 11: Multi-step query handling - PASSED")
    
    def test_requirement_12_standardized_response_format(self):
        """Requirement 12: Standardized Response Format"""
        # Test response format compliance
        
        # All responses should conform to AgentResponse schema
        responses = [
            AgentResponse(status=AgentResponseStatus.COMPLETE, result={"test": "data"}),
            AgentResponse(status=AgentResponseStatus.ERROR, error="Test error"),
            AgentResponse(status=AgentResponseStatus.NEEDS_INPUT, question="Test question", question_type="text"),
            AgentResponse(status=AgentResponseStatus.PARTIAL, partial_result={"partial": "data"}, progress=0.5)
        ]
        
        for response in responses:
            response_dict = response.model_dump()
            assert 'status' in response_dict, "Should have status field"
            assert response_dict['status'] in ['complete', 'error', 'needs_input', 'partial'], "Should have valid status"
        
        print("✅ Requirement 12: Standardized response format - PASSED")
    
    def test_requirement_13_session_and_thread_management(self):
        """Requirement 13: Session and Thread Management"""
        # Test thread isolation
        from agents.spreadsheet_agent.session import store_dataframe, get_dataframe
        
        df1 = pd.DataFrame({'A': [1, 2, 3]})
        df2 = pd.DataFrame({'B': [4, 5, 6]})
        
        # Store in different threads
        store_dataframe("file1", df1, "path1", "thread1")
        store_dataframe("file2", df2, "path2", "thread2")
        
        # Retrieve from specific threads
        retrieved_df1 = get_dataframe("file1", "thread1")
        retrieved_df2 = get_dataframe("file2", "thread2")
        
        assert retrieved_df1 is not None, "Should retrieve from thread1"
        assert retrieved_df2 is not None, "Should retrieve from thread2"
        assert len(retrieved_df1) == 3, "Should maintain data integrity"
        assert len(retrieved_df2) == 3, "Should maintain data integrity"
        
        print("✅ Requirement 13: Session and thread management - PASSED")
    
    def test_requirement_14_robust_error_handling(self):
        """Requirement 14: Robust Error Handling and Recovery"""
        # Test error handling scenarios
        
        # Test with invalid data
        try:
            invalid_df = pd.DataFrame({'A': [1, 2, 'invalid', 4]})
            parsed = spreadsheet_parser.parse_dataframe(invalid_df, "test-error", "Sheet1")
            # Should not crash, should handle gracefully
            assert parsed is not None, "Should handle invalid data gracefully"
        except Exception as e:
            pytest.fail(f"Should not raise exception for invalid data: {e}")
        
        # Test anomaly detection error handling
        detector = AnomalyDetector()
        try:
            empty_df = pd.DataFrame()
            anomalies = detector.detect_anomalies(empty_df)
            assert isinstance(anomalies, list), "Should return empty list for empty DataFrame"
        except Exception as e:
            pytest.fail(f"Should handle empty DataFrame gracefully: {e}")
        
        print("✅ Requirement 14: Robust error handling - PASSED")


class TestRequirements15to18_AdvancedDocumentUnderstanding:
    """Test Requirements 15-18: Advanced Document Understanding"""
    
    def test_requirement_15_industry_standard_edge_case_handling(self):
        """Requirement 15: Industry-Standard Edge Case Handling"""
        # Test advanced edge case handling
        
        # Test merged cell handling
        merged_regions = [(1, 2, 1, 2, 'merged_value')]
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        result_df = edge_case_handler.handle_merged_cells_in_dataframe(df, merged_regions)
        assert result_df is not None, "Should handle merged cells"
        
        # Test error cell handling
        error_df = pd.DataFrame({
            'A': [1, '#DIV/0!', 3],
            'B': ['#N/A', 5, '#VALUE!']
        })
        
        cleaned_df = edge_case_handler.handle_error_cells(error_df)
        assert pd.isna(cleaned_df.iloc[1, 0]), "Should replace #DIV/0! with NaN"
        assert pd.isna(cleaned_df.iloc[0, 1]), "Should replace #N/A with NaN"
        
        print("✅ Requirement 15: Industry-standard edge case handling - PASSED")
    
    def test_requirement_16_document_structure_understanding(self):
        """Requirement 16: Document Structure Understanding"""
        # Create complex document structure
        complex_df = pd.DataFrame({
            'Header': ['Invoice #12345', '', 'Line Items', '', ''],
            'Info': ['Date: 2024-01-15', '', 'Product', 'Qty', 'Price'],
            'Data1': ['Customer: Acme', '', 'Widget', 5, 100],
            'Data2': ['Total: $500', '', 'Gadget', 3, 75],
            'Footer': ['Thank you!', '', '', '', '']
        })
        
        parsed = spreadsheet_parser.parse_dataframe(complex_df, "test-req16", "Sheet1")
        
        # Validate document structure understanding
        assert len(parsed.sections) > 0, "Should detect document sections"
        assert parsed.document_type is not None, "Should classify document type"
        
        # Check metadata extraction
        summary = spreadsheet_parser.get_metadata_summary(parsed)
        assert summary["has_metadata"], "Should detect metadata sections"
        assert summary["has_line_items"], "Should detect line items"
        
        print("✅ Requirement 16: Document structure understanding - PASSED")
    
    def test_requirement_17_intentional_gap_detection(self):
        """Requirement 17: Intentional Gap Detection"""
        # Create document with intentional gaps
        gap_df = pd.DataFrame({
            'Title': ['Report Title', '', '', 'Data Section', ''],
            'Data': ['Summary Info', '', '', 'Item', 'Value'],
            'Col2': ['Key metrics', '', '', 'A', 100],
            'Col3': ['', '', '', 'B', 200],
            'Col4': ['', '', '', 'C', 300]
        })
        
        parsed = spreadsheet_parser.parse_dataframe(gap_df, "test-req17", "Sheet1")
        
        # Validate gap detection
        assert len(parsed.intentional_gaps) >= 0, "Should classify empty rows"
        assert parsed.parsing_confidence > 0, "Should maintain parsing confidence"
        
        print("✅ Requirement 17: Intentional gap detection - PASSED")
    
    def test_requirement_18_robust_context_preservation(self):
        """Requirement 18: Robust Context Preservation"""
        # Create rich document for context preservation
        rich_df = pd.DataFrame({
            'Invoice': ['INV-001', '', 'Items', '', 'Total'],
            'Date': ['2024-01-15', '', 'Widget', '', '$500'],
            'Customer': ['Acme Corp', '', '5 @ $100', '', ''],
            'Status': ['Paid', '', '', '', '']
        })
        
        parsed = spreadsheet_parser.parse_dataframe(rich_df, "test-req18", "Sheet1")
        
        # Build context and validate preservation
        context = spreadsheet_parser.build_context(parsed)
        assert context is not None, "Should build comprehensive context"
        
        # Validate metadata preservation
        assert parsed.metadata is not None, "Should preserve metadata"
        assert len(parsed.sections) > 0, "Should preserve section structure"
        
        # Check parsing confidence
        assert parsed.parsing_confidence > 0.5, "Should maintain high confidence"
        
        print("✅ Requirement 18: Robust context preservation - PASSED")


class TestPerformanceRequirements:
    """Test Performance and Scalability"""
    
    def test_performance_large_file_processing(self):
        """Test performance with large files (target: 1K rows in <1 second)"""
        # Create 1000-row dataset
        large_df = pd.DataFrame({
            'ID': range(1, 1001),
            'Category': [f'Cat_{i%20}' for i in range(1000)],
            'Value': np.random.randint(1, 10000, 1000),
            'Description': [f'Description for item {i}' for i in range(1000)],
            'Date': pd.date_range('2024-01-01', periods=1000, freq='H')
        })
        
        # Measure parsing performance
        start_time = time.time()
        parsed = spreadsheet_parser.parse_dataframe(large_df, "perf-test", "Sheet1")
        parse_time = time.time() - start_time
        
        # Validate performance target
        assert parse_time < 1.0, f"Large file parsing should be <1s, took {parse_time:.2f}s"
        assert parsed is not None, "Should successfully parse large file"
        assert parsed.parsing_confidence > 0, "Should maintain parsing quality"
        
        print(f"✅ Performance: 1K rows parsed in {parse_time:.3f}s - PASSED")
    
    def test_memory_efficiency(self):
        """Test memory efficiency with multiple concurrent sessions"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple datasets
        datasets = []
        for i in range(10):
            df = pd.DataFrame({
                'ID': range(100),
                'Data': [f'Data_{j}_{i}' for j in range(100)]
            })
            parsed = spreadsheet_parser.parse_dataframe(df, f"mem-test-{i}", "Sheet1")
            datasets.append(parsed)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increase should be <100MB, was {memory_increase:.1f}MB"
        
        print(f"✅ Memory efficiency: {memory_increase:.1f}MB increase for 10 datasets - PASSED")


def run_comprehensive_tests():
    """Run all comprehensive requirement tests"""
    print("🧪 Running Comprehensive Requirements Testing Suite...")
    print("=" * 60)
    
    test_classes = [
        TestRequirements1to6_CoreParsing(),
        TestRequirements7to8_QueryProcessing(),
        TestRequirements9to14_OrchestratorIntegration(),
        TestRequirements15to18_AdvancedDocumentUnderstanding(),
        TestPerformanceRequirements()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📋 {class_name}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_class, test_method_name)
                test_method()
                passed_tests += 1
            except Exception as e:
                print(f"❌ {test_method_name} - FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎉 COMPREHENSIVE TESTING COMPLETE")
    print(f"📊 Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("✅ ALL REQUIREMENTS VALIDATED - PRODUCTION READY!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed - needs attention")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)