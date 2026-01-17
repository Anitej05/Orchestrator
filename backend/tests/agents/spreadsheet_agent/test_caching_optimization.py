"""
Tests for caching and optimization features (Task 11)

Tests Requirements 8.4, 8.5, 13.3, 13.4
"""

import pytest
import pandas as pd
from datetime import datetime

from backend.agents.spreadsheet_agent.dataframe_cache import DataFrameCache
from backend.agents.spreadsheet_agent.parse_cache import ParseCache
from backend.agents.spreadsheet_agent.parsing_models import (
    ParsedSpreadsheet,
    DocumentType,
    TableRegion,
    TableSchema,
    DocumentSection,
    SectionType,
    ContentType
)


class TestDataFrameCacheThreadSwitching:
    """Test thread context switching in DataFrameCache (Requirement 13.3)"""
    
    def test_switch_thread_context_success(self):
        """Test successful thread context switching"""
        cache = DataFrameCache()
        
        # Store data in thread1
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        cache.store('thread1', 'file1', df1, {'source': 'test1'})
        
        # Store data in thread2
        df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
        cache.store('thread2', 'file2', df2, {'source': 'test2'})
        
        # Switch from thread1 to thread2
        switched_df, metadata = cache.switch_thread_context('thread1', 'thread2', 'file2')
        
        assert switched_df is not None
        assert switched_df.equals(df2)
        assert metadata['source'] == 'test2'
    
    def test_switch_thread_context_nonexistent_thread(self):
        """Test switching to non-existent thread"""
        cache = DataFrameCache()
        
        # Store data in thread1
        df1 = pd.DataFrame({'A': [1, 2, 3]})
        cache.store('thread1', 'file1', df1)
        
        # Try to switch to non-existent thread
        switched_df, metadata = cache.switch_thread_context('thread1', 'thread_nonexistent')
        
        assert switched_df is None
        assert metadata == {}
    
    def test_switch_thread_context_most_recent(self):
        """Test switching without specifying file_id (should get most recent)"""
        cache = DataFrameCache()
        
        # Store multiple files in thread2
        df1 = pd.DataFrame({'A': [1, 2]})
        df2 = pd.DataFrame({'B': [3, 4]})
        
        cache.store('thread2', 'file1', df1)
        cache.store('thread2', 'file2', df2)  # This is more recent
        
        # Switch to thread2 without file_id
        switched_df, metadata = cache.switch_thread_context('thread1', 'thread2')
        
        assert switched_df is not None
        assert switched_df.equals(df2)  # Should get most recent
    
    def test_get_thread_context(self):
        """Test getting thread context information"""
        cache = DataFrameCache()
        
        # Store data
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        cache.store('thread1', 'file1', df, {'test': 'metadata'})
        
        # Get context
        context = cache.get_thread_context('thread1')
        
        assert context['exists'] is True
        assert context['thread_id'] == 'thread1'
        assert context['file_count'] == 1
        assert len(context['files']) == 1
        assert context['files'][0]['file_id'] == 'file1'
        assert context['files'][0]['shape'] == (3, 2)
        assert 'A' in context['files'][0]['columns']
        assert 'B' in context['files'][0]['columns']


class TestDataFrameCacheCrossThread:
    """Test cross-thread file access (Requirement 13.4)"""
    
    def test_retrieve_cross_thread_success(self):
        """Test retrieving file from another thread"""
        cache = DataFrameCache()
        
        # Store in thread1
        df = pd.DataFrame({'A': [1, 2, 3]})
        cache.store('thread1', 'file1', df, {'source': 'thread1'})
        
        # Retrieve from thread2 with cross-thread access
        retrieved_df, metadata = cache.retrieve('thread2', 'file1', allow_cross_thread=True)
        
        assert retrieved_df is not None
        assert retrieved_df.equals(df)
        assert metadata['source'] == 'thread1'
    
    def test_retrieve_cross_thread_disabled(self):
        """Test that cross-thread access is disabled by default"""
        cache = DataFrameCache()
        
        # Store in thread1
        df = pd.DataFrame({'A': [1, 2, 3]})
        cache.store('thread1', 'file1', df)
        
        # Try to retrieve from thread2 without cross-thread access
        retrieved_df, metadata = cache.retrieve('thread2', 'file1', allow_cross_thread=False)
        
        assert retrieved_df is None
        assert metadata == {}
    
    def test_retrieve_cross_thread_not_found(self):
        """Test cross-thread retrieval when file doesn't exist anywhere"""
        cache = DataFrameCache()
        
        # Store in thread1
        df = pd.DataFrame({'A': [1, 2, 3]})
        cache.store('thread1', 'file1', df)
        
        # Try to retrieve non-existent file with cross-thread access
        retrieved_df, metadata = cache.retrieve('thread2', 'file_nonexistent', allow_cross_thread=True)
        
        assert retrieved_df is None
        assert metadata == {}


class TestParseCacheBasic:
    """Test basic parse cache functionality (Requirement 8.5)"""
    
    def test_store_and_retrieve_parsed(self):
        """Test storing and retrieving parsed spreadsheet"""
        cache = ParseCache()
        
        # Create a simple parsed spreadsheet
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df,
            intentional_gaps=[]
        )
        
        # Store
        cache.store_parsed('thread1', 'file1', parsed)
        
        # Retrieve
        retrieved = cache.retrieve_parsed('thread1', 'file1')
        
        assert retrieved is not None
        assert retrieved.file_id == 'file1'
        assert retrieved.document_type == DocumentType.DATA_TABLE
    
    def test_cache_hit_tracking(self):
        """Test that cache tracks access counts"""
        cache = ParseCache()
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df
        )
        
        cache.store_parsed('thread1', 'file1', parsed)
        
        # Retrieve multiple times
        cache.retrieve_parsed('thread1', 'file1')
        cache.retrieve_parsed('thread1', 'file1')
        cache.retrieve_parsed('thread1', 'file1')
        
        # Check stats
        stats = cache.get_cache_stats('thread1')
        assert stats['file_count'] == 1
        assert stats['files'][0]['access_count'] == 3


class TestParseCacheContextCaching:
    """Test context caching (Requirement 8.4)"""
    
    def test_store_and_retrieve_context(self):
        """Test storing and retrieving generated contexts"""
        cache = ParseCache()
        
        # Create parsed spreadsheet
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df
        )
        
        cache.store_parsed('thread1', 'file1', parsed)
        
        # Store contexts
        cache.store_context('thread1', 'file1', 'structured', {'test': 'data'}, max_tokens=8000)
        cache.store_context('thread1', 'file1', 'compact', 'compact_string', max_tokens=8000)
        
        # Retrieve contexts
        structured = cache.retrieve_context('thread1', 'file1', 'structured', max_tokens=8000)
        compact = cache.retrieve_context('thread1', 'file1', 'compact', max_tokens=8000)
        
        assert structured == {'test': 'data'}
        assert compact == 'compact_string'
    
    def test_context_cache_key_with_tokens(self):
        """Test that different token limits create different cache keys"""
        cache = ParseCache()
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df
        )
        
        cache.store_parsed('thread1', 'file1', parsed)
        
        # Store contexts with different token limits
        cache.store_context('thread1', 'file1', 'structured', {'data': '8000'}, max_tokens=8000)
        cache.store_context('thread1', 'file1', 'structured', {'data': '4000'}, max_tokens=4000)
        
        # Retrieve with specific token limits
        context_8000 = cache.retrieve_context('thread1', 'file1', 'structured', max_tokens=8000)
        context_4000 = cache.retrieve_context('thread1', 'file1', 'structured', max_tokens=4000)
        
        assert context_8000 == {'data': '8000'}
        assert context_4000 == {'data': '4000'}


class TestParseCacheThreadSwitching:
    """Test thread context switching in ParseCache (Requirement 13.3)"""
    
    def test_switch_thread_context_success(self):
        """Test successful thread context switching"""
        cache = ParseCache()
        
        # Store in thread1
        df1 = pd.DataFrame({'A': [1, 2, 3]})
        parsed1 = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df1
        )
        cache.store_parsed('thread1', 'file1', parsed1)
        
        # Store in thread2
        df2 = pd.DataFrame({'B': [4, 5, 6]})
        parsed2 = ParsedSpreadsheet(
            file_id='file2',
            sheet_name='Sheet2',
            document_type=DocumentType.INVOICE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df2
        )
        cache.store_parsed('thread2', 'file2', parsed2)
        
        # Switch from thread1 to thread2
        switched = cache.switch_thread_context('thread1', 'thread2', 'file2')
        
        assert switched is not None
        assert switched.file_id == 'file2'
        assert switched.document_type == DocumentType.INVOICE
    
    def test_get_thread_context(self):
        """Test getting thread context information"""
        cache = ParseCache()
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df
        )
        
        cache.store_parsed('thread1', 'file1', parsed)
        cache.store_context('thread1', 'file1', 'structured', {'test': 'data'}, max_tokens=8000)
        
        # Get context
        context = cache.get_thread_context('thread1')
        
        assert context['exists'] is True
        assert context['thread_id'] == 'thread1'
        assert context['file_count'] == 1
        assert len(context['files']) == 1
        assert context['files'][0]['file_id'] == 'file1'
        assert context['files'][0]['document_type'] == 'data_table'
        assert 'structured_8000' in context['files'][0]['cached_contexts']


class TestParseCacheCrossThread:
    """Test cross-thread access in ParseCache (Requirement 13.4)"""
    
    def test_retrieve_parsed_cross_thread(self):
        """Test retrieving parsed result from another thread"""
        cache = ParseCache()
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df
        )
        
        cache.store_parsed('thread1', 'file1', parsed)
        
        # Retrieve from thread2 with cross-thread access
        retrieved = cache.retrieve_parsed('thread2', 'file1', allow_cross_thread=True)
        
        assert retrieved is not None
        assert retrieved.file_id == 'file1'
    
    def test_retrieve_context_cross_thread(self):
        """Test retrieving context from another thread"""
        cache = ParseCache()
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df
        )
        
        cache.store_parsed('thread1', 'file1', parsed)
        cache.store_context('thread1', 'file1', 'structured', {'test': 'data'}, max_tokens=8000)
        
        # Retrieve from thread2 with cross-thread access
        context = cache.retrieve_context('thread2', 'file1', 'structured', max_tokens=8000, allow_cross_thread=True)
        
        assert context is not None
        assert context == {'test': 'data'}


class TestCacheInvalidation:
    """Test cache invalidation"""
    
    def test_invalidate_file(self):
        """Test invalidating cache for a specific file"""
        cache = ParseCache()
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df
        )
        
        cache.store_parsed('thread1', 'file1', parsed)
        cache.store_context('thread1', 'file1', 'structured', {'test': 'data'})
        
        # Verify it's cached
        assert cache.has_parsed('thread1', 'file1')
        
        # Invalidate
        result = cache.invalidate_file('thread1', 'file1')
        
        assert result is True
        assert not cache.has_parsed('thread1', 'file1')
    
    def test_clear_thread(self):
        """Test clearing all cache for a thread"""
        cache = ParseCache()
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        parsed = ParsedSpreadsheet(
            file_id='file1',
            sheet_name='Sheet1',
            document_type=DocumentType.DATA_TABLE,
            metadata={},
            sections=[],
            tables=[],
            raw_df=df
        )
        
        cache.store_parsed('thread1', 'file1', parsed)
        cache.store_parsed('thread1', 'file2', parsed)
        
        # Verify files are cached
        assert cache.has_parsed('thread1', 'file1')
        assert cache.has_parsed('thread1', 'file2')
        
        # Clear thread
        cache.clear_thread('thread1')
        
        assert not cache.has_parsed('thread1', 'file1')
        assert not cache.has_parsed('thread1', 'file2')
