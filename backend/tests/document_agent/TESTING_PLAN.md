# Document Agent - Comprehensive Unit Testing Plan

## Overview
Complete unit test coverage for the Document Analysis Agent, covering all endpoints, file formats, error handling, and edge cases.

## Test Structure

```
document_agent/
├── test_file_processor.py          # Existing tests (expand)
├── test_core_functionality.py       # NEW: Core agent initialization & config
├── test_upload_endpoint.py          # NEW: /upload endpoint tests
├── test_analyze_endpoint.py         # NEW: /analyze endpoint tests
├── test_display_endpoint.py         # NEW: /display endpoint tests
├── test_create_endpoint.py          # NEW: /create endpoint tests
├── test_edit_endpoint.py            # NEW: /edit endpoint tests
├── test_undo_redo_endpoint.py       # NEW: /undo & /redo endpoint tests
├── test_history_endpoint.py         # NEW: /history endpoint tests
├── test_error_handling.py           # NEW: Error scenarios & edge cases
├── test_concurrent_operations.py    # NEW: Concurrency & session management
└── conftest.py                      # NEW: Shared fixtures & setup
```

---

## 1. Unit Tests by Endpoint

### 1.1 Core Functionality Tests (`test_core_functionality.py`)

**Purpose:** Test agent initialization, configuration, and state management

**Test Cases:**
```python
def test_agent_initialization():
    """Verify agent initializes with correct config"""
    - Check agent is instantiated
    - Verify LLM chain is configured
    - Confirm document storage is ready

def test_agent_configuration():
    """Test configuration loading"""
    - Load from environment variables
    - Load from config files
    - Verify timeout settings

def test_session_management():
    """Test session lifecycle"""
    - Create new session
    - Retrieve session by ID
    - Clear session data
    - Handle stale sessions

def test_document_storage_initialization():
    """Test file storage system"""
    - Create storage directory
    - Verify permissions
    - Test cleanup of temp files
```

---

### 1.2 Upload Endpoint Tests (`test_upload_endpoint.py`)

**Purpose:** Test file upload, validation, and storage

**File Formats to Test:**
- ✅ PDF (simple, complex layouts, scanned)
- ✅ DOCX (simple, with tables, with images)
- ✅ Images (PNG, JPG, TIFF, high-res)
- ✅ TXT
- ❌ Unsupported formats (XLS, ZIP, etc.)

**Test Cases:**
```python
class TestUploadEndpoint:
    def test_upload_valid_pdf():
        """Upload PDF and verify metadata"""
        - File stored correctly
        - Metadata extracted (title, pages, size)
        - File ID returned
        
    def test_upload_valid_docx():
        """Upload DOCX and verify storage"""
        - File parsed correctly
        - Images in DOCX extracted
        - File ID returned
        
    def test_upload_valid_image():
        """Upload image file"""
        - Image stored
        - Dimensions extracted
        - OCR readiness verified
        
    def test_upload_multiple_files_same_session():
        """Upload multiple files to same session"""
        - Each file gets unique ID
        - Files don't overwrite each other
        - Session tracks all files
        
    def test_upload_file_too_large():
        """Reject files exceeding size limit"""
        - Returns 413 error
        - Appropriate error message
        - No partial file stored
        
    def test_upload_unsupported_format():
        """Reject unsupported file types"""
        - Returns 400 error
        - Lists supported formats
        - No corrupted file stored
        
    def test_upload_corrupted_file():
        """Handle corrupted/invalid files"""
        - Detects corruption
        - Returns meaningful error
        - Cleanup performed
        
    def test_upload_concurrent_files():
        """Handle concurrent uploads"""
        - Multiple files upload simultaneously
        - All stored correctly
        - No race conditions
```

---

### 1.3 Analyze Endpoint Tests (`test_analyze_endpoint.py`)

**Purpose:** Test document content extraction and analysis

**Test Cases:**
```python
class TestAnalyzeEndpoint:
    def test_extract_text_from_pdf():
        """Extract text from PDF"""
        - Full text extracted
        - Formatting preserved where relevant
        - Handles multiple pages
        - Works with scanned PDFs (OCR)
        
    def test_extract_text_from_docx():
        """Extract text from DOCX"""
        - Paragraphs extracted
        - Formatting tags preserved
        - Handles nested structures
        
    def test_extract_tables_from_pdf():
        """Extract tables from PDF"""
        - Table detected
        - Rows/columns parsed correctly
        - Data types recognized
        - Nested tables handled
        
    def test_extract_tables_from_docx():
        """Extract tables from DOCX"""
        - Table structure preserved
        - Cell content accurate
        - Merged cells handled
        - Empty cells tracked
        
    def test_extract_metadata():
        """Extract document metadata"""
        - Title extracted
        - Author information
        - Creation/modification dates
        - Page count
        - Language detection
        
    def test_extract_images_metadata():
        """Extract image metadata"""
        - Image count
        - Image dimensions
        - Image file paths
        - Captions detected
        
    def test_analyze_with_custom_options():
        """Analyze with specific extraction options"""
        - Include/exclude images
        - Include/exclude tables
        - Language-specific processing
        - Formatting level control
        
    def test_analyze_empty_document():
        """Handle empty/blank documents"""
        - Returns empty result gracefully
        - No errors thrown
        - Session remains valid
        
    def test_analyze_performance_large_document():
        """Analyze large documents within timeout"""
        - 100+ page PDF analyzed
        - Results returned within timeout
        - Memory usage reasonable
```

---

### 1.4 Display Endpoint Tests (`test_display_endpoint.py`)

**Purpose:** Test canvas rendering and preview generation

**Test Cases:**
```python
class TestDisplayEndpoint:
    def test_display_pdf_preview():
        """Generate PDF preview"""
        - First page rendered to canvas
        - Image format correct (base64 or URL)
        - Zoom level configurable
        
    def test_display_docx_preview():
        """Render DOCX preview"""
        - Document layout displayed
        - Text formatted correctly
        - Images embedded
        
    def test_display_specific_page():
        """Display specific page number"""
        - Correct page rendered
        - Page exists validation
        - Navigation metadata provided
        
    def test_display_extracted_content():
        """Display previously extracted content"""
        - Formatted output shown
        - Tables rendered as tables
        - Text properly escaped
        
    def test_display_with_annotations():
        """Display with highlighting/annotations"""
        - Highlights rendered correctly
        - Annotation text visible
        - Multiple annotations supported
        
    def test_display_performance():
        """Canvas rendering completes quickly"""
        - Preview generated <2s
        - Memory efficient
        - No memory leaks on repeated calls
```

---

### 1.5 Create Endpoint Tests (`test_create_endpoint.py`)

**Purpose:** Test document generation from prompts

**Test Cases:**
```python
class TestCreateEndpoint:
    def test_create_document_from_prompt():
        """Generate document from text prompt"""
        - Document created
        - Content matches intent
        - Format is valid
        
    def test_create_with_template():
        """Generate document using template"""
        - Template loaded
        - Variables substituted
        - Output matches template structure
        
    def test_create_document_format_options():
        """Create in different formats"""
        - PDF generation
        - DOCX generation
        - HTML generation
        - Plain text generation
        
    def test_create_document_with_formatting():
        """Create formatted document"""
        - Headings applied
        - Lists generated
        - Tables created
        - Images embedded if provided
        
    def test_create_document_length_control():
        """Control document length"""
        - Short summary (1-2 pages)
        - Medium document (5-10 pages)
        - Long document (20+ pages)
        
    def test_create_empty_prompt():
        """Handle empty/invalid prompt"""
        - Validation error returned
        - Helpful error message
        - No partial document created
        
    def test_create_document_token_limit():
        """Respect LLM token limits"""
        - Long prompt truncated appropriately
        - Warning issued if truncated
        - Content remains coherent
```

---

### 1.6 Edit Endpoint Tests (`test_edit_endpoint.py`)

**Purpose:** Test document modification operations

**Edit Operations:**
1. Insert text at position
2. Replace text range
3. Delete text range
4. Insert table
5. Modify table cell
6. Insert image
7. Modify heading
8. Modify list

**Test Cases:**
```python
class TestEditEndpoint:
    def test_insert_text():
        """Insert text at specified position"""
        - Text inserted at correct location
        - Document structure preserved
        - Content before/after unchanged
        
    def test_replace_text_range():
        """Replace text in range"""
        - Old text removed
        - New text inserted
        - Surrounding content intact
        - Formatting preserved where relevant
        
    def test_delete_text_range():
        """Delete text range"""
        - Text removed
        - No orphaned formatting
        - Document remains valid
        
    def test_insert_table():
        """Insert table at position"""
        - Table structure created
        - Rows/columns as specified
        - Empty cells initialized
        - Numbering updated if needed
        
    def test_modify_table_cell():
        """Modify specific table cell"""
        - Cell content updated
        - Table structure preserved
        - Other cells unchanged
        
    def test_insert_image():
        """Insert image at position"""
        - Image file embedded
        - Dimensions set correctly
        - Caption added if provided
        - File reference valid
        
    def test_modify_heading():
        """Modify heading level/text"""
        - Heading text changed
        - Level changed
        - TOC updated if applicable
        
    def test_modify_list():
        """Modify list items"""
        - Add item to list
        - Remove list item
        - Change list type (bullet/numbered)
        - Preserve nesting
        
    def test_edit_invalid_position():
        """Handle invalid edit positions"""
        - Out of bounds caught
        - Error returned
        - Document unchanged
        
    def test_concurrent_edits():
        """Handle concurrent edits to same document"""
        - Edits applied sequentially
        - No data loss
        - Final state consistent
        
    def test_edit_persists_to_storage():
        """Verify edits are saved"""
        - Edit saved to file
        - Reloading shows changes
        - No data loss on server restart
```

---

### 1.7 Undo/Redo Endpoint Tests (`test_undo_redo_endpoint.py`)

**Purpose:** Test undo/redo stack management

**Test Cases:**
```python
class TestUndoRedoEndpoint:
    def test_undo_single_operation():
        """Undo last operation"""
        - Document reverted to previous state
        - Redo available
        - History updated
        
    def test_undo_multiple_operations():
        """Undo multiple operations"""
        - Each undo reverts one operation
        - Stack traversed correctly
        - Can undo all the way to start
        
    def test_undo_at_start():
        """Cannot undo before first operation"""
        - Returns appropriate message
        - No error
        - Document unchanged
        
    def test_redo_after_undo():
        """Redo after undo"""
        - Document state restored
        - Redo stack cleared on new edit
        
    def test_undo_redo_state_consistency():
        """Verify state consistency through undo/redo"""
        - Undo then redo equals original
        - Multiple undo/redo cycles work
        - No state corruption
        
    def test_undo_stack_limit():
        """Respect undo stack size limit"""
        - Only last N operations tracked
        - Oldest operations dropped
        - Memory usage bounded
        
    def test_undo_persists_across_calls():
        """Undo history persists in session"""
        - History survives API calls
        - History survives page reloads
        - History cleared with new document
```

---

### 1.8 History Endpoint Tests (`test_history_endpoint.py`)

**Purpose:** Test operation history and versioning

**Test Cases:**
```python
class TestHistoryEndpoint:
    def test_get_operation_history():
        """Retrieve full operation history"""
        - All operations listed
        - Chronological order
        - Operation details included
        
    def test_history_metadata():
        """Operation history includes metadata"""
        - Timestamp
        - Operation type
        - Parameters used
        - Result/outcome
        
    def test_history_filtering():
        """Filter history by operation type"""
        - Filter by insert/replace/delete
        - Filter by date range
        - Combined filters work
        
    def test_history_pagination():
        """Page through large histories"""
        - Limit/offset parameters work
        - Total count provided
        - Correct subset returned
        
    def test_get_version_at_timestamp():
        """Retrieve document state at specific time"""
        - Version state accurate
        - Can jump to any point in history
        - Performance acceptable
        
    def test_history_diff():
        """Get diff between versions"""
        - Changes highlighted
        - Additions/deletions shown
        - Diff format clear
        
    def test_history_branching():
        """Handle undo + new edit branching"""
        - Original branch preserved
        - Can switch between branches
        - Both branches stored separately
```

---

## 2. Error Handling Tests (`test_error_handling.py`)

**Purpose:** Test all error scenarios and edge cases

```python
class TestErrorHandling:
    def test_invalid_file_id():
        """Handle invalid/missing file IDs"""
        - Returns 404
        - Helpful error message
        
    def test_invalid_session_id():
        """Handle invalid session"""
        - Returns 401 or appropriate error
        - Session not created
        
    def test_authentication_failure():
        """Test auth failures"""
        - Missing auth token
        - Invalid token
        - Expired token
        - Insufficient permissions
        
    def test_file_deleted_during_processing():
        """Handle file deleted mid-operation"""
        - Graceful error
        - No partial results
        - Session recoverable
        
    def test_storage_permission_errors():
        """Handle storage access issues"""
        - Read permission denied
        - Write permission denied
        - Disk full
        - Meaningful error messages
        
    def test_llm_api_timeout():
        """Handle LLM timeouts"""
        - Timeout respected
        - Error returned quickly
        - Retry logic if configured
        - No hanging requests
        
    def test_llm_api_rate_limit():
        """Handle rate limiting"""
        - Rate limit error caught
        - Backoff implemented
        - User informed
        
    def test_llm_api_failure():
        """Handle LLM unavailability"""
        - Service down handled
        - Graceful degradation if possible
        - User informed
        
    def test_concurrent_session_access():
        """Handle concurrent access to same session"""
        - Race conditions prevented
        - Last-write-wins or explicit handling
        - No data corruption
        
    def test_memory_leak_detection():
        """Detect memory leaks"""
        - Large file processing
        - Many operations in session
        - Cleanup verified
        - Memory returned after operation
        
    def test_input_injection_prevention():
        """Prevent injection attacks"""
        - File path traversal prevented
        - Prompt injection handled
        - Special characters escaped
```

---

## 3. Concurrency & Performance Tests (`test_concurrent_operations.py`)

**Purpose:** Test thread-safety and performance under load

```python
class TestConcurrency:
    def test_concurrent_uploads():
        """Multiple users upload simultaneously"""
        - All uploads succeed
        - Files don't collide
        - Session isolation verified
        
    def test_concurrent_edits_different_documents():
        """Edit different documents concurrently"""
        - No cross-contamination
        - All edits applied
        
    def test_concurrent_operations_same_document():
        """Handle concurrent ops on same document"""
        - Operations queued/sequenced
        - Final state consistent
        - No data loss
        
    def test_session_concurrency():
        """Multiple sessions exist concurrently"""
        - Isolation verified
        - Session IDs unique
        - Operations scoped correctly
        
    def test_load_performance():
        """Performance under load"""
        - Upload: 100 files/sec acceptable?
        - Analyze: Process X pages/sec?
        - Edit: X edits/sec?
        - Response times within SLA

class TestMemoryManagement:
    def test_large_document_processing():
        """Memory usage with large documents"""
        - 500+ page PDF processed
        - Memory grows linearly, not exponentially
        - Cleanup happens after processing
        
    def test_session_memory_cleanup():
        """Memory freed after session ends"""
        - Long-running session closed
        - Resources released
        - No memory leak
        
    def test_undo_stack_memory():
        """Undo stack doesn't consume excessive memory"""
        - 100+ operations
        - Memory usage bounded
        - Old operations dropped appropriately
```

---

## 4. Fixtures & Shared Setup (`conftest.py`)

```python
# conftest.py

@pytest.fixture
def sample_pdf():
    """Sample PDF file for testing"""
    return generate_test_pdf()

@pytest.fixture
def sample_docx():
    """Sample DOCX file for testing"""
    return generate_test_docx()

@pytest.fixture
def sample_image():
    """Sample image file for testing"""
    return generate_test_image()

@pytest.fixture
def document_agent():
    """Initialized document agent"""
    agent = DocumentAnalysisAgent()
    yield agent
    agent.cleanup()

@pytest.fixture
def test_session():
    """Test session with cleanup"""
    session = create_test_session()
    yield session
    cleanup_test_session(session)

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    return MockLanguageModel()

@pytest.fixture
def temp_storage():
    """Temporary storage for test files"""
    path = create_temp_directory()
    yield path
    cleanup_temp_directory(path)
```

---

## 5. Test Execution Strategy

### Phase 1: Core Tests (Week 1)
- ✅ Core functionality
- ✅ Upload endpoint
- ✅ Analyze endpoint

### Phase 2: Extended Tests (Week 2)
- ✅ Display endpoint
- ✅ Create endpoint
- ✅ Edit endpoint

### Phase 3: Advanced Tests (Week 3)
- ✅ Undo/Redo
- ✅ History
- ✅ Error handling

### Phase 4: Performance & Load (Week 4)
- ✅ Concurrency tests
- ✅ Performance benchmarks
- ✅ Memory profiling

---

## 6. Coverage Goals

| Component | Target Coverage | Status |
|-----------|-----------------|--------|
| Core agent | 95% | ⏳ |
| Upload | 100% | ⏳ |
| Analyze | 95% | ⏳ |
| Display | 90% | ⏳ |
| Create | 85% | ⏳ |
| Edit | 95% | ⏳ |
| Undo/Redo | 100% | ⏳ |
| History | 90% | ⏳ |
| Error handling | 100% | ⏳ |
| **Overall** | **93%** | ⏳ |

---

## 7. Test Data Requirements

**Fixtures needed:**
- ✅ Simple PDF (1 page, text only)
- ✅ Complex PDF (20 pages, tables, images)
- ✅ Scanned PDF (OCR needed)
- ✅ Simple DOCX (text + basic formatting)
- ✅ Complex DOCX (tables, nested lists, images)
- ✅ High-res image (PNG, JPG)
- ✅ Low-res image (handling degradation)
- ✅ Empty document
- ✅ Malformed files (test corruption handling)

---

## 8. Running the Tests

```bash
# Run all document agent tests
pytest backend/tests/document_agent/ -v

# Run specific test file
pytest backend/tests/document_agent/test_upload_endpoint.py -v

# Run with coverage
pytest backend/tests/document_agent/ --cov=agents.document_agent --cov-report=html

# Run performance tests only
pytest backend/tests/document_agent/test_concurrent_operations.py -v -m performance

# Watch mode (auto-run on file changes)
pytest-watch backend/tests/document_agent/
```

---

## 9. Continuous Integration

**GitHub Actions workflow:**
```yaml
- Run all tests on PR
- Generate coverage report
- Upload to Codecov
- Performance regression detection
- Block merge if coverage < 90%
```

---

## Success Criteria

✅ All tests pass
✅ Coverage > 90%
✅ No performance regressions
✅ All endpoints documented with test examples
✅ Integration tests demonstrate real workflows
