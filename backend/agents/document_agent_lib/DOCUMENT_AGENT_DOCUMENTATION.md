# Document Agent - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Component Details](#component-details)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Data Flows](#data-flows)
9. [Security & Best Practices](#security--best-practices)
10. [Performance & Scalability](#performance--scalability)
11. [Reliability & Error Handling](#reliability--error-handling)
12. [Troubleshooting](#troubleshooting)
13. [Testing](#testing)
14. [Cloud Deployment](#cloud-deployment)

---

## Overview

The Document Agent is a cloud-optimized system for intelligent document processing combining:
- **LLM-based Planning** for natural language editing instructions
- **RAG Integration** for document-based Q&A
- **Version Control** with full undo/redo capability
- **Session Management** for multi-user environments
- **Format Support** for DOCX, PDF, and TXT files
- **Cloud-First Design** with minimal memory footprint

**Key Features:**
- ✅ Natural Language Document Editing (e.g., "Make all headers bold")
- ✅ Document Analysis with RAG and FAISS vector stores
- ✅ Full Version History with undo/redo
- ✅ Multi-user Session Management
- ✅ Structured Data Extraction (tables, key points, summaries)
- ✅ Canvas-based Display for Web UI
- ✅ Thread-safe Operations for concurrent access
- ✅ Cloud-optimized with lazy loading and cleanup

---

## Quick Start

### Prerequisites
```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r backend/requirements.txt

# Required environment variables (.env file)
CEREBRAS_API_KEY=csk_your_key_here
NVIDIA_API_KEY=nvidia_your_key_here  # Optional
GROQ_API_KEY=gsk_your_key_here       # Optional
```

### Launch the Agent Server
```bash
# Development mode
cd backend/agents/document_agent
python -m uvicorn __init__:app --reload --port 8081

# Production mode
uvicorn __init__:app --host 0.0.0.0 --port 8081 --workers 4
```

### Run a Simple Task
```bash
# Terminal 1: Start server
python -m uvicorn backend.agents.document_agent:app --port 8081

# Terminal 2: Create a document
curl -X POST http://localhost:8081/create \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Introduction\nThis is a document.\n\nConclusion\nThank you.",
    "file_name": "my_document.docx",
    "file_type": "docx",
    "output_dir": "storage/document_agent"
  }'
```

### Expected Response
```json
{
  "success": true,
  "message": "Created my_document.docx",
  "file_path": "storage/document_agent/my_document.docx"
}
```

---

## Architecture

### System Diagram
```
┌─────────────────────────────────────────────────────┐
│  FastAPI Server (__init__.py)                      │
│  • POST /create    (create new documents)          │
│  • POST /edit      (edit with natural language)    │
│  • POST /analyze   (RAG-based Q&A)                 │
│  • POST /extract   (data extraction)               │
│  • POST /undo-redo (version control)               │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
    ┌───▼─────┐          ┌───────▼──┐
    │   LLM   │          │  Agent   │ (main orchestrator)
    │ Client  │          │(coordinates
    │(Cerebras│          │ components)
    │/NVIDIA) │          └────┬─────┘
    └─────────┘               │
        ┌─────────┬───────────┼──────────┬──────────┐
        │         │           │          │          │
    ┌───▼──┐  ┌──▼───┐   ┌───▼──┐  ┌──▼──┐  ┌─────▼─┐
    │Editor│  │Session│   │Version│  │Utils│  │Schemas│
    │      │  │Manager│   │Manager│  │     │  │       │
    └──────┘  └───────┘   └───────┘  └─────┘  └───────┘
        │
    ┌───▼──────────────────────────────┐
    │   File System                     │
    │   • Documents (DOCX, PDF, TXT)   │
    │   • Sessions & History            │
    │   • Versions & Backups            │
    └──────────────────────────────────┘
```

### Component Overview

| Module | Responsibility | Key Classes |
|--------|-----------------|-------------|
| **__init__.py** | FastAPI server and endpoints | FastAPI app, route handlers |
| **agent.py** | Orchestration and workflow | DocumentAgent (main coordinator) |
| **editors.py** | Document editing operations | DocumentEditor (unified editor) |
| **llm.py** | LLM planning and analysis | DocumentLLMClient (multi-provider) |
| **state.py** | Sessions and version control | DocumentSessionManager, DocumentVersionManager |
| **schemas.py** | API contracts | Pydantic models for all endpoints |
| **utils.py** | Helper functions | Conversion, extraction, display utilities |

---

## Component Details

### 1. agent.py — DocumentAgent (Orchestrator)
**Purpose:** Coordinates all components and manages complete document workflow.

**Key Methods:**
```python
DocumentAgent()                                 # Initialize all components
.analyze_document(request)                      # RAG-based analysis
.display_document(file_path)                    # Canvas display
.create_document(request)                       # New document creation
.edit_document(request)                         # NL-based editing
.undo_redo(request)                            # Version control
.extract_data(request)                         # Data extraction
.cleanup_old_versions(file_path, keep_count)   # Cloud optimization
```

### 2. editors.py — DocumentEditor
**Purpose:** Unified document editing with intelligent and advanced capabilities.

**Key Methods:**
```python
DocumentEditor(file_path)                   # Initialize with file
.analyze_state()                           # Analyze document structure
.add_paragraph(text, style)                # Add paragraph
.add_heading(text, level)                  # Add heading
.add_table(rows, cols)                     # Add table
.add_image(image_path, width)              # Add image
.format_text(text, bold, italic, color)    # Format text
.replace_text(old_text, new_text)          # Replace content
.delete_paragraph(index)                   # Delete paragraph
.set_paragraph_style(index, style)         # Change style
.save()                                    # Save to disk
```

### 3. llm.py — DocumentLLMClient
**Purpose:** LLM-based planning with provider fallback (Cerebras → NVIDIA → Groq).

**Key Methods:**
```python
DocumentLLMClient()                                  # Auto-detect provider
.interpret_edit_instruction(instruction, content, structure)   # Plan edits
.analyze_document_with_query(content, query)       # RAG analysis
.extract_structured_data(content, type)            # Extract data
```

### 4. state.py — Session & Version Management
**Purpose:** Stateful session tracking and version control.

**Key Classes:**
```python
DocumentSessionManager()                           # Session management
.get_or_create_session(path, name, thread_id)     # Get/create session
.add_edit_action(session_id, action)              # Track edits
.get_session_history(session_id)                  # Get history

DocumentVersionManager()                           # Version control
.save_version(file_path, description)             # Save version
.restore_version(file_path, version_id)           # Restore version
.get_versions(file_path)                          # List versions
.cleanup_old_versions(file_path, keep_count)      # Cloud cleanup
```

### 5. schemas.py — API Contracts
**Purpose:** Pydantic models for type-safe API contracts.

**Key Models:**
```python
AnalyzeDocumentRequest / AnalyzeDocumentResponse
CreateDocumentRequest / CreateDocumentResponse
EditDocumentRequest / EditDocumentResponse
ExtractDataRequest / ExtractDataResponse
UndoRedoRequest / UndoRedoResponse
VersionHistoryRequest / VersionHistoryResponse
```

### 6. utils.py — Helper Functions
**Purpose:** Document conversion, extraction, and display utilities.

**Key Functions:**
```python
extract_document_content(file_path)           # Extract text from docs
create_docx(content, file_path)              # Create Word document
create_pdf(content, file_path)               # Create PDF
convert_docx_to_pdf(docx_path, pdf_path)    # Convert for display
analyze_document_structure(file_path)        # Get doc structure
get_file_base64(file_path)                   # Encode for display
create_canvas_display(type, data)            # Create frontend display
```

---

## Configuration

### Environment Variables
```bash
# LLM API Keys
CEREBRAS_API_KEY=csk_xxxxxxxxxxxx           # Primary LLM
NVIDIA_API_KEY=nvidia_xxxxxxxxxxxx          # Fallback LLM
GROQ_API_KEY=gsk_xxxxxxxxxxxx              # Last resort LLM

# Storage Paths (optional, auto-created)
DOCUMENTS_DIR=storage/document_agent
SESSIONS_DIR=backend/storage/document_sessions
VERSIONS_DIR=backend/storage/document_versions

# Agent Settings (optional)
KEEP_VERSIONS=10                            # Versions per document (cloud cleanup)
LAZY_LOAD_LLM=true                         # Load LLM on first request
```

### File Storage Structure
```
backend/storage/
├── documents/                  # User documents
│   ├── my_document.docx
│   └── report.pdf
├── document_sessions/          # Session state
│   └── hash_sessionid.json
└── document_versions/          # Version history
    └── normalized_path/
        ├── v1234567890/
        │   ├── document.docx
        │   └── metadata.json
        └── v1234567900/
            └── ...
```

---

## API Reference

### POST /health
Check agent health and status.

**Response:**
```json
{
  "status": "healthy",
  "service": "document-agent",
  "version": "2.0.0"
}
```

### POST /create
Create a new document.

**Request:**
```json
{
  "content": "Your document content here",
  "file_name": "report.docx",
  "file_type": "docx",
  "output_dir": "storage/document_agent",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Created report.docx",
  "file_path": "storage/document_agent/report.docx"
}
```

### POST /edit
Edit document with natural language instruction.

**Request:**
```json
{
  "file_path": "storage/document_agent/report.docx",
  "instruction": "Make all headings bold and blue, add a table of contents",
  "thread_id": "optional-thread-id",
  "use_vision": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Applied 3 edits",
  "file_path": "storage/document_agent/report.docx",
  "can_undo": true,
  "can_redo": false,
  "edit_summary": "Applied formatting and structure changes"
}
```

### POST /analyze
Analyze document with RAG and answer queries.

**Request:**
```json
{
  "vector_store_paths": ["path/to/faiss/index"],
  "query": "What are the main findings?",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "The main findings are...",
  "sources": ["path/to/faiss/index"]
}
```

### POST /extract
Extract structured data from document.

**Request:**
```json
{
  "file_path": "storage/document_agent/report.docx",
  "extraction_type": "structured",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Data extracted",
  "extracted_data": {
    "title": "Document Title",
    "sections": ["Intro", "Methods", "Results"],
    "key_points": ["Point 1", "Point 2"]
  },
  "data_format": "structured"
}
```

### POST /undo-redo
Undo or redo edits.

**Request:**
```json
{
  "file_path": "storage/document_agent/report.docx",
  "action": "undo",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Undo successful",
  "file_path": "storage/document_agent/report.docx",
  "can_undo": true,
  "can_redo": true
}
```

### POST /versions
Get version history for a document.

**Request:**
```json
{
  "file_path": "storage/document_agent/report.docx",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Version history retrieved",
  "versions": [
    {
      "version_id": "v1234567890",
      "timestamp": 1234567890.0,
      "description": "Initial creation",
      "file_path": "backend/storage/document_versions/..."
    }
  ],
  "current_version": 0
}
```

### POST /display
Display document with canvas rendering.

**Request:**
```json
{
  "file_path": "storage/document_agent/report.docx",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Document displayed",
  "canvas_display": {
    "canvas_type": "pdf",
    "pdf_data": "data:application/pdf;base64,...",
    "file_path": "...",
    "original_type": "docx"
  },
  "file_type": ".docx"
}
```

---

## Usage Examples

### Example 1: Create and Edit a Document
```bash
# Create document
curl -X POST http://localhost:8081/create \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Introduction\nThis is a test.\n\nConclusion",
    "file_name": "test.docx",
    "file_type": "docx"
  }'

# Edit document (natural language)
curl -X POST http://localhost:8081/edit \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "storage/document_agent/test.docx",
    "instruction": "Make the introduction section bold. Add a section header called Methods"
  }'
```

### Example 2: Extract Data from Document
```bash
curl -X POST http://localhost:8081/extract \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "storage/document_agent/report.docx",
    "extraction_type": "structured"
  }'
```

### Example 3: Version Control
```bash
# Get version history
curl -X POST http://localhost:8081/versions \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "storage/document_agent/report.docx"
  }'

# Undo last edit
curl -X POST http://localhost:8081/undo-redo \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "storage/document_agent/report.docx",
    "action": "undo"
  }'
```

### Example 4: Python Integration
```python
import httpx

client = httpx.AsyncClient(base_url="http://localhost:8081")

# Create document
response = await client.post("/create", json={
    "content": "My document content",
    "file_name": "doc.docx",
    "file_type": "docx"
})

# Edit with NL instruction
response = await client.post("/edit", json={
    "file_path": "storage/document_agent/doc.docx",
    "instruction": "Make all headings blue and underlined"
})

# Get version history
response = await client.post("/versions", json={
    "file_path": "storage/document_agent/doc.docx"
})
```

---

## Data Flows

### High-Level Editing Flow

```
User NL Instruction
      ↓
[LLM Planning] → Structure edits as actions
      ↓
[DocumentEditor] → Execute each action (add, format, replace)
      ↓
[Save to Disk] → Write document
      ↓
[Version Manager] → Create snapshot
      ↓
[Session Manager] → Log action in history
      ↓
Result to User → Success + undo/redo state
```

### Data Transformations

1. **Instruction → Actions**
   - Input: "Make headings bold and add intro"
   - Output: [{"type": "format_text", ...}, {"type": "add_paragraph", ...}]

2. **Action → Edit Result**
   - Input: {"type": "add_paragraph", "text": "...", "style": "Normal"}
   - Output: "✓ Added paragraph" / "✗ Failed: ..."

3. **Document → Display**
   - DOCX → PDF (via convert_docx_to_pdf)
   - PDF → Base64 (via get_file_base64)
   - Base64 → Canvas data (for browser rendering)

---

## Security & Best Practices

### 1. API Key Management
```bash
# ✅ GOOD: Environment variables
export CEREBRAS_API_KEY="csk_..."
export NVIDIA_API_KEY="nvidia_..."

# ❌ BAD: Hardcoded in code
API_KEY = "secret_key_123"
```

### 2. File Path Validation
```python
# ✅ GOOD: Validate paths exist
if not Path(file_path).exists():
    raise FileNotFoundError(...)

# ❌ BAD: Trust user input
doc = Document(user_provided_path)
```

### 3. Thread Safety
```python
# ✅ GOOD: Lock for concurrent access
with self._session_lock:
    session = self._active_sessions[session_id]

# ❌ BAD: Race conditions
session = self._active_sessions[session_id]
```

### 4. Session Isolation
```python
# ✅ GOOD: Isolate by thread_id
session_id = self._get_session_id(path, thread_id)

# ❌ BAD: Share sessions across threads
session_id = self._get_session_id(path)
```

### 5. Error Handling
```python
# ✅ GOOD: Graceful error response
try:
    result = llm.invoke(prompt)
    return {"success": True, "data": result}
except Exception as e:
    logger.error(f"Error: {e}")
    return {"success": False, "error": str(e)}

# ❌ BAD: Crash without recovery
result = llm.invoke(prompt)  # May crash
return {"data": result}
```

### 6. Sensitive Data
- Never log API keys or credentials
- Redact user content in error messages
- Use secure file permissions on storage directories

---

## Performance & Scalability

### Cloud Optimizations

**Memory Efficiency:**
- Lazy load LLM on first request (not on startup)
- Limit document content to 5000-10000 chars for LLM
- Stream large files instead of loading entirely
- Auto-cleanup old versions (keep only 10 recent)

**Storage Efficiency:**
| Operation | Size | Strategy |
|-----------|------|----------|
| Document content | Variable | Limit to 10MB per file |
| Version storage | 2-5MB per version | Keep only 10 versions |
| Session data | <100KB | Auto-expire old sessions |
| Cache | Unlimited | Cleanup 72+ hour old files |

**Concurrent Access:**
- Thread-safe with locks on session/version access
- Supports multiple simultaneous edits
- Session isolation prevents cross-talk

### Performance Tips
1. **Limit version retention** → `cleanup_old_versions(keep_count=10)`
2. **Use headless DOCX viewing** → Skip PDF conversion for display
3. **Cache LLM responses** → Reuse analysis for repeated queries
4. **Batch operations** → Multiple edits in single request when possible

---

## Reliability & Error Handling

### LLM Provider Fallback
```
Request → CEREBRAS (try first)
          ↓ (if fails)
          NVIDIA (secondary)
          ↓ (if fails)
          GROQ (last resort)
          ↓ (if all fail)
          Error response
```

### Retry Strategy
- **LLM Calls**: 3 attempts with exponential backoff (1s, 3s, 5s)
- **File Operations**: 2 attempts (immediate retry)
- **Session Access**: Automatic retry with lock

### Error Categories
| Error | Handling | Recovery |
|-------|----------|----------|
| File not found | HTTP 404 | User provides correct path |
| LLM timeout | Fall back to next provider | Auto-attempt all providers |
| Disk full | HTTP 507 | Cleanup old versions |
| Invalid instruction | HTTP 400 | User clarifies instruction |
| Session conflict | Automatic locking | Resolve contention |

---

## Troubleshooting

### LLM Provider not initialized
```bash
# Check environment variables
echo $CEREBRAS_API_KEY
echo $NVIDIA_API_KEY

# Ensure one is set
export CEREBRAS_API_KEY="csk_..."
python -m uvicorn backend.agents.document_agent:app
```

### Document editing fails
1. Check file path exists: `ls -la storage/document_agent/`
2. Check file is valid DOCX/PDF/TXT
3. Check disk space: `df -h`
4. Try simpler instruction first

### Version restoration fails
1. Check version exists: `GET /versions`
2. Check version file not corrupted: `backend/storage/document_versions/`
3. Try creating new version first

### Out of memory
1. Reduce document content limit in llm.py
2. Cleanup old versions: `cleanup_old_versions(keep_count=5)`
3. Increase server memory or split documents

### Session issues
1. Clear sessions: `rm -rf backend/storage/document_sessions/*`
2. Verify thread_id is consistent
3. Check session file permissions

---

## Testing

### Test Categories

**Unit Tests:**
- DocumentEditor methods (add, format, replace)
- LLMClient interpretation accuracy
- Version manager save/restore
- Session isolation

**Integration Tests:**
- Create → Edit → Save → Undo/Redo workflow
- Multi-user session isolation
- Provider fallback functionality
- Version history with cleanup

**Cloud Tests:**
- Memory usage with 100+ document edits
- Concurrent 10+ simultaneous requests
- Version cleanup with 1000+ old versions
- API response time under load

### High-Priority Test Cases
```python
# Test document creation and editing
def test_create_and_edit():
    agent = DocumentAgent()
    # Create doc
    result = agent.create_document(CreateDocumentRequest(...))
    assert result['success']
    # Edit doc
    result = agent.edit_document(EditDocumentRequest(...))
    assert result['success']
    assert result['can_undo']

# Test version control
def test_undo_redo():
    # ... create and edit ...
    # Undo
    result = agent.undo_redo(UndoRedoRequest(..., action='undo'))
    assert result['success']
    assert result['can_redo']
    # Redo
    result = agent.undo_redo(UndoRedoRequest(..., action='redo'))
    assert result['success']

# Test session isolation
def test_session_isolation():
    # Same doc, different threads
    session1 = manager.get_or_create_session(path, name, thread_id="1")
    session2 = manager.get_or_create_session(path, name, thread_id="2")
    assert session1.session_id != session2.session_id
```

### Running Tests
```bash
pip install pytest pytest-asyncio

# Run all tests
pytest tests/document_agent/

# With coverage
pytest tests/document_agent/ --cov=backend.agents.document_agent
```

---

## Cloud Deployment

### Docker Configuration
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY backend/agents/document_agent ./document_agent

# Health check
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8081/health

# Run
CMD ["uvicorn", "document_agent:app", "--host", "0.0.0.0", "--port", "8081"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: document-agent
  template:
    metadata:
      labels:
        app: document-agent
    spec:
      containers:
      - name: document-agent
        image: orbimesh/document-agent:latest
        ports:
        - containerPort: 8081
        env:
        - name: CEREBRAS_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-keys
              key: cerebras
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 10
```

### Environment Variables for Cloud
```bash
# API Keys (use secrets management)
CEREBRAS_API_KEY=<from-secret-store>
NVIDIA_API_KEY=<from-secret-store>

# Storage (use cloud storage)
DOCUMENTS_DIR=/mnt/cloud-storage/documents
SESSIONS_DIR=/mnt/cloud-storage/sessions
VERSIONS_DIR=/mnt/cloud-storage/versions

# Settings
KEEP_VERSIONS=5              # Cloud: keep fewer versions
LAZY_LOAD_LLM=true          # Cloud: load on demand
LOG_LEVEL=INFO              # Cloud: INFO/WARNING only
```

---

## Implementation Roadmap

### Current (v2.0)
- ✅ Document creation (DOCX, PDF, TXT)
- ✅ LLM-based natural language editing
- ✅ Full version control with undo/redo
- ✅ Session management for multi-user
- ✅ Data extraction (text, tables, structured)
- ✅ Canvas display rendering
- ✅ Cloud-optimized architecture

### Planned (v2.1)
- 🔲 FAISS vector store integration for full RAG
- 🔲 More LLM providers (Anthropic, OpenAI)
- 🔲 Document templates and auto-formatting
- 🔲 Collaborative editing with real-time sync
- 🔲 Advanced formatting (styles, themes, branding)

### Future (v3.0)
- 🔲 Document signing and certification
- 🔲 Workflow automation (approval chains)
- 🔲 Advanced analytics (document insights)
- 🔲 Multi-language support
- 🔲 Custom LLM fine-tuning

---

## Support & References

**Key Directories:**
- Configuration: `backend/agents/document_agent/`
- Storage: `backend/storage/documents/`, `sessions/`, `versions/`
- Tests: `backend/tests/document_agent/`

**External Resources:**
- Python-docx: https://python-docx.readthedocs.io/
- FastAPI: https://fastapi.tiangolo.com/
- Langchain: https://python.langchain.com/
- Cerebras API: https://docs.cerebras.ai/

**Common Tasks:**
- Check API health: `curl http://localhost:8081/health`
- View logs: `tail -f backend/storage/logs/document_agent.log`
- Cleanup old data: Manual version cleanup or `agent.cleanup_old_versions()`
