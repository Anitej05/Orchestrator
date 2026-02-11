"""
Document Agent Manual Test Suite (Phase 2 Refactor)

Async, registry-driven test harness for document agent with comprehensive metrics,
library detection, and ASCII-safe output. Mirrors test_spreadsheet_manual.py architecture.
"""

import asyncio
import logging
import json
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Get workspace root (project root)
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))

from backend.agents.document_agent.agent import DocumentAgent
from backend.agents.document_agent.schemas import AnalyzeDocumentRequest

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class DifficultyLevel(str, Enum):
    """Query difficulty levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TestResult:
    """Result of a single test."""
    document_key: str
    query_index: int
    query: str
    difficulty: DifficultyLevel
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    provider: str = "cerebras"
    libraries: List[str] = field(default_factory=list)
    execution_steps: int = 0
    status: Optional[str] = None
    phase_trace: Optional[List[str]] = None
    grounding: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    review_required: Optional[bool] = None


@dataclass
class TestSummary:
    """Summary of all test results."""
    timestamp: str
    document_key: str
    document_filename: str
    document_type: str
    total_queries: int
    successful: int
    failed: int
    success_rate: float
    avg_duration_ms: float
    total_tokens: int
    results: List[TestResult] = field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_queries == 0:
            return 0.0
        return (self.successful / self.total_queries) * 100
    
    def get_avg_duration(self) -> float:
        """Calculate average duration."""
        if not self.results:
            return 0.0
        durations = [r.duration_ms for r in self.results if r.duration_ms > 0]
        return sum(durations) / len(durations) if durations else 0.0
    
    def get_total_tokens(self) -> int:
        """Calculate total tokens."""
        return sum(r.tokens_used for r in self.results)


# ============================================================================
# DOCUMENT TEST REGISTRY
# ============================================================================

DOCUMENT_TEST_REGISTRY = {
    "simple_text": {
        "filename": "backend/tests/agents/document_agent/test_data/simple_test.txt",
        "type": "TXT",
        "description": "Simple Text Document",
        "queries": {
            DifficultyLevel.SIMPLE: [
                "What is the first line of this document?",
                "Summarize the document in one sentence.",
                "How many words are in this document?",
            ],
            DifficultyLevel.MEDIUM: [
                "List any key terms or phrases that stand out.",
                "What is the main topic and supporting points?",
                "Identify any dates, numbers, or specific data mentioned.",
                "What type of document is this (email, report, note, etc.)?",
            ],
            DifficultyLevel.HARD: [
                "Provide a concise outline of the document content.",
                "Analyze the writing style and tone of this document.",
                "Extract all actionable items or instructions if any exist.",
                "Compare the beginning and ending sections - what changed?",
            ],
        },
    },
    "simple_pdf": {
        "filename": "backend/tests/agents/document_agent/test_data/simple_test.pdf",
        "type": "PDF",
        "description": "Simple PDF Document",
        "queries": {
            DifficultyLevel.SIMPLE: [
                "What is the title of this document?",
                "Summarize the document in one sentence.",
                "How many pages does this PDF have?",
            ],
            DifficultyLevel.MEDIUM: [
                "List any headings or section titles.",
                "Extract all bold or emphasized text.",
                "What is the document structure (paragraphs, lists, tables)?",
                "Identify any metadata (author, creation date, etc.).",
            ],
            DifficultyLevel.HARD: [
                "Provide a concise outline of the document content.",
                "Analyze the document layout and formatting choices.",
                "Compare the font styles used across different sections.",
                "Generate a table of contents based on the document structure.",
            ],
        },
    },
    "simple_docx": {
        "filename": "backend/tests/agents/document_agent/test_data/simple_test.docx",
        "type": "DOCX",
        "description": "Simple Word Document",
        "queries": {
            DifficultyLevel.SIMPLE: [
                "What is the title of this document?",
                "Summarize the document in one sentence.",
                "What is the first paragraph about?",
            ],
            DifficultyLevel.MEDIUM: [
                "List any headings or section titles.",
                "Extract any bulleted or numbered lists.",
                "Identify any tables, images, or embedded objects.",
                "What formatting styles are applied (bold, italic, underline)?",
            ],
            DifficultyLevel.HARD: [
                "Provide a concise outline of the document content.",
                "Analyze the document's information hierarchy.",
                "Extract and organize all structured data (lists, tables, etc.).",
                "Suggest improvements to document organization and clarity.",
            ],
        },
    },
    "simple_png": {
        "filename": "backend/tests/agents/document_agent/test_data/simple_test.png",
        "type": "PNG",
        "description": "Simple Image Document",
        "queries": {
            DifficultyLevel.SIMPLE: [
                "Describe the visible text in this image.",
                "What is the main content of this image?",
            ],
            DifficultyLevel.MEDIUM: [
                "Extract any readable text from the image.",
                "Describe the layout and organization of content in the image.",
                "Identify any headers, titles, or emphasized text.",
            ],
            DifficultyLevel.HARD: [
                "Summarize the content of the image in one sentence.",
                "Extract all text and organize it by sections.",
                "Analyze the text quality and OCR confidence level.",
            ],
        },
    },
    "unsupported_file": {
        "filename": "backend/tests/agents/document_agent/test_data/unsupported.xyz",
        "type": "XYZ",
        "description": "Unsupported File (should fail gracefully)",
        "queries": {
            DifficultyLevel.SIMPLE: [
                "What does this file contain?",
                "Can you read this file?",
            ],
            DifficultyLevel.MEDIUM: [
                "Attempt to summarize this file.",
                "What file format is this?",
                "Provide information about this file type.",
            ],
            DifficultyLevel.HARD: [
                "Extract any structured information from this file.",
                "Suggest alternative methods to process this file.",
                "Explain why this file cannot be processed.",
            ],
        },
    },
    "corrupted_pdf": {
        "filename": "backend/tests/agents/document_agent/test_data/corrupted.pdf",
        "type": "PDF",
        "description": "Corrupted PDF (edge case for error handling)",
        "queries": {
            DifficultyLevel.SIMPLE: [
                "Can you read this document?",
                "What content is visible in this file?",
            ],
            DifficultyLevel.MEDIUM: [
                "Attempt to extract any readable portions.",
                "Identify what parts of the document are corrupted.",
                "What error messages appear when processing this file?",
            ],
            DifficultyLevel.HARD: [
                "Provide a detailed diagnosis of the file corruption.",
                "Suggest recovery methods for this corrupted PDF.",
                "Compare what should be in the file versus what's accessible.",
            ],
        },
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_libraries_in_step(step_dict: Optional[Dict]) -> List[str]:
    """Detect libraries used in execution step."""
    if not step_dict or not step_dict.get('code'):
        return []
    
    code = str(step_dict.get('code', '')).lower()
    libraries = []
    
    lib_patterns = {
        'pytesseract': ['pytesseract', 'image_to_string'],
        'python-docx': ['from docx', 'document(', 'docx'],
        'pypdf': ['pdfreader', 'pypdf'],
        'pdf2image': ['pdf2image', 'convert_from_path'],
        'cerebras': ['cerebras', 'llm.invoke'],
        'anthropic': ['anthropic', 'claude'],
    }
    
    for lib, patterns in lib_patterns.items():
        for pattern in patterns:
            if pattern in code:
                if lib not in libraries:
                    libraries.append(lib)
                break
    
    return libraries


def print_separator(char: str = "=", width: int = 70):
    """Print a separator line."""
    print(char * width)


def print_header(text: str, char: str = "*"):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"{text:^70}")
    print(f"{char * 70}\n")


def print_summary(summary: TestSummary):
    """Print test summary."""
    print_separator("=")
    print("TEST EXECUTION SUMMARY")
    print_separator("=")
    print(f"  Document: {summary.document_key} ({summary.document_type})")
    print(f"  Filename: {summary.document_filename}")
    print(f"  Timestamp: {summary.timestamp}")
    print("")
    print("  Results:")
    print(f"    [SUCCESSFUL]: {summary.successful}/{summary.total_queries} ({summary.success_rate:.1f}%)")
    print(f"    [FAILED]:     {summary.failed}/{summary.total_queries}")
    print(f"    Avg Duration: {summary.avg_duration_ms:.2f} ms")
    print(f"    Total Tokens: {summary.total_tokens}")
    print("")
    print_separator("=")


def resolve_test_file_path(rel_path: str) -> Path:
    """Resolve a dataset path to backend/tests/agents/document_agent/test_data."""
    original = Path(rel_path)
    base_dir = Path(__file__).parent / "test_data"

    if original.exists():
        return original

    name = original.name
    candidate = base_dir / name
    if candidate.exists():
        return candidate

    repo_root = Path(__file__).parent.parent.parent.parent.parent
    candidate2 = repo_root / "backend" / "tests" / "agents" / "document_agent" / "test_data" / name
    if candidate2.exists():
        return candidate2

    return candidate


def list_available_test_files() -> List[str]:
    base_dir = Path(__file__).parent / "test_data"
    if not base_dir.exists():
        return []
    return [p.name for p in base_dir.iterdir() if p.is_file()]


# ============================================================================
# TEST EXECUTION
# ============================================================================

async def run_document_test(
    document_key: str,
    query_index: int,
    query: str,
    difficulty: DifficultyLevel,
    doc_info: Dict
) -> TestResult:
    """Run a single document query test."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Get full file path
        file_path = resolve_test_file_path(doc_info['filename'])
        
        if not file_path.exists():
            return TestResult(
                document_key=document_key,
                query_index=query_index,
                query=query,
                difficulty=difficulty,
                success=False,
                error=f"File not found: {file_path}",
                duration_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
        
        agent = DocumentAgent()
        
        # Build vector store path (if using RAG)
        # Prefer test-store naming, then fall back to orchestrator naming (filename.faiss)
        vector_store_path = WORKSPACE_ROOT / "storage" / "vector_store" / f"{document_key}_store"
        orchestrator_store_path = WORKSPACE_ROOT / "storage" / "vector_store" / f"{file_path.name}.faiss"
        
        # Pick whichever exists
        resolved_store = None
        if vector_store_path.exists():
            resolved_store = str(vector_store_path)
        elif orchestrator_store_path.exists():
            resolved_store = str(orchestrator_store_path)

        request = AnalyzeDocumentRequest(
            file_path=str(file_path),
            vector_store_path=resolved_store,
            query=query,
            thread_id=f"test-{document_key}-{query_index}"
        )
        
        # Execute analysis (may not have async yet, but prepare for it)
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.analyze_document(request)
        )
        
        # Extract execution details
        execution_steps = result.get('execution_steps', [])
        execution_metrics = result.get('execution_metrics', {})
        
        # Collect all libraries from steps
        all_libraries = []
        for step in execution_steps:
            libs = detect_libraries_in_step(step)
            all_libraries.extend([l for l in libs if l not in all_libraries])
        
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Handle None result (e.g., from unsupported file types)
        if result is None:
            return TestResult(
                document_key=document_key,
                query_index=query_index,
                query=query,
                difficulty=difficulty,
                success=False,
                error="Agent returned None - file processing failed",
                duration_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
        
        success = result.get('success', False)
        error_message = None
        if not success:
            error_message = result.get('error') or result.get('message')
        
        # Safely get answer with fallback
        answer = result.get('answer', '') or ''
        answer_truncated = answer[:200] if answer else None

        return TestResult(
            document_key=document_key,
            query_index=query_index,
            query=query,
            difficulty=difficulty,
            success=success,
            answer=answer_truncated,
            error=error_message,
            duration_ms=duration_ms,
            tokens_used=execution_metrics.get('total_tokens_input', 0),
            provider=execution_metrics.get('provider_used', 'cerebras'),
            libraries=all_libraries,
            execution_steps=len(execution_steps),
            status=result.get('status'),
            phase_trace=result.get('phase_trace'),
            grounding=result.get('grounding'),
            confidence=result.get('confidence'),
            review_required=result.get('review_required')
        )
    
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        return TestResult(
            document_key=document_key,
            query_index=query_index,
            query=query,
            difficulty=difficulty,
            success=False,
            error=str(e),
            duration_ms=(asyncio.get_event_loop().time() - start_time) * 1000
        )


async def run_document_tests(
    document_key: str,
    difficulty: Optional[DifficultyLevel] = None
) -> TestSummary:
    """Run all tests for a document."""
    if document_key not in DOCUMENT_TEST_REGISTRY:
        raise ValueError(f"Unknown document: {document_key}")
    
    doc_info = DOCUMENT_TEST_REGISTRY[document_key]
    file_path = resolve_test_file_path(doc_info['filename'])
    
    # Verify file exists
    if not file_path.exists():
        print(f"\n[ERROR] Document file not found: {file_path}")
        available = list_available_test_files()
        if available:
            print(f"       Available test documents: {', '.join(available)}")
        sys.exit(1)
    
    print_header(f"Testing Document: {document_key.upper()}", "*")
    print(f"Loaded: {file_path.name}")
    print(f"Type: {doc_info['type']}")
    print(f"Description: {doc_info['description']}\n")
    
    # Get queries for specified difficulty or all
    if difficulty:
        queries = doc_info['queries'].get(difficulty, [])
        difficulties = [difficulty]
    else:
        queries = []
        difficulties = []
        for diff in [DifficultyLevel.SIMPLE, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
            queries.extend(doc_info['queries'].get(diff, []))
            difficulties.extend([diff] * len(doc_info['queries'].get(diff, [])))
    
    # Run all tests sequentially
    print(f"Running {len(queries)} queries...\n")
    print_separator("-", 70)

    results = []
    for idx, (query, diff) in enumerate(zip(queries, difficulties), 1):
        print(f"[{idx}/{len(queries)}] ({diff.value}) {query[:60]}...")
        result = await run_document_test(document_key, idx, query, diff, doc_info)
        results.append(result)
    
    print("\n" + "=" * 70)
    
    # Create summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    summary = TestSummary(
        timestamp=datetime.now().isoformat(),
        document_key=document_key,
        document_filename=doc_info['filename'],
        document_type=doc_info['type'],
        total_queries=len(results),
        successful=successful,
        failed=failed,
        success_rate=0.0,
        avg_duration_ms=0.0,
        total_tokens=0,
        results=results
    )
    
    # Update calculated fields
    summary.success_rate = summary.get_success_rate()
    summary.avg_duration_ms = summary.get_avg_duration()
    summary.total_tokens = summary.get_total_tokens()
    
    # Print summary
    print_summary(summary)
    
    # Print detailed results
    for result in results:
        status = "[SUCCESSFUL]" if result.success else "[FAILED]"
        print(f"\n{status} Query {result.query_index} ({result.difficulty.value})")
        print(f"  Question: {result.query}")
        
        if result.success:
            if result.answer:
                print(f"  Answer: {result.answer}")
            if result.libraries:
                print(f"  Libraries Used: {', '.join(result.libraries)}")
            print(f"  Duration: {result.duration_ms:.2f}ms")
            print(f"  Tokens: {result.tokens_used}")
            print(f"  Provider: {result.provider}")
            if result.execution_steps > 0:
                print(f"  Execution Steps: {result.execution_steps}")
        else:
            print(f"  [ERROR] {result.error}")
            print(f"  Duration: {result.duration_ms:.2f}ms")
    
    return summary


async def list_available_documents():
    """List all available test documents."""
    print_header("AVAILABLE TEST DOCUMENTS", "*")
    
    for key, doc_info in DOCUMENT_TEST_REGISTRY.items():
        file_path = resolve_test_file_path(doc_info['filename'])
        exists = "[OK]" if file_path.exists() else "[MISSING]"
        
        print(f"{key:20} {exists:12} {doc_info['type']:6} {doc_info['description']}")
        
        # List query counts per difficulty
        for difficulty in [DifficultyLevel.SIMPLE, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
            count = len(doc_info['queries'].get(difficulty, []))
            if count > 0:
                print(f"  - {difficulty.value:8} queries: {count}")
    
    print("")


async def main():
    """Main test runner with CLI support."""
    parser = argparse.ArgumentParser(
        description="Document Agent Manual Test Suite"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test documents"
    )
    parser.add_argument(
        "--document",
        type=str,
        help="Test a specific document (e.g., phd_thesis)"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["simple", "medium", "hard"],
        help="Run only queries of specified difficulty"
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--all-documents",
        action="store_true",
        help="Test all available documents"
    )
    
    args = parser.parse_args()
    
    if args.list:
        await list_available_documents()
        return
    
    all_summaries: List[TestSummary] = []
    
    if args.all_documents:
        print_header("TESTING ALL DOCUMENTS", "*")
        for doc_key in DOCUMENT_TEST_REGISTRY.keys():
            try:
                difficulty = DifficultyLevel(args.difficulty) if args.difficulty else None
                summary = await run_document_tests(doc_key, difficulty)
                all_summaries.append(summary)
            except Exception as e:
                logger.error(f"Error testing {doc_key}: {e}", exc_info=True)
    
    elif args.document:
        try:
            difficulty = DifficultyLevel(args.difficulty) if args.difficulty else None
            summary = await run_document_tests(args.document, difficulty)
            all_summaries.append(summary)
        except ValueError as e:
            print(f"[ERROR] {e}")
            await list_available_documents()
            sys.exit(1)
    
    else:
        # Run all documents if no specific document specified
        print_header("DOCUMENT AGENT TEST SUITE", "*")
        for doc_key in DOCUMENT_TEST_REGISTRY.keys():
            try:
                difficulty = DifficultyLevel(args.difficulty) if args.difficulty else None
                summary = await run_document_tests(doc_key, difficulty)
                all_summaries.append(summary)
            except Exception as e:
                logger.error(f"Error testing {doc_key}: {e}", exc_info=True)
    
    # Output results
    if args.output == "json":
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "summaries": [
                {
                    "document_key": s.document_key,
                    "document_filename": s.document_filename,
                    "document_type": s.document_type,
                    "total_queries": s.total_queries,
                    "successful": s.successful,
                    "failed": s.failed,
                    "success_rate": s.success_rate,
                    "avg_duration_ms": s.avg_duration_ms,
                    "total_tokens": s.total_tokens,
                    "results": [asdict(r) for r in s.results]
                }
                for s in all_summaries
            ]
        }
        
        output_file = WORKSPACE_ROOT / "backend" / "tests" / "document_test_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n[OK] Results saved to: {output_file}\n")
    
    # Print overall summary
    if all_summaries:
        print_header("OVERALL SUMMARY", "=")
        total_queries = sum(s.total_queries for s in all_summaries)
        total_successful = sum(s.successful for s in all_summaries)
        total_failed = sum(s.failed for s in all_summaries)
        overall_rate = (total_successful / total_queries * 100) if total_queries > 0 else 0
        
        print(f"  Total Documents Tested: {len(all_summaries)}")
        print(f"  Total Queries: {total_queries}")
        print(f"  Successful: {total_successful} ({overall_rate:.1f}%)")
        print(f"  Failed: {total_failed}")
        print("")


if __name__ == "__main__":
    asyncio.run(main())

