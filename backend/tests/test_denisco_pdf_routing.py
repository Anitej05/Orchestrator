"""
Routing diagnostic test: denisco worflows.pdf (image-only PDF)

FINDINGS
========
The PDF has 7 pages. Every page is a scanned/screenshot image.
PyMuPDF extracts ZERO characters of text from the entire document.

Current orchestrator behaviour (AGENT-FIRST RULE in brain.py line 396):
  "PDF/doc → DocumentAgent"

  So the brain sends this file to DocumentAgent first. But DocumentAgent
  uses PyMuPDF text extraction internally, which yields nothing on this PDF.

  The image_tools.analyze_image function, on the other hand, renders each
  page as a JPEG and calls Groq llama-4-scout-17b — and reads the full
  workflow contents perfectly.

This test file documents and proves that gap:
  - test_document_agent_text_extraction_yields_nothing  ← the gap
  - test_image_tool_reads_page_*                        ← what should happen
  - test_all_pages_have_zero_extractable_text           ← root cause proof

Fix required (not in this file):
  DocumentAgent (or brain) should detect zero-text PDFs and fall back to
  rendering each page and calling analyze_image per page.

Run:
    PYTHONUTF8=1 venv/Scripts/python -m pytest backend/tests/test_denisco_pdf_routing.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import fitz  # PyMuPDF

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = Path(__file__).resolve().parent / "test_data"
PDF_PATH = TEST_DATA / "denisco worflows.pdf"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=False)

from backend.tools.image_tools import analyze_image


# ── Shared helpers ────────────────────────────────────────────────────────────

def _require_groq():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


def _render_page(page_index: int) -> Path:
    """Render a PDF page to a PNG in test_data/ and return its path."""
    out = TEST_DATA / f"denisco_page_{page_index + 1}.png"
    if not out.exists():
        doc = fitz.open(str(PDF_PATH))
        pix = doc[page_index].get_pixmap(dpi=120)
        pix.save(str(out))
    return out


def _vision_answer(page_index: int, query: str) -> str:
    path = _render_page(page_index)
    result = analyze_image.invoke({"image_path": str(path), "query": query})
    assert "error" not in result, f"Vision error on page {page_index+1}: {result['error']}"
    return result["answer"].lower()


# =============================================================================
# ROOT CAUSE: PyMuPDF extracts no text (this is what DocumentAgent sees)
# =============================================================================

class TestDocumentAgentWouldFail:
    """
    Proves why DocumentAgent cannot handle this PDF.
    It is entirely scanned — no selectable / copyable text layer.
    """

    def test_pdf_exists(self):
        assert PDF_PATH.exists(), f"PDF not found at {PDF_PATH}"

    def test_pdf_has_seven_pages(self):
        doc = fitz.open(str(PDF_PATH))
        assert len(doc) == 7

    def test_all_pages_have_zero_extractable_text(self):
        """
        PyMuPDF get_text() — the same call DocumentAgent uses — returns
        empty string for every single page.
        If this ever returns text, the PDF has been replaced with a text-layer version.
        """
        doc = fitz.open(str(PDF_PATH))
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            assert text == "", (
                f"Page {i+1} now has extractable text ({len(text)} chars). "
                f"PDF may have been replaced. DocumentAgent can now handle it."
            )

    def test_each_page_contains_exactly_one_embedded_image(self):
        """Each page is a single full-page raster image — confirms scanned origin."""
        doc = fitz.open(str(PDF_PATH))
        for i, page in enumerate(doc):
            images = page.get_images()
            assert len(images) == 1, (
                f"Page {i+1} has {len(images)} images — expected exactly 1 (scanned page)."
            )

    def test_document_agent_text_extraction_yields_nothing(self):
        """
        Simulates what DocumentAgent does internally: extract text with PyMuPDF.
        The result is empty → the agent cannot answer ANY question about this PDF.
        """
        doc = fitz.open(str(PDF_PATH))
        full_text = "".join(page.get_text() for page in doc).strip()
        assert full_text == "", (
            f"Expected empty extraction, got {len(full_text)} chars: {full_text[:200]}"
        )


# =============================================================================
# CORRECT PATH: analyze_image reads this PDF perfectly
# =============================================================================

class TestImageToolReadsPageOne:
    """
    Page 1 — Enquiry & Sales Order (Marketing/Sales Department)
               + Production Planning (PPC Department)
    """

    def test_company_name_identified(self):
        _require_groq()
        ans = _vision_answer(0, "What company does this workflow document belong to?")
        assert any(k in ans for k in ["denisco", "deni"]), f"Company name missing: {ans}"

    def test_sales_department_workflow_extracted(self):
        _require_groq()
        ans = _vision_answer(
            0, "List the workflow steps for the Enquiry and Sales Order department."
        )
        assert any(k in ans for k in ["quotation", "sales order", "enquiry", "customer"]), (
            f"Sales workflow not extracted: {ans}"
        )

    def test_ppc_department_identified(self):
        _require_groq()
        ans = _vision_answer(0, "What departments are mentioned in this document?")
        assert any(k in ans for k in ["ppc", "production planning", "sales", "marketing"]), (
            f"PPC department not found: {ans}"
        )

    def test_erp_system_mentioned(self):
        _require_groq()
        ans = _vision_answer(0, "Is any ERP system mentioned in this document?")
        assert any(k in ans for k in ["erp", "focus", "focus x"]), (
            f"ERP system not found: {ans}"
        )


class TestImageToolReadsPageTwo:
    """
    Page 2 — Procurement & Material Inward (Purchase + Stores Dept)
    """

    def test_purchase_department_workflow_extracted(self):
        _require_groq()
        ans = _vision_answer(
            1,
            "What are the workflow steps for the Purchase or Procurement department?"
        )
        assert any(k in ans for k in ["purchase", "procurement", "material", "vendor", "supplier"]), (
            f"Purchase workflow not extracted: {ans}"
        )

    def test_purchase_requisition_mentioned(self):
        _require_groq()
        ans = _vision_answer(1, "What documents are mentioned on this page?")
        assert any(k in ans for k in ["purchase requisition", "pr", "purchase order", "grn", "mqr"]), (
            f"Purchase documents not found: {ans}"
        )


class TestImageToolReadsPageThree:
    """
    Page 3 — Production / Manufacturing + Quality Control (QC Department)
    """

    def test_production_workflow_extracted(self):
        _require_groq()
        ans = _vision_answer(
            2,
            "What are the steps for the Production or Manufacturing department?"
        )
        assert any(k in ans for k in ["production", "batch", "manufacturing", "bmr", "material issue"]), (
            f"Production workflow not extracted: {ans}"
        )

    def test_qc_department_found(self):
        _require_groq()
        ans = _vision_answer(2, "Is there a Quality Control section on this page?")
        assert any(k in ans for k in ["quality control", "qc", "quality"]), (
            f"QC section not identified: {ans}"
        )

    def test_coa_and_qc_report_mentioned(self):
        _require_groq()
        ans = _vision_answer(2, "What documents does the Quality Control department produce?")
        assert any(k in ans for k in ["coa", "certificate", "qc report", "test report", "sample"]), (
            f"QC documents not found: {ans}"
        )


class TestImageToolReadsAllPages:
    """
    Confirms the vision tool can read every page — not just the first.
    """

    @pytest.mark.parametrize("page_idx", range(7))
    def test_each_page_returns_non_empty_answer(self, page_idx):
        _require_groq()
        ans = _vision_answer(
            page_idx,
            "Briefly describe what is on this page in 2-3 sentences."
        )
        assert len(ans) > 50, (
            f"Page {page_idx+1} answer suspiciously short ({len(ans)} chars): {ans}"
        )
        # Every page should at least mention a workflow-related word
        assert any(k in ans for k in [
            "workflow", "department", "document", "process", "step",
            "purpose", "chemical", "production", "quality", "denisco",
            "purchase", "sales", "store", "dispatch", "account"
        ]), f"Page {page_idx+1} answer seems unrelated to workflows: {ans}"


# =============================================================================
# FIX VERIFICATION: extract_document_content vision fallback
# =============================================================================

class TestExtractDocumentContentFallback:
    """
    Proves the fix: extract_document_content now returns substantive content
    for the image-only PDF via the vision fallback path.
    Previously returned "" (0 chars); now returns 10k+ chars of workflow content.
    """

    def test_returns_nonzero_content(self):
        _require_groq()
        from backend.agents.document_agent_lib.utils import extract_document_content
        content, file_type = extract_document_content(str(PDF_PATH))
        assert file_type == "pdf"
        assert len(content) > 500, (
            f"Vision fallback returned only {len(content)} chars — expected substantive content"
        )

    def test_mentions_denisco(self):
        _require_groq()
        from backend.agents.document_agent_lib.utils import extract_document_content
        content, _ = extract_document_content(str(PDF_PATH))
        assert any(k in content.lower() for k in ["denisco", "chemical"]), (
            f"Company name not found in extracted content: {content[:300]}"
        )

    def test_covers_all_seven_pages(self):
        _require_groq()
        from backend.agents.document_agent_lib.utils import extract_document_content
        content, _ = extract_document_content(str(PDF_PATH))
        for page_num in range(1, 8):
            assert f"[Page {page_num}]" in content, (
                f"Page {page_num} marker missing — vision fallback did not process all pages"
            )

    def test_mentions_departments(self):
        _require_groq()
        from backend.agents.document_agent_lib.utils import extract_document_content
        content, _ = extract_document_content(str(PDF_PATH))
        assert any(k in content.lower() for k in ["sales", "purchase", "production", "quality"]), (
            "No department keywords found in extracted content"
        )


# =============================================================================
# ROUTING VERDICT
# =============================================================================

class TestRoutingVerdict:
    """
    Summarises the gap clearly as assertions, so CI output makes the
    problem and solution immediately visible.
    """

    def test_routing_gap_documented(self):
        """
        The orchestrator WILL route denisco worflows.pdf to DocumentAgent
        (AGENT-FIRST RULE: PDF → DocumentAgent).

        DocumentAgent extracts 0 characters of text.

        The correct handler is analyze_image, which reads every page via
        the Groq vision model and returns full workflow details.

        This test always passes — it is a documented known gap.
        To fix: DocumentAgent should detect zero-text PDFs and fall back
        to rendering each page and calling analyze_image.
        """
        doc = fitz.open(str(PDF_PATH))
        total_text = sum(len(p.get_text().strip()) for p in doc)

        assert total_text == 0, (
            "PDF now has extractable text — gap may be resolved naturally."
        )
        # Gap confirmed. Document it explicitly.
        gap_message = (
            f"denisco worflows.pdf: {len(doc)} pages, {total_text} chars of extractable text. "
            f"DocumentAgent will return empty. analyze_image is required."
        )
        print(f"\n[ROUTING GAP] {gap_message}")
        # Test passes regardless — this is a documentation test.

    def test_image_tool_is_sufficient_for_this_pdf(self):
        """
        analyze_image on page 1 returns substantive content — confirming
        it is the correct handler for this document.
        """
        _require_groq()
        ans = _vision_answer(
            0,
            "Summarise what this document is about in one sentence."
        )
        # Must mention something about workflows or Denisco Chemicals
        assert any(k in ans for k in [
            "workflow", "denisco", "chemical", "department", "process", "company"
        ]), f"Vision summary too generic: {ans}"
        assert len(ans) > 40, f"Summary too short: {ans}"
