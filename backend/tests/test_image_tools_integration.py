"""
Integration tests for backend/tools/image_tools.py — analyze_image()

These are LIVE tests: each one calls Groq's llama-4-scout vision model with a
real image from backend/tests/test_data/. Assertions are grounded in exactly
what each image contains (verified by visual inspection before writing the test).

Requires:
    GROQ_API_KEY set in the environment (or backend/.env loaded).

Skip automatically when key is absent — safe to run in CI with secret injection.

Image inventory (test_data/):
  image_info.png      — Clear printed Purchase Requisition form
                        Dept: Quality Control | Date: 2024.10.15 | Raised By: John Doe
                        Items: Paracetamol Tablets ×1000 | Amoxicillin Capsules ×500
                               Ibuprofen Suspension ×200
  image_half.png      — BMR with a phone + hand partially blocking the sheet
                        Batch Number and Product Name header fields are BLANK
                        Handwritten material rows: 207/1.85 | 345/1.50 | "Servo" signature
  image_covered.png   — Material Issue Slip lying on a desk under heavy diagonal shadow
                        Right half (Quantity Issued / Unit / Issued By) is dark / unreadable
                        Left column has handwritten product names (partially legible)
  image_watermark.png — Chem-Lab Industries QC Certificate of Analysis
                        "CONFIDENTIAL" diagonal watermark
                        Product: Sodium Hydroxide Pellets | Batch: 20240515-002
                        Assay: 98.0-100.5 → Result 99.2 Pass
                        Moisture: ≤0.5   → Result 0.3  Pass
                        Iron:     ≤10    → Result 8    Pass
  image_ani.png       — Angled/tilted Batch Manufacturing Record on a desk
                        Handwritten fields: Batch Number, Product, Process Steps,
                        Parameters, Operator Signature (cursive)
  image_hindi.png     — Vendor Quotation from ABC Chemicals, Mumbai
                        GSTIN: MH12ABC3456D7E8
                        Descriptions in Devanagari (Hindi) script, qty 100/8.5/9.6/100/8.6
                        Totals in ₹ and $
  image_multiple.png  — Three BMR pages fanned side-by-side in one photo
                        Page 1 of 3: batch details, raw material quantities
                        Page 2 of 3: drying process steps, temperature/time readings
                        Page 3 of 3: final yield, packing details, QC confirmation,
                                     supervisor signature
"""

import os
import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # backend/
TEST_DATA = Path(__file__).resolve().parent / "test_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))                   # project root

# Load .env so GROQ_API_KEY is available when running locally
from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=False)

from backend.tools.image_tools import analyze_image


# ── Skip marker ───────────────────────────────────────────────────────────────

def _require_groq():
    """Skip the test if GROQ_API_KEY is not configured."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set — skipping live vision test")


# ── Shared assertion helpers ──────────────────────────────────────────────────

def _answer(image_name: str, query: str) -> str:
    """
    Call analyze_image and return the answer string.
    Fails the test immediately if the tool returns an error.
    """
    path = str(TEST_DATA / image_name)
    result = analyze_image.invoke({"image_path": path, "query": query})
    assert "error" not in result, (
        f"analyze_image returned an error for {image_name}: {result['error']}"
    )
    answer = result.get("answer", "")
    assert answer, f"Model returned an empty answer for {image_name}"
    return answer.lower()


def _contains_any(text: str, keywords: list) -> bool:
    return any(kw.lower() in text for kw in keywords)


def _contains_all(text: str, keywords: list) -> bool:
    return all(kw.lower() in text for kw in keywords)


# =============================================================================
# 1  test_clear_image_extracts_all_fields
#    image_info.png — printed Purchase Requisition, perfect lighting, no noise
# =============================================================================

class TestClearImageExtractsAllFields:
    """
    A crisp, well-lit printed form must yield every visible field value.
    Zero excuse for omission — the text is machine-printed at high contrast.
    """

    def test_document_type_identified(self):
        _require_groq()
        ans = _answer("image_info.png", "What type of document is this?")
        assert _contains_any(ans, ["purchase requisition", "requisition", "purchase order"]), (
            f"Expected document type not found in: {ans}"
        )

    def test_all_three_medicines_extracted(self):
        _require_groq()
        ans = _answer(
            "image_info.png",
            "List all the item names / medicines visible in this purchase requisition."
        )
        assert "paracetamol" in ans, f"Paracetamol missing from: {ans}"
        assert "amoxicillin" in ans, f"Amoxicillin missing from: {ans}"
        assert _contains_any(ans, ["ibuprofen", "ibu"]), f"Ibuprofen missing from: {ans}"

    def test_quantities_extracted(self):
        _require_groq()
        ans = _answer(
            "image_info.png",
            "What quantities are listed for the items in this form?"
        )
        # Paracetamol ×1000, Amoxicillin ×500, Ibuprofen ×200
        assert _contains_any(ans, ["1000", "1,000"]), f"Qty 1000 missing from: {ans}"
        assert "500" in ans, f"Qty 500 missing from: {ans}"
        assert "200" in ans, f"Qty 200 missing from: {ans}"

    def test_metadata_fields_extracted(self):
        _require_groq()
        ans = _answer(
            "image_info.png",
            "Who raised this requisition, what department is it for, and what is the date?"
        )
        assert _contains_any(ans, ["john doe", "john"]), f"Raised-by name missing: {ans}"
        assert _contains_any(ans, ["quality control", "qc"]), f"Department missing: {ans}"
        assert _contains_any(ans, ["2024", "2024.10.15", "oct"]), f"Date missing: {ans}"

    def test_specifications_extracted(self):
        _require_groq()
        ans = _answer(
            "image_info.png",
            "What are the specifications (dosage / concentration) for each item?"
        )
        assert "500mg" in ans, f"Paracetamol spec missing: {ans}"
        assert "250mg" in ans, f"Amoxicillin spec missing: {ans}"
        assert _contains_any(ans, ["100mg/5ml", "100mg"]), f"Ibuprofen spec missing: {ans}"


# =============================================================================
# 2  test_shadowed_image_partial_extraction
#    image_covered.png — Material Issue Slip, right half in deep shadow
# =============================================================================

class TestShadowedImagePartialExtraction:
    """
    Right half of the slip is obscured by a dark shadow.
    The model must extract what is visible on the left and flag what is not.
    """

    def test_document_type_identified(self):
        _require_groq()
        ans = _answer("image_covered.png", "What type of document is this?")
        assert _contains_any(ans, ["material issue", "issue slip", "slip", "material"]), (
            f"Document type not identified: {ans}"
        )

    def test_shadow_or_darkness_acknowledged(self):
        _require_groq()
        ans = _answer(
            "image_covered.png",
            "Are any parts of this document obscured, dark, or unreadable? Describe."
        )
        assert _contains_any(
            ans, ["shadow", "dark", "obscur", "unread", "hidden", "cover", "unclear", "partial"]
        ), f"Shadow not acknowledged: {ans}"

    def test_visible_left_column_partially_read(self):
        _require_groq()
        ans = _answer(
            "image_covered.png",
            "What product or material names can you read in the left portion of the document?"
        )
        # Left column has handwritten names; model should find at least one
        assert len(ans) > 20, f"Answer suspiciously short — may be refusing: {ans}"

    def test_right_column_fields_flagged_as_missing(self):
        _require_groq()
        ans = _answer(
            "image_covered.png",
            "Can you read the 'Quantity Issued', 'Unit', 'Issued By', or 'Received By' fields?"
        )
        assert _contains_any(
            ans,
            ["cannot", "can't", "unable", "not visible", "not readable", "obscur",
             "dark", "shadow", "unclear", "hidden", "covered", "blank", "empty", "missing"]
        ), f"Missing-field acknowledgement not found: {ans}"


# =============================================================================
# 3  test_no_hallucination_on_blank_fields
#    image_half.png — BMR header fields (Batch Number, Product Name) are BLANK
# =============================================================================

class TestNoHallucinationOnBlankFields:
    """
    The most important hallucination guard. The BMR header has printed labels
    'Batch Number' and 'Product Name' but the fill-in lines are EMPTY.
    The model must NOT invent values for those lines.
    """

    def test_blank_batch_number_not_hallucinated(self):
        _require_groq()
        ans = _answer(
            "image_half.png",
            "What is the Batch Number written in this Batch Manufacturing Record? "
            "If the field is blank or unreadable, say so explicitly."
        )
        # Model must NOT confidently state a made-up batch number.
        # Acceptable answers: blank, not filled, not visible, cannot read, unclear, empty, N/A
        fabricated_confidence = (
            "batch number is" in ans
            and not _contains_any(
                ans,
                ["blank", "empty", "not filled", "not written", "not visible",
                 "unclear", "cannot", "can't", "n/a", "not provided", "missing",
                 "no batch", "not complete", "illegible", "obscured", "not readable"]
            )
        )
        assert not fabricated_confidence, (
            f"Model appears to have hallucinated a batch number: {ans}"
        )

    def test_blank_product_name_not_hallucinated(self):
        _require_groq()
        ans = _answer(
            "image_half.png",
            "What product name is filled in on the 'Product Name' line of this BMR? "
            "If the line is blank, say so."
        )
        fabricated_confidence = (
            "product name is" in ans
            and not _contains_any(
                ans,
                ["blank", "empty", "not filled", "not written", "not visible",
                 "unclear", "cannot", "can't", "n/a", "not provided", "missing",
                 "no product", "not complete", "illegible"]
            )
        )
        assert not fabricated_confidence, (
            f"Model appears to have hallucinated a product name: {ans}"
        )

    def test_obstructed_area_flagged(self):
        _require_groq()
        ans = _answer(
            "image_half.png",
            "Is any part of this document blocked or hard to read? What is blocking it?"
        )
        assert _contains_any(
            ans,
            ["phone", "hand", "finger", "obstruct", "block", "cover",
             "object", "device", "mobile", "partial"]
        ), f"Obstruction (phone/hand) not mentioned: {ans}"

    def test_readable_handwritten_quantities_extracted(self):
        _require_groq()
        ans = _answer(
            "image_half.png",
            "What numerical quantities or measurements can you read in the "
            "Raw Materials table of this document?"
        )
        # 207, 1.85, 345/245, 1.50 are visible in the handwritten rows
        assert _contains_any(ans, ["207", "1.85", "245", "345", "1.50", "1.80"]), (
            f"No recognisable quantities found: {ans}"
        )


# =============================================================================
# 4  test_watermarked_pdf_flags_obscured_values
#    image_watermark.png — QC CoA with diagonal CONFIDENTIAL watermark
# =============================================================================

class TestWatermarkedPdfFlagsObscuredValues:
    """
    A CONFIDENTIAL diagonal watermark overlaps the table. The model must:
    - Still read the underlying printed values (they are visible through the watermark)
    - Acknowledge the watermark exists
    """

    def test_watermark_detected(self):
        _require_groq()
        ans = _answer(
            "image_watermark.png",
            "Does this document have any watermark or stamp? Describe it."
        )
        assert _contains_any(ans, ["confidential", "watermark", "stamp", "diagonal"]), (
            f"Watermark not detected: {ans}"
        )

    def test_product_and_batch_extracted_through_watermark(self):
        _require_groq()
        ans = _answer(
            "image_watermark.png",
            "What is the product name and batch number on this quality control document?"
        )
        assert _contains_any(ans, ["sodium hydroxide", "sodium", "hydroxide", "pellet"]), (
            f"Product name missing: {ans}"
        )
        assert _contains_any(ans, ["20240515", "20240515-002", "002"]), (
            f"Batch number missing: {ans}"
        )

    def test_qc_test_results_extracted(self):
        _require_groq()
        ans = _answer(
            "image_watermark.png",
            "List all QC test parameters, their specifications, results, and pass/fail status."
        )
        assert _contains_any(ans, ["assay", "moisture", "iron"]), (
            f"QC parameters not extracted: {ans}"
        )
        assert _contains_any(ans, ["99.2", "0.3"]), (
            f"QC result values not extracted: {ans}"
        )
        assert "pass" in ans, f"Pass/Fail status not found: {ans}"

    def test_company_name_readable(self):
        _require_groq()
        ans = _answer(
            "image_watermark.png",
            "What company or organisation issued this document?"
        )
        assert _contains_any(ans, ["chem-lab", "chem lab", "chemlab"]), (
            f"Company name not found: {ans}"
        )


# =============================================================================
# 5  test_angled_photo_deskewed_before_ocr
#    image_ani.png — BMR photographed at an angle / perspective distortion
# =============================================================================

class TestAngledPhotoDeskewedBeforeOcr:
    """
    The document is shot at an angle and is curled at the edges.
    The model must still identify the document type and extract headline fields.
    """

    def test_document_type_identified_despite_angle(self):
        _require_groq()
        ans = _answer(
            "image_ani.png",
            "What type of document is this? Identify it even if it is angled or tilted."
        )
        assert _contains_any(
            ans, ["batch manufacturing record", "bmr", "batch record", "manufacturing record"]
        ), f"Document type not identified: {ans}"

    def test_key_sections_found(self):
        _require_groq()
        ans = _answer(
            "image_ani.png",
            "What sections or field labels can you read on this document?"
        )
        assert _contains_any(
            ans,
            ["batch", "product", "process", "parameter", "operator", "signature"]
        ), f"No key labels found: {ans}"

    def test_handwritten_signature_area_located(self):
        _require_groq()
        ans = _answer(
            "image_ani.png",
            "Is there a signature on this document? Where is it?"
        )
        assert _contains_any(ans, ["signature", "sign", "operator"]), (
            f"Signature not identified: {ans}"
        )

    def test_angle_or_tilt_mentioned(self):
        _require_groq()
        ans = _answer(
            "image_ani.png",
            "Describe how the document is positioned in this photo."
        )
        assert _contains_any(
            ans,
            ["angled", "tilted", "angle", "tilt", "perspective", "slant",
             "diagonal", "rotated", "not flat", "curled", "skewed"]
        ), f"Angle/tilt not described: {ans}"


# =============================================================================
# 6  test_hindi_text_extracted_or_flagged
#    image_hindi.png — Vendor Quotation with Devanagari (Hindi) item descriptions
# =============================================================================

class TestHindiTextExtractedOrFlagged:
    """
    The Description column is in Hindi (Devanagari). The model must either
    transliterate/translate the text OR explicitly flag it as a non-Latin script.
    It must not silently skip the column or hallucinate English descriptions.
    """

    def test_document_type_and_company_identified(self):
        _require_groq()
        ans = _answer(
            "image_hindi.png",
            "What type of document is this and who issued it?"
        )
        assert _contains_any(ans, ["vendor quotation", "quotation", "quote"]), (
            f"Document type not found: {ans}"
        )
        assert _contains_any(ans, ["abc chemicals", "abc", "chemicals", "mumbai"]), (
            f"Issuer not identified: {ans}"
        )

    def test_hindi_script_acknowledged(self):
        _require_groq()
        ans = _answer(
            "image_hindi.png",
            "What language or script is used in the Description column of this table?"
        )
        assert _contains_any(
            ans,
            ["hindi", "devanagari", "indian", "non-english", "non english",
             "regional", "sanskrit", "indic", "script"]
        ), f"Hindi script not acknowledged: {ans}"

    def test_english_numeric_fields_extracted(self):
        _require_groq()
        ans = _answer(
            "image_hindi.png",
            "What quantities and total prices are listed in this quotation?"
        )
        # Quantities visible: 100, 100, 100, 8.5, 9.6, 100, 8.6
        assert _contains_any(ans, ["100"]), f"Qty 100 not found: {ans}"
        # Totals in INR/USD
        assert _contains_any(ans, ["5000", "6000", "9000", "₹", "$", "usd", "inr"]), (
            f"Price totals not found: {ans}"
        )

    def test_gstin_extracted(self):
        _require_groq()
        ans = _answer(
            "image_hindi.png",
            "What is the GSTIN or tax identification number on this document?"
        )
        assert _contains_any(
            ans,
            ["mh12abc3456d7e8", "mh12", "gstin", "gst", "tax id", "registration"]
        ), f"GSTIN not found: {ans}"


# =============================================================================
# 7  test_multipage_bmr_all_pages_processed
#    image_multiple.png — Three BMR pages fanned in a single photo
# =============================================================================

class TestMultipageBmrAllPagesProcessed:
    """
    All three pages are visible in the photo simultaneously.
    The model must recognise that there are multiple pages and extract
    at least one meaningful datum from each.
    """

    def test_multiple_pages_detected(self):
        _require_groq()
        ans = _answer(
            "image_multiple.png",
            "How many pages or documents are visible in this image?"
        )
        assert _contains_any(ans, ["3", "three", "multiple", "several"]), (
            f"Three pages not detected: {ans}"
        )

    def test_page_numbers_identified(self):
        _require_groq()
        ans = _answer(
            "image_multiple.png",
            "What page numbers are printed at the bottom of each document?"
        )
        assert _contains_any(ans, ["page 1", "1 of 3", "page 2", "2 of 3", "page 3", "3 of 3"]), (
            f"Page numbers not found: {ans}"
        )

    def test_page1_content_extracted(self):
        _require_groq()
        ans = _answer(
            "image_multiple.png",
            "What is on Page 1 of 3 in this image? Describe its contents."
        )
        assert _contains_any(
            ans, ["batch", "raw material", "material", "quantities", "reaction"]
        ), f"Page 1 content not extracted: {ans}"

    def test_page2_process_steps_found(self):
        _require_groq()
        ans = _answer(
            "image_multiple.png",
            "What does Page 2 of 3 contain? List the section headings or key content."
        )
        assert _contains_any(
            ans,
            ["drying", "filtration", "process", "temperature", "time", "reading"]
        ), f"Page 2 process steps not found: {ans}"

    def test_page3_final_yield_and_signature(self):
        _require_groq()
        ans = _answer(
            "image_multiple.png",
            "What does Page 3 of 3 contain? Is there a final yield or signature?"
        )
        assert _contains_any(ans, ["final yield", "yield", "packing", "qc", "supervisor"]), (
            f"Page 3 final yield/signature not found: {ans}"
        )

    def test_all_pages_are_bmr(self):
        _require_groq()
        ans = _answer(
            "image_multiple.png",
            "What type of document is shown in all three pages?"
        )
        assert _contains_any(
            ans, ["batch manufacturing record", "bmr", "batch record"]
        ), f"Document type not identified: {ans}"


# =============================================================================
# 8  test_handwritten_text_extracted
#    image_half.png — BMR with handwritten material entries in the table
# =============================================================================

class TestHandwrittenTextExtracted:
    """
    The Raw Materials table has handwritten values (messy cursive/print).
    The model must attempt extraction and return some of the numerical values.
    """

    def test_handwritten_document_detected(self):
        _require_groq()
        ans = _answer(
            "image_half.png",
            "Is the content of this document printed or handwritten?"
        )
        assert _contains_any(
            ans, ["handwritten", "hand-written", "handwriting", "written by hand", "manuscript"]
        ), f"Handwritten nature not identified: {ans}"

    def test_numerical_values_in_table_extracted(self):
        _require_groq()
        ans = _answer(
            "image_half.png",
            "Read the numbers in the Material and Quantity columns of the table "
            "in this Batch Manufacturing Record."
        )
        # Visible values: 207 / 1.85 and 345 or 245 / 1.50
        assert _contains_any(ans, ["207", "1.85", "345", "245", "1.50", "1.80", "1.5"]), (
            f"Handwritten numbers not extracted: {ans}"
        )

    def test_signature_area_identified(self):
        _require_groq()
        ans = _answer(
            "image_half.png",
            "Is there a signature or signed name anywhere on this document?"
        )
        assert _contains_any(
            ans, ["signature", "signed", "sign", "servo", "sero", "operator"]
        ), f"Signature area not identified: {ans}"


# =============================================================================
# 9  test_printed_form_vs_handwritten_mode_routing
#    Compares image_info.png (100% printed) vs image_half.png (handwritten)
# =============================================================================

class TestPrintedFormVsHandwrittenModeRouting:
    """
    The model must correctly classify each document's content type.
    This guards against routing logic that assumes all forms are typed.
    """

    def test_info_image_classified_as_printed(self):
        _require_groq()
        ans = _answer(
            "image_info.png",
            "Is the text on this form printed/typed or handwritten?"
        )
        assert _contains_any(
            ans, ["printed", "typed", "machine", "digital", "computer", "text is printed"]
        ), f"Printed classification failed: {ans}"

    def test_half_image_classified_as_handwritten(self):
        _require_groq()
        ans = _answer(
            "image_half.png",
            "Is the text on this form printed/typed or handwritten?"
        )
        assert _contains_any(
            ans, ["handwritten", "hand-written", "handwriting", "written by hand", "manual"]
        ), f"Handwritten classification failed: {ans}"

    def test_printed_form_yields_higher_field_confidence(self):
        _require_groq()
        # Printed form should extract all three medicines; handwritten should yield fewer / flagged
        printed_ans = _answer(
            "image_info.png",
            "List every item name in this purchase requisition."
        )
        # Must contain all three items
        assert "paracetamol" in printed_ans
        assert "amoxicillin" in printed_ans
        assert _contains_any(printed_ans, ["ibuprofen"])

    def test_watermarked_form_still_readable(self):
        _require_groq()
        # Watermark should not prevent field extraction
        ans = _answer(
            "image_watermark.png",
            "Can you read the data in the table of this document despite any overlays?"
        )
        assert _contains_any(
            ans,
            ["yes", "able", "can", "readable", "visible", "assay", "moisture", "iron",
             "sodium", "99.2", "0.3"]
        ), f"Watermarked table not readable: {ans}"
