"""
PDF Agent - Utilities

Helper functions for PDF processing based on anthropics/skills PDF patterns.
Uses pypdf, pdfplumber, and reportlab for comprehensive PDF operations.
"""

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("agents.pdf_agent.utils")


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_pypdf(file_path: str) -> str:
    """Extract text from PDF using pypdf (fast, basic)."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(texts)


def extract_text_pdfplumber(file_path: str) -> str:
    """Extract text from PDF using pdfplumber (preserves layout)."""
    import pdfplumber

    texts = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                texts.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(texts)


def extract_text_with_vision(file_path: str) -> str:
    """
    Fallback for image-only / scanned PDFs.
    Renders each page via PyMuPDF and sends to Groq vision model.
    """
    try:
        import fitz  # PyMuPDF
        from backend.tools.image_tools import analyze_image

        doc = fitz.open(file_path)
        page_texts: list = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, page in enumerate(doc):
                img_path = Path(tmpdir) / f"page_{i + 1}.png"
                pix = page.get_pixmap(dpi=150)
                pix.save(str(img_path))

                result = analyze_image.invoke({
                    "image_path": str(img_path),
                    "query": (
                        "Extract and describe all text, data, tables, and content "
                        "visible on this page in full detail."
                    ),
                })

                if "error" not in result:
                    page_texts.append(f"[Page {i + 1}]\n{result['answer']}")
                else:
                    logger.warning(
                        f"Vision extraction failed on page {i + 1}: "
                        f"{result.get('error')}"
                    )
                    page_texts.append(f"[Page {i + 1}] (vision extraction failed)")

        return "\n\n".join(page_texts)

    except Exception as e:
        logger.error(f"Vision fallback failed for {file_path}: {e}")
        return ""


# ============================================================================
# TABLE EXTRACTION
# ============================================================================

def extract_tables(file_path: str) -> List[Dict[str, Any]]:
    """Extract all tables from a PDF using pdfplumber."""
    import pdfplumber

    all_tables = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                if table:
                    # First row as headers, rest as data
                    headers = table[0] if table else []
                    rows = table[1:] if len(table) > 1 else []
                    all_tables.append({
                        "page": i + 1,
                        "table_index": j + 1,
                        "headers": headers,
                        "rows": rows,
                        "row_count": len(rows),
                    })
    return all_tables


def tables_to_dataframes(file_path: str):
    """Extract tables and return as pandas DataFrames."""
    import pdfplumber
    import pandas as pd

    dataframes = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    dataframes.append(df)
    return dataframes


# ============================================================================
# MERGE / SPLIT
# ============================================================================

def merge_pdfs(file_paths: List[str], output_path: str) -> str:
    """Merge multiple PDF files into one."""
    from pypdf import PdfWriter, PdfReader

    writer = PdfWriter()
    for pdf_file in file_paths:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            writer.add_page(page)

    with open(output_path, "wb") as output:
        writer.write(output)

    logger.info(f"Merged {len(file_paths)} PDFs into {output_path}")
    return output_path


def split_pdf(
    file_path: str,
    output_dir: str,
    page_ranges: Optional[List[Tuple[int, int]]] = None,
) -> List[str]:
    """
    Split PDF into separate files.

    Args:
        file_path: Source PDF path
        output_dir: Directory for output files
        page_ranges: Optional list of (start, end) tuples (1-indexed, inclusive).
                     If None, splits into individual pages.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(file_path)
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    stem = Path(file_path).stem

    if page_ranges:
        for idx, (start, end) in enumerate(page_ranges):
            writer = PdfWriter()
            for page_num in range(start - 1, min(end, len(reader.pages))):
                writer.add_page(reader.pages[page_num])
            out_path = os.path.join(output_dir, f"{stem}_pages_{start}-{end}.pdf")
            with open(out_path, "wb") as f:
                writer.write(f)
            output_files.append(out_path)
    else:
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            out_path = os.path.join(output_dir, f"{stem}_page_{i + 1}.pdf")
            with open(out_path, "wb") as f:
                writer.write(f)
            output_files.append(out_path)

    logger.info(f"Split {file_path} into {len(output_files)} files")
    return output_files


# ============================================================================
# CREATE
# ============================================================================

def create_pdf_from_text(content: str, file_path: str) -> str:
    """Create a basic PDF from text content using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for paragraph in content.split("\n"):
        if paragraph.strip():
            # Escape XML special characters for reportlab
            safe_text = (
                paragraph
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(Paragraph(safe_text, styles["Normal"]))
            story.append(Spacer(1, 12))

    doc.build(story)
    logger.info(f"Created PDF: {file_path}")
    return file_path


def create_pdf_with_structure(
    sections: List[Dict[str, Any]], file_path: str
) -> str:
    """
    Create a structured PDF with headings and body text.

    Each section: {"heading": str, "body": str, "level": int}
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    style_map = {
        1: "Title",
        2: "Heading1",
        3: "Heading2",
    }

    for section in sections:
        heading = section.get("heading", "")
        body = section.get("body", "")
        level = section.get("level", 2)

        if heading:
            style_name = style_map.get(level, "Heading2")
            safe_heading = (
                heading
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(Paragraph(safe_heading, styles[style_name]))
            story.append(Spacer(1, 6))

        if body:
            for para in body.split("\n"):
                if para.strip():
                    safe_para = (
                        para
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    story.append(Paragraph(safe_para, styles["Normal"]))
                    story.append(Spacer(1, 6))

        if section.get("page_break"):
            story.append(PageBreak())

    doc.build(story)
    logger.info(f"Created structured PDF: {file_path}")
    return file_path


# ============================================================================
# OCR
# ============================================================================

def ocr_pdf(file_path: str) -> str:
    """OCR a scanned PDF using pytesseract + pdf2image."""
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(file_path)
        texts = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            texts.append(f"[Page {i + 1}]\n{text}")

        return "\n\n".join(texts)
    except ImportError as e:
        logger.warning(f"OCR dependencies not available: {e}")
        # Fall back to vision extraction
        return extract_text_with_vision(file_path)


# ============================================================================
# WATERMARK / ROTATE / METADATA / PASSWORD
# ============================================================================

def add_watermark(
    source_path: str, watermark_path: str, output_path: str
) -> str:
    """Apply a watermark PDF to all pages of the source PDF."""
    from pypdf import PdfReader, PdfWriter

    watermark = PdfReader(watermark_path).pages[0]
    reader = PdfReader(source_path)
    writer = PdfWriter()

    for page in reader.pages:
        page.merge_page(watermark)
        writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)

    logger.info(f"Added watermark to {output_path}")
    return output_path


def rotate_pages(
    file_path: str,
    output_path: str,
    pages: Optional[List[int]] = None,
    angle: int = 90,
) -> str:
    """Rotate specific pages (or all) by the given angle."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(file_path)
    writer = PdfWriter()

    for i, page in enumerate(reader.pages):
        if pages is None or (i + 1) in pages:
            page.rotate(angle)
        writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)

    logger.info(f"Rotated pages in {output_path}")
    return output_path


def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract PDF metadata."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    meta = reader.metadata or {}
    return {
        "title": getattr(meta, "title", None),
        "author": getattr(meta, "author", None),
        "subject": getattr(meta, "subject", None),
        "creator": getattr(meta, "creator", None),
        "producer": getattr(meta, "producer", None),
        "page_count": len(reader.pages),
        "is_encrypted": reader.is_encrypted,
    }


def password_protect(
    file_path: str,
    output_path: str,
    user_password: str,
    owner_password: Optional[str] = None,
) -> str:
    """Encrypt a PDF with passwords."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(file_path)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    writer.encrypt(user_password, owner_password or user_password)

    with open(output_path, "wb") as f:
        writer.write(f)

    logger.info(f"Password-protected PDF saved to {output_path}")
    return output_path


def extract_images_from_pdf(
    file_path: str, output_dir: str
) -> List[str]:
    """Extract embedded images from a PDF."""
    from pypdf import PdfReader

    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(file_path)
    extracted = []
    stem = Path(file_path).stem

    img_counter = 0
    for page_num, page in enumerate(reader.pages):
        for image_obj in page.images:
            img_counter += 1
            ext = Path(image_obj.name).suffix or ".png"
            out_path = os.path.join(
                output_dir, f"{stem}_p{page_num + 1}_img{img_counter}{ext}"
            )
            with open(out_path, "wb") as f:
                f.write(image_obj.data)
            extracted.append(out_path)

    logger.info(f"Extracted {len(extracted)} images from {file_path}")
    return extracted


# ============================================================================
# CANVAS DISPLAY HELPERS
# ============================================================================

def get_file_base64(file_path: str) -> str:
    """Convert file to base64 for display."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_pdf_canvas_display(
    file_path: str, title: str
) -> Dict[str, Any]:
    """Create PDF canvas display payload for frontend."""
    import time

    pdf_base64 = get_file_base64(file_path)
    return {
        "canvas_type": "pdf",
        "title": title,
        "pdf_data": f"data:application/pdf;base64,{pdf_base64}",
        "file_path": file_path,
        "timestamp": int(time.time() * 1000),
        "no_cache": True,
    }
