"""Create validation images with annotated bounding boxes for PDF QA.

Renders PDF pages with visual overlays showing text regions, form fields,
and table boundaries for quality assurance.

Usage:
    python create_validation_image.py <pdf_file> <output_dir>
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from pdf2image import convert_from_path
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: pdf2image and Pillow are required.")
    print("Install with: pip install pdf2image Pillow")
    sys.exit(1)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def create_validation_image(
    pdf_path: str,
    output_dir: str,
    dpi: int = 150,
    show_text: bool = True,
    show_tables: bool = True,
) -> list:
    """Create annotated validation images from a PDF.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save validation images
        dpi: DPI for rendering
        show_text: Whether to annotate text bounding boxes
        show_tables: Whether to annotate table bounding boxes

    Returns:
        List of output image paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Render pages to images
    images = convert_from_path(pdf_path, dpi=dpi)
    output_paths = []

    # Get structural info from pdfplumber if available
    plumber_pages = []
    if pdfplumber and (show_text or show_tables):
        try:
            pdf = pdfplumber.open(pdf_path)
            plumber_pages = pdf.pages
        except Exception:
            plumber_pages = []

    for i, image in enumerate(images):
        draw = ImageDraw.Draw(image)
        page_width, page_height = image.size

        if i < len(plumber_pages):
            page = plumber_pages[i]
            # Scale factors from PDF points to image pixels
            scale_x = page_width / float(page.width)
            scale_y = page_height / float(page.height)

            # Draw table bounding boxes in blue
            if show_tables:
                tables = page.find_tables()
                for table in tables:
                    bbox = table.bbox  # (x0, top, x1, bottom)
                    rect = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y,
                    ]
                    draw.rectangle(rect, outline="blue", width=2)
                    draw.text(
                        (rect[0], rect[1] - 12),
                        f"Table ({len(table.rows)} rows)",
                        fill="blue",
                    )

            # Draw text line bounding boxes in green (lighter, for reference)
            if show_text:
                words = page.extract_words()
                for word in words[:200]:  # Limit to avoid clutter
                    rect = [
                        word["x0"] * scale_x,
                        word["top"] * scale_y,
                        word["x1"] * scale_x,
                        word["bottom"] * scale_y,
                    ]
                    draw.rectangle(rect, outline=(0, 200, 0, 80), width=1)

        # Add page label
        draw.text((10, 10), f"Page {i + 1}", fill="red")

        output_path = os.path.join(output_dir, f"validation_page_{i+1}.png")
        image.save(output_path)
        output_paths.append(output_path)
        print(f"Created validation image: {output_path}")

    if plumber_pages:
        try:
            pdf.close()
        except Exception:
            pass

    print(f"Created {len(output_paths)} validation images")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Create PDF validation images with annotations")
    parser.add_argument("pdf_file", help="Input PDF file")
    parser.add_argument("output_dir", help="Output directory for validation images")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    parser.add_argument("--no-text", action="store_true", help="Skip text bounding boxes")
    parser.add_argument("--no-tables", action="store_true", help="Skip table bounding boxes")
    args = parser.parse_args()

    create_validation_image(
        args.pdf_file,
        args.output_dir,
        dpi=args.dpi,
        show_text=not args.no_text,
        show_tables=not args.no_tables,
    )


if __name__ == "__main__":
    main()
