"""Convert PDF pages to PNG images for visual inspection or processing.

Usage:
    python convert_pdf_to_images.py <input_pdf> <output_dir> [--max-dim N] [--dpi N]

Examples:
    python convert_pdf_to_images.py document.pdf output/
    python convert_pdf_to_images.py document.pdf output/ --max-dim 1500 --dpi 300
"""

import argparse
import os
import sys

try:
    from pdf2image import convert_from_path
except ImportError:
    print("Error: pdf2image is required. Install with: pip install pdf2image")
    print("Also requires poppler: https://poppler.freedesktop.org/")
    sys.exit(1)


def convert(pdf_path: str, output_dir: str, max_dim: int = 1000, dpi: int = 200) -> list:
    """Convert PDF pages to PNG images.

    Args:
        pdf_path: Path to the input PDF
        output_dir: Directory to save images
        max_dim: Maximum dimension (width or height) for output images
        dpi: DPI for rendering

    Returns:
        List of output image paths
    """
    os.makedirs(output_dir, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=dpi)
    output_paths = []

    for i, image in enumerate(images):
        width, height = image.size
        if width > max_dim or height > max_dim:
            scale_factor = min(max_dim / width, max_dim / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height))

        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        image.save(image_path)
        output_paths.append(image_path)
        print(f"Saved page {i+1} as {image_path} (size: {image.size})")

    print(f"Converted {len(images)} pages to PNG images")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to images")
    parser.add_argument("input_pdf", help="Input PDF file")
    parser.add_argument("output_dir", help="Output directory for images")
    parser.add_argument("--max-dim", type=int, default=1000, help="Max dimension (default: 1000)")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200)")
    args = parser.parse_args()

    convert(args.input_pdf, args.output_dir, max_dim=args.max_dim, dpi=args.dpi)


if __name__ == "__main__":
    main()
