"""Fill fillable form fields in a PDF from a JSON mapping.

Usage:
    python fill_fillable_fields.py <pdf_file> <output_file> <field_values_json>

Examples:
    python fill_fillable_fields.py form.pdf filled.pdf '{"Name": "John", "Date": "2025-01-01"}'
    python fill_fillable_fields.py form.pdf filled.pdf --json-file values.json
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("Error: pypdf is required. Install with: pip install pypdf")
    sys.exit(1)


def fill_fields(pdf_path: str, output_path: str, field_values: dict, flatten: bool = False) -> dict:
    """Fill form fields in a PDF.

    Args:
        pdf_path: Path to the input PDF with form fields
        output_path: Path for the output filled PDF
        field_values: Dict mapping field names to values
        flatten: If True, flatten form fields (makes them non-editable)

    Returns:
        Dict with results: filled_count, skipped_fields, errors
    """
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    # Clone all pages
    for page in reader.pages:
        writer.add_page(page)

    # Copy metadata
    if reader.metadata:
        writer.add_metadata(reader.metadata)

    available_fields = set()
    if reader.get_fields():
        available_fields = set(reader.get_fields().keys())

    filled = []
    skipped = []
    not_found = []

    for field_name, value in field_values.items():
        if field_name in available_fields:
            try:
                writer.update_page_form_field_values(
                    writer.pages[0],  # update_page_form_field_values works across all pages
                    {field_name: value},
                )
                filled.append(field_name)
            except Exception as e:
                skipped.append({"field": field_name, "error": str(e)})
        else:
            not_found.append(field_name)

    with open(output_path, "wb") as f:
        writer.write(f)

    return {
        "output": output_path,
        "filled_count": len(filled),
        "filled_fields": filled,
        "not_found": not_found,
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Fill PDF form fields")
    parser.add_argument("pdf_file", help="Input PDF with form fields")
    parser.add_argument("output_file", help="Output filled PDF")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("field_values", nargs="?", help="JSON string of field:value pairs")
    group.add_argument("--json-file", help="Path to JSON file with field:value pairs")
    parser.add_argument("--flatten", action="store_true", help="Flatten form fields")
    args = parser.parse_args()

    if args.json_file:
        with open(args.json_file) as f:
            values = json.load(f)
    else:
        values = json.loads(args.field_values)

    result = fill_fields(args.pdf_file, args.output_file, values, flatten=args.flatten)

    print(f"Filled {result['filled_count']} field(s)")
    if result["not_found"]:
        print(f"Fields not found in PDF: {result['not_found']}")
    if result["skipped"]:
        print(f"Skipped due to errors: {result['skipped']}")


if __name__ == "__main__":
    main()
