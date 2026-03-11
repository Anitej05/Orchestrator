"""Check and list fillable form fields in a PDF.

Usage:
    python check_fillable_fields.py <pdf_file>

Example:
    python check_fillable_fields.py form.pdf
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("Error: pypdf is required. Install with: pip install pypdf")
    sys.exit(1)


def check_fields(pdf_path: str) -> list:
    """List all fillable form fields in a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with field info: name, type, value, rect, page
    """
    reader = PdfReader(pdf_path)
    fields = []

    if reader.get_fields():
        for field_name, field_data in reader.get_fields().items():
            field_info = {
                "name": field_name,
                "type": str(field_data.get("/FT", "Unknown")),
                "value": field_data.get("/V", ""),
                "default_value": field_data.get("/DV", ""),
                "flags": field_data.get("/Ff", 0),
            }

            # Map field type codes to human-readable names
            type_map = {
                "/Tx": "Text",
                "/Btn": "Button/Checkbox",
                "/Ch": "Choice/Dropdown",
                "/Sig": "Signature",
            }
            field_info["type_name"] = type_map.get(field_info["type"], field_info["type"])

            # Check for options (dropdowns)
            if "/Opt" in field_data:
                field_info["options"] = field_data["/Opt"]

            fields.append(field_info)

    return fields


def main():
    parser = argparse.ArgumentParser(description="List fillable PDF form fields")
    parser.add_argument("pdf_file", help="PDF file to check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not Path(args.pdf_file).exists():
        print(f"Error: {args.pdf_file} not found")
        sys.exit(1)

    fields = check_fields(args.pdf_file)

    if not fields:
        print("No fillable form fields found.")
        return

    if args.json:
        print(json.dumps(fields, indent=2, default=str))
    else:
        print(f"Found {len(fields)} fillable field(s):\n")
        for f in fields:
            print(f"  [{f['type_name']}] {f['name']}")
            if f["value"]:
                print(f"    Current value: {f['value']}")
            if f.get("options"):
                print(f"    Options: {f['options']}")


if __name__ == "__main__":
    main()
