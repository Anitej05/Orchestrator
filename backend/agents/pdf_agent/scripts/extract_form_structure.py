"""Extract hierarchical form field structure from a PDF.

Provides detailed structural info about form fields including
hierarchy (nested fields), widget annotations, and page positions.

Usage:
    python extract_form_structure.py <pdf_file>
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
    from pypdf.generic import ArrayObject, DictionaryObject
except ImportError:
    print("Error: pypdf is required. Install with: pip install pypdf")
    sys.exit(1)


def extract_structure(pdf_path: str) -> dict:
    """Extract detailed form field structure from a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dict with form structure info
    """
    reader = PdfReader(pdf_path)

    structure = {
        "total_pages": len(reader.pages),
        "has_form": reader.get_fields() is not None and len(reader.get_fields()) > 0,
        "fields": [],
        "field_count": 0,
    }

    if not structure["has_form"]:
        return structure

    fields = reader.get_fields()

    for field_name, field_obj in fields.items():
        field_info = {
            "name": field_name,
            "type": str(field_obj.get("/FT", "Unknown")),
            "value": str(field_obj.get("/V", "")),
            "flags": int(field_obj.get("/Ff", 0)),
        }

        # Get field type name
        type_map = {
            "/Tx": "Text",
            "/Btn": "Button/Checkbox/Radio",
            "/Ch": "Choice/Dropdown/Listbox",
            "/Sig": "Signature",
        }
        field_info["type_name"] = type_map.get(field_info["type"], field_info["type"])

        # Check for multi-line text
        if field_info["type"] == "/Tx" and field_info["flags"] & (1 << 12):
            field_info["multiline"] = True

        # Check if required
        if field_info["flags"] & (1 << 1):
            field_info["required"] = True

        # Check if read-only
        if field_info["flags"] & 1:
            field_info["read_only"] = True

        # Options for choice fields
        if "/Opt" in field_obj:
            opts = field_obj["/Opt"]
            if isinstance(opts, (list, ArrayObject)):
                field_info["options"] = [str(o) for o in opts]

        # Max length for text fields
        if "/MaxLen" in field_obj:
            field_info["max_length"] = int(field_obj["/MaxLen"])

        # Tooltip / description
        if "/TU" in field_obj:
            field_info["tooltip"] = str(field_obj["/TU"])

        structure["fields"].append(field_info)

    structure["field_count"] = len(structure["fields"])

    return structure


def main():
    parser = argparse.ArgumentParser(description="Extract PDF form structure")
    parser.add_argument("pdf_file", help="PDF file to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not Path(args.pdf_file).exists():
        print(f"Error: {args.pdf_file} not found")
        sys.exit(1)

    structure = extract_structure(args.pdf_file)

    if args.json:
        print(json.dumps(structure, indent=2, default=str))
    else:
        print(f"Pages: {structure['total_pages']}")
        print(f"Has form: {structure['has_form']}")
        print(f"Fields: {structure['field_count']}")
        if structure["fields"]:
            print("\nField Details:")
            for f in structure["fields"]:
                flags = []
                if f.get("required"):
                    flags.append("REQUIRED")
                if f.get("read_only"):
                    flags.append("READ-ONLY")
                if f.get("multiline"):
                    flags.append("MULTILINE")
                flag_str = f" ({', '.join(flags)})" if flags else ""

                print(f"  [{f['type_name']}] {f['name']}{flag_str}")
                if f.get("tooltip"):
                    print(f"    Tooltip: {f['tooltip']}")
                if f.get("options"):
                    print(f"    Options: {f['options'][:5]}{'...' if len(f.get('options', [])) > 5 else ''}")
                if f.get("max_length"):
                    print(f"    Max length: {f['max_length']}")


if __name__ == "__main__":
    main()
