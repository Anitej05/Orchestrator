"""Pack a directory into a DOCX, PPTX, or XLSX file.

Validates with auto-repair, condenses XML formatting, and creates the Office file.

Usage:
    python pack.py <input_directory> <output_file> [--original <file>] [--validate true|false]

Examples:
    python pack.py unpacked/ output.docx --original input.docx
    python pack.py unpacked/ output.pptx --validate false
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.dom import minidom

from .validate import DOCXSchemaValidator, PPTXSchemaValidator, RedliningValidator


SMART_QUOTE_UNESCAPE = {
    "&#x201C;": "\u201c",
    "&#x201D;": "\u201d",
    "&#x2018;": "\u2018",
    "&#x2019;": "\u2019",
}


def pack(
    input_directory: str,
    output_file: str,
    original_file: str = None,
    validate: bool = True,
) -> tuple:
    """Pack a directory of XML files back into an Office file.

    Args:
        input_directory: Directory containing unpacked Office content
        output_file: Output Office file (.docx, .pptx, .xlsx)
        original_file: Original file for validation comparison
        validate: Whether to run validation with auto-repair

    Returns:
        Tuple of (None, message_string)
    """
    input_dir = Path(input_directory)
    output_path = Path(output_file)
    suffix = output_path.suffix.lower()

    if not input_dir.is_dir():
        return None, f"Error: {input_dir} is not a directory"

    if suffix not in {".docx", ".pptx", ".xlsx"}:
        return None, f"Error: {output_file} must be a .docx, .pptx, or .xlsx file"

    # Run validation if enabled and original file is provided
    if validate and original_file:
        original_path = Path(original_file)
        if original_path.exists():
            success, output = _run_validation(input_dir, original_path, suffix)
            if output:
                print(output)
            if not success:
                return None, f"Error: Validation failed for {input_dir}"

    # Pack into a ZIP (Office file)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_content_dir = Path(temp_dir) / "content"
        shutil.copytree(input_dir, temp_content_dir)

        # Condense XML and restore smart quotes
        for pattern in ["*.xml", "*.rels"]:
            for xml_file in temp_content_dir.rglob(pattern):
                _restore_smart_quotes(xml_file)
                _condense_xml(xml_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in temp_content_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(temp_content_dir))

    return None, f"Successfully packed {input_dir} to {output_file}"


def _run_validation(
    unpacked_dir: Path,
    original_file: Path,
    suffix: str,
) -> tuple:
    """Run schema + redlining validation with auto-repair."""
    output_lines = []
    validators = []

    if suffix == ".docx":
        validators = [
            DOCXSchemaValidator(unpacked_dir, original_file),
            RedliningValidator(unpacked_dir, original_file, author="Orbimesh"),
        ]
    elif suffix == ".pptx":
        validators = [PPTXSchemaValidator(unpacked_dir, original_file)]

    if not validators:
        return True, None

    total_repairs = sum(v.repair() for v in validators)
    if total_repairs:
        output_lines.append(f"Auto-repaired {total_repairs} issue(s)")

    success = all(v.validate() for v in validators)

    if success:
        output_lines.append("All validations PASSED!")

    return success, "\n".join(output_lines) if output_lines else None


def _restore_smart_quotes(xml_file: Path) -> None:
    """Restore smart quote entities back to Unicode characters."""
    try:
        content = xml_file.read_text(encoding="utf-8")
        for entity, char in SMART_QUOTE_UNESCAPE.items():
            content = content.replace(entity, char)
        xml_file.write_text(content, encoding="utf-8")
    except Exception:
        pass


def _condense_xml(xml_file: Path) -> None:
    """Condense pretty-printed XML back to compact form for packaging."""
    try:
        with open(xml_file, encoding="utf-8") as f:
            dom = minidom.parse(f)

        # Remove whitespace-only text nodes (except in text elements like w:t)
        for element in dom.getElementsByTagName("*"):
            # Skip text content elements
            if element.tagName.endswith(":t"):
                continue

            for child in list(element.childNodes):
                if (
                    child.nodeType == child.TEXT_NODE
                    and child.nodeValue
                    and child.nodeValue.strip() == ""
                ) or child.nodeType == child.COMMENT_NODE:
                    element.removeChild(child)

        xml_file.write_bytes(dom.toxml(encoding="UTF-8"))
    except Exception as e:
        print(f"ERROR: Failed to parse {xml_file.name}: {e}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Pack a directory into a DOCX, PPTX, or XLSX file"
    )
    parser.add_argument("input_directory", help="Unpacked Office document directory")
    parser.add_argument("output_file", help="Output Office file (.docx/.pptx/.xlsx)")
    parser.add_argument(
        "--original",
        help="Original file for validation comparison",
    )
    parser.add_argument(
        "--validate",
        type=lambda x: x.lower() == "true",
        default=True,
        metavar="true|false",
        help="Run validation with auto-repair (default: true)",
    )
    args = parser.parse_args()

    _, message = pack(
        args.input_directory,
        args.output_file,
        original_file=args.original,
        validate=args.validate,
    )
    print(message)

    if "Error" in message:
        sys.exit(1)


if __name__ == "__main__":
    main()
