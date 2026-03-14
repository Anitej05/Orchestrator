"""Accept all tracked changes in a DOCX document.

Resolves all insertions (keeps inserted text) and deletions (removes deleted text),
producing a clean document without any revision marks.

Usage:
    python accept_changes.py <unpacked_dir>

Example:
    python -m agents.scripts.office.unpack document.docx unpacked/
    python scripts/accept_changes.py unpacked/
    python -m agents.scripts.office.pack unpacked/ clean.docx
"""

import argparse
import sys
from pathlib import Path
from xml.dom import minidom


def accept_changes(unpacked_dir: str) -> tuple:
    """Accept all tracked changes in an unpacked DOCX.

    Args:
        unpacked_dir: Path to the unpacked DOCX directory

    Returns:
        Tuple of (changes_accepted, message)
    """
    unpacked = Path(unpacked_dir)
    doc_xml = unpacked / "word" / "document.xml"

    if not doc_xml.exists():
        return 0, "Error: word/document.xml not found"

    total_accepted = 0

    # Process document.xml and headers/footers
    xml_files = [doc_xml]
    word_dir = unpacked / "word"
    xml_files.extend(word_dir.glob("header*.xml"))
    xml_files.extend(word_dir.glob("footer*.xml"))

    for xml_file in xml_files:
        accepted = _accept_in_file(xml_file)
        total_accepted += accepted

    return total_accepted, f"Accepted {total_accepted} tracked changes"


def _accept_in_file(xml_file: Path) -> int:
    """Accept tracked changes in a single XML file."""
    try:
        dom = minidom.parse(str(xml_file))
    except Exception as e:
        print(f"Warning: Could not parse {xml_file.name}: {e}", file=sys.stderr)
        return 0

    accepted = 0
    modified = False

    # Accept insertions: unwrap <w:ins> elements (keep content)
    for ins_elem in list(dom.getElementsByTagName("w:ins")):
        parent = ins_elem.parentNode
        if parent is None:
            continue

        # Move all children out of <w:ins> to its parent
        while ins_elem.firstChild:
            parent.insertBefore(ins_elem.firstChild, ins_elem)

        parent.removeChild(ins_elem)
        accepted += 1
        modified = True

    # Accept deletions: remove <w:del> elements entirely (remove deleted text)
    for del_elem in list(dom.getElementsByTagName("w:del")):
        parent = del_elem.parentNode
        if parent is None:
            continue
        parent.removeChild(del_elem)
        accepted += 1
        modified = True

    # Remove paragraph-level deletion marks in <w:pPr><w:rPr><w:del/>
    for rpr in list(dom.getElementsByTagName("w:rPr")):
        for child in list(rpr.childNodes):
            if child.nodeType == child.ELEMENT_NODE and child.tagName == "w:del":
                rpr.removeChild(child)
                modified = True

    # Remove move-from/move-to markers
    for tag in ["w:moveFrom", "w:moveTo", "w:moveFromRangeStart", "w:moveFromRangeEnd",
                "w:moveToRangeStart", "w:moveToRangeEnd"]:
        for elem in list(dom.getElementsByTagName(tag)):
            parent = elem.parentNode
            if parent:
                if tag == "w:moveTo":
                    # Keep the content of moveTo (it's the new location)
                    while elem.firstChild:
                        parent.insertBefore(elem.firstChild, elem)
                parent.removeChild(elem)
                modified = True

    if modified:
        xml_file.write_bytes(dom.toxml(encoding="UTF-8"))

    return accepted


def main():
    parser = argparse.ArgumentParser(description="Accept all tracked changes in DOCX")
    parser.add_argument("unpacked_dir", help="Unpacked DOCX directory")
    args = parser.parse_args()

    count, msg = accept_changes(args.unpacked_dir)
    print(msg)

    if "Error" in msg:
        sys.exit(1)


if __name__ == "__main__":
    main()
