"""Simplify adjacent tracked changes from the same author in DOCX XML.

When multiple edits are made by the same author in sequence, Word creates
separate <w:ins>/<w:del> elements for each change. This helper merges
adjacent tracked changes from the same author for readability.

Usage:
    from helpers.simplify_redlines import simplify_redlines
    count, message = simplify_redlines("unpacked/")
"""

from pathlib import Path
from xml.dom import minidom


def simplify_redlines(unpacked_dir: str) -> tuple:
    """Simplify adjacent tracked changes from the same author.

    Args:
        unpacked_dir: Path to the unpacked DOCX directory

    Returns:
        Tuple of (simplify_count, message)
    """
    unpacked_path = Path(unpacked_dir)
    doc_xml = unpacked_path / "word" / "document.xml"

    if not doc_xml.exists():
        return 0, "No document.xml found"

    total_simplified = 0

    xml_files = [doc_xml]
    word_dir = unpacked_path / "word"
    xml_files.extend(word_dir.glob("header*.xml"))
    xml_files.extend(word_dir.glob("footer*.xml"))

    for xml_file in xml_files:
        simplified = _simplify_in_file(xml_file)
        total_simplified += simplified

    return total_simplified, f"Simplified {total_simplified} tracked changes"


def _simplify_in_file(xml_file: Path) -> int:
    """Simplify adjacent tracked changes in a single XML file."""
    try:
        dom = minidom.parse(str(xml_file))
    except Exception:
        return 0

    simplified_count = 0

    for para in dom.getElementsByTagName("w:p"):
        children = [
            child for child in para.childNodes
            if child.nodeType == child.ELEMENT_NODE
            and child.tagName in ("w:ins", "w:del")
        ]

        i = 0
        while i < len(children) - 1:
            current = children[i]
            next_elem = children[i + 1]

            # Check if both are the same type and same author
            if (
                current.tagName == next_elem.tagName
                and current.getAttribute("w:author") == next_elem.getAttribute("w:author")
            ):
                # Move all child nodes from next_elem into current
                while next_elem.firstChild:
                    current.appendChild(next_elem.firstChild)

                # Remove next_elem from parent
                para.removeChild(next_elem)
                children.pop(i + 1)
                simplified_count += 1
                continue  # Check new next element

            i += 1

    if simplified_count > 0:
        xml_file.write_bytes(dom.toxml(encoding="UTF-8"))

    return simplified_count
