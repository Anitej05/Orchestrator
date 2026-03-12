"""Merge adjacent runs with identical formatting in DOCX XML.

When Word saves a document, it often splits text into many small <w:r> (run)
elements even when they share the same formatting. This makes XML editing
harder because a single word might span multiple runs.

This helper merges adjacent <w:r> elements that have identical <w:rPr>
(run properties), making the XML cleaner and easier to edit.

Usage:
    from helpers.merge_runs import merge_runs
    count, message = merge_runs("unpacked/")
"""

from pathlib import Path
from xml.dom import minidom


def merge_runs(unpacked_dir: str) -> tuple:
    """Merge adjacent runs with identical formatting in all DOCX XML files.

    Args:
        unpacked_dir: Path to the unpacked DOCX directory

    Returns:
        Tuple of (merge_count, message)
    """
    unpacked_path = Path(unpacked_dir)
    doc_xml = unpacked_path / "word" / "document.xml"

    if not doc_xml.exists():
        return 0, "No document.xml found"

    total_merged = 0

    # Process document.xml and any header/footer files
    xml_files = [doc_xml]
    word_dir = unpacked_path / "word"
    xml_files.extend(word_dir.glob("header*.xml"))
    xml_files.extend(word_dir.glob("footer*.xml"))

    for xml_file in xml_files:
        merged = _merge_runs_in_file(xml_file)
        total_merged += merged

    return total_merged, f"Merged {total_merged} adjacent runs"


def _merge_runs_in_file(xml_file: Path) -> int:
    """Merge adjacent runs in a single XML file."""
    try:
        dom = minidom.parse(str(xml_file))
    except Exception:
        return 0

    merged_count = 0

    # Find all paragraphs
    for para in dom.getElementsByTagName("w:p"):
        runs = [
            child for child in para.childNodes
            if child.nodeType == child.ELEMENT_NODE and child.tagName == "w:r"
        ]

        i = 0
        while i < len(runs) - 1:
            current_run = runs[i]
            next_run = runs[i + 1]

            # Check if runs have identical formatting
            current_rpr = _get_rpr_text(current_run)
            next_rpr = _get_rpr_text(next_run)

            if current_rpr == next_rpr:
                # Merge text content from next_run into current_run
                current_text = _get_text_element(current_run)
                next_text = _get_text_element(next_run)

                if current_text is not None and next_text is not None:
                    current_value = current_text.firstChild.nodeValue if current_text.firstChild else ""
                    next_value = next_text.firstChild.nodeValue if next_text.firstChild else ""

                    # Update current run's text
                    if current_text.firstChild:
                        current_text.firstChild.nodeValue = current_value + next_value
                    else:
                        current_text.appendChild(dom.createTextNode(current_value + next_value))

                    # Preserve xml:space if either has whitespace
                    combined = current_value + next_value
                    if combined.startswith(" ") or combined.endswith(" ") or "  " in combined:
                        current_text.setAttribute("xml:space", "preserve")

                    # Remove next run
                    para.removeChild(next_run)
                    runs.pop(i + 1)
                    merged_count += 1
                    continue  # Don't increment — check the new next run

            i += 1

    if merged_count > 0:
        xml_file.write_bytes(dom.toxml(encoding="UTF-8"))

    return merged_count


def _get_rpr_text(run_element) -> str:
    """Get a normalized string representation of a run's <w:rPr> formatting."""
    for child in run_element.childNodes:
        if child.nodeType == child.ELEMENT_NODE and child.tagName == "w:rPr":
            return child.toxml()
    return ""  # No formatting = default formatting


def _get_text_element(run_element):
    """Get the <w:t> element from a run, if present."""
    for child in run_element.childNodes:
        if child.nodeType == child.ELEMENT_NODE and child.tagName == "w:t":
            return child
    return None
