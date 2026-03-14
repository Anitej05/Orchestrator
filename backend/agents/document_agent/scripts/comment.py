"""Add comments and replies to DOCX XML files.

Handles the boilerplate of updating comments.xml, [Content_Types].xml,
and relationship files. After running this script, add comment range markers
to document.xml manually.

Usage:
    python comment.py <unpacked_dir> <comment_id> <text> [--parent <id>] [--author <name>]

Examples:
    python comment.py unpacked/ 0 "This needs review"
    python comment.py unpacked/ 1 "I agree" --parent 0
    python comment.py unpacked/ 0 "Looks good" --author "Reviewer"
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.dom import minidom


COMMENTS_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def add_comment(
    unpacked_dir: str,
    comment_id: int,
    text: str,
    parent_id: int = None,
    author: str = "Orbimesh",
    date: str = None,
) -> tuple:
    """Add a comment to an unpacked DOCX.

    Args:
        unpacked_dir: Path to the unpacked DOCX directory
        comment_id: Unique comment ID (0-based integer)
        text: Comment text (pre-escaped XML)
        parent_id: Parent comment ID for replies
        author: Author name
        date: ISO date string (default: current UTC time)

    Returns:
        Tuple of (success, message)
    """
    unpacked = Path(unpacked_dir)
    word_dir = unpacked / "word"

    if not word_dir.exists():
        return False, "Error: word/ directory not found"

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure comments.xml exists
    comments_xml = word_dir / "comments.xml"
    if not comments_xml.exists():
        _create_comments_file(comments_xml)
        _add_comments_relationship(unpacked)
        _add_comments_content_type(unpacked)

    # Add the comment
    _add_comment_entry(comments_xml, comment_id, text, author, date, parent_id)

    # Generate instructions for marker placement
    marker_instructions = _get_marker_instructions(comment_id, parent_id)

    return True, f"Added comment {comment_id}: '{text[:50]}...'\n\n{marker_instructions}"


def _create_comments_file(filepath: Path) -> None:
    """Create a new comments.xml file."""
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
</w:comments>"""
    filepath.write_text(xml, encoding="utf-8")


def _add_comment_entry(
    comments_xml: Path,
    comment_id: int,
    text: str,
    author: str,
    date: str,
    parent_id: int = None,
) -> None:
    """Add a comment entry to comments.xml."""
    dom = minidom.parse(str(comments_xml))
    comments = dom.getElementsByTagName("w:comments")[0]

    # Create <w:comment>
    comment = dom.createElement("w:comment")
    comment.setAttribute("w:id", str(comment_id))
    comment.setAttribute("w:author", author)
    comment.setAttribute("w:date", date)
    comment.setAttribute("w:initials", author[0].upper() if author else "O")

    # Create paragraph with text
    p = dom.createElement("w:p")
    r = dom.createElement("w:r")
    t = dom.createElement("w:t")
    t.setAttribute("xml:space", "preserve")
    t.appendChild(dom.createTextNode(text))
    r.appendChild(t)
    p.appendChild(r)
    comment.appendChild(p)

    comments.appendChild(comment)
    comments_xml.write_bytes(dom.toxml(encoding="UTF-8"))


def _add_comments_relationship(unpacked_dir: Path) -> None:
    """Add comments.xml relationship to document.xml.rels."""
    rels_file = unpacked_dir / "word" / "_rels" / "document.xml.rels"
    if not rels_file.exists():
        return

    try:
        dom = minidom.parse(str(rels_file))
        relationships = dom.getElementsByTagName("Relationships")[0]

        # Check if already exists
        for rel in dom.getElementsByTagName("Relationship"):
            if "comments" in rel.getAttribute("Type"):
                return  # Already exists

        # Find next rId
        import re
        max_id = 0
        for rel in dom.getElementsByTagName("Relationship"):
            match = re.match(r"rId(\d+)", rel.getAttribute("Id"))
            if match:
                max_id = max(max_id, int(match.group(1)))

        new_rel = dom.createElement("Relationship")
        new_rel.setAttribute("Id", f"rId{max_id + 1}")
        new_rel.setAttribute(
            "Type",
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
        )
        new_rel.setAttribute("Target", "comments.xml")
        relationships.appendChild(new_rel)

        rels_file.write_bytes(dom.toxml(encoding="UTF-8"))
    except Exception as e:
        print(f"Warning: Could not update rels: {e}", file=sys.stderr)


def _add_comments_content_type(unpacked_dir: Path) -> None:
    """Add comments content type to [Content_Types].xml."""
    ct_file = unpacked_dir / "[Content_Types].xml"
    if not ct_file.exists():
        return

    try:
        dom = minidom.parse(str(ct_file))
        types = dom.getElementsByTagName("Types")[0]

        # Check if already exists
        for override in dom.getElementsByTagName("Override"):
            if "comments.xml" in override.getAttribute("PartName"):
                return

        override = dom.createElement("Override")
        override.setAttribute("PartName", "/word/comments.xml")
        override.setAttribute(
            "ContentType",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"
        )
        types.appendChild(override)

        ct_file.write_bytes(dom.toxml(encoding="UTF-8"))
    except Exception:
        pass


def _get_marker_instructions(comment_id: int, parent_id: int = None) -> str:
    """Generate instructions for placing comment markers in document.xml."""
    if parent_id is not None:
        return f"""Add these markers to document.xml, nested inside parent comment {parent_id}'s markers:
  <w:commentRangeStart w:id="{comment_id}"/>
  ... commented text ...
  <w:commentRangeEnd w:id="{comment_id}"/>
  <w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="{comment_id}"/></w:r>"""
    else:
        return f"""Add these markers to document.xml around the commented text:
  <w:commentRangeStart w:id="{comment_id}"/>
  ... commented text ...
  <w:commentRangeEnd w:id="{comment_id}"/>
  <w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="{comment_id}"/></w:r>"""


def main():
    parser = argparse.ArgumentParser(description="Add comments to DOCX")
    parser.add_argument("unpacked_dir", help="Unpacked DOCX directory")
    parser.add_argument("comment_id", type=int, help="Comment ID (0-based)")
    parser.add_argument("text", help="Comment text")
    parser.add_argument("--parent", type=int, default=None, help="Parent comment ID for replies")
    parser.add_argument("--author", default="Orbimesh", help="Author name (default: Orbimesh)")
    args = parser.parse_args()

    success, msg = add_comment(
        args.unpacked_dir,
        args.comment_id,
        args.text,
        parent_id=args.parent,
        author=args.author,
    )
    print(msg)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
