"""Remove orphaned slides, unreferenced media, and orphaned relationships.

Cleans an unpacked PPTX by removing files not referenced in the slide ID list.

Usage:
    python clean.py <unpacked_dir>
"""

import argparse
import re
import sys
from pathlib import Path
from xml.dom import minidom


def clean(unpacked_dir: str) -> tuple:
    """Clean orphaned files from an unpacked PPTX.

    Args:
        unpacked_dir: Path to the unpacked PPTX directory

    Returns:
        Tuple of (removed_count, message)
    """
    unpacked = Path(unpacked_dir)
    ppt_dir = unpacked / "ppt"
    slides_dir = ppt_dir / "slides"
    pres_xml = ppt_dir / "presentation.xml"

    if not pres_xml.exists():
        return 0, "Error: ppt/presentation.xml not found"

    removed = []

    # 1. Get slides referenced in <p:sldIdLst>
    referenced_slides = _get_referenced_slides(pres_xml, ppt_dir)

    # 2. Remove slides not in the list
    if slides_dir.exists():
        for slide_file in sorted(slides_dir.glob("slide*.xml")):
            if slide_file.name not in referenced_slides:
                slide_file.unlink()
                removed.append(f"Removed orphaned slide: {slide_file.name}")

                # Remove corresponding .rels
                rels_file = slides_dir / "_rels" / f"{slide_file.name}.rels"
                if rels_file.exists():
                    rels_file.unlink()
                    removed.append(f"Removed orphaned rels: {rels_file.name}")

    # 3. Remove unreferenced media
    media_dir = ppt_dir / "media"
    if media_dir.exists():
        referenced_media = _get_referenced_media(ppt_dir)
        for media_file in media_dir.iterdir():
            if media_file.is_file() and media_file.name not in referenced_media:
                media_file.unlink()
                removed.append(f"Removed unreferenced media: {media_file.name}")

    # 4. Clean up [Content_Types].xml
    _clean_content_types(unpacked, referenced_slides)

    msg = f"Cleaned {len(removed)} orphaned file(s)"
    if removed:
        msg += ":\n  " + "\n  ".join(removed)

    return len(removed), msg


def _get_referenced_slides(pres_xml: Path, ppt_dir: Path) -> set:
    """Get slide filenames referenced in the presentation's sldIdLst."""
    referenced = set()

    try:
        dom = minidom.parse(str(pres_xml))
        rels_file = ppt_dir / "_rels" / "presentation.xml.rels"

        # Build rId -> target mapping from rels
        rid_map = {}
        if rels_file.exists():
            rels_dom = minidom.parse(str(rels_file))
            for rel in rels_dom.getElementsByTagName("Relationship"):
                rid = rel.getAttribute("Id")
                target = rel.getAttribute("Target")
                rid_map[rid] = target

        # Get rIds from sldIdLst
        for sld_id in dom.getElementsByTagName("p:sldId"):
            rid = sld_id.getAttribute("r:id")
            if rid in rid_map:
                target = rid_map[rid]
                # Target is like "slides/slide1.xml"
                slide_name = Path(target).name
                referenced.add(slide_name)

    except Exception as e:
        print(f"Warning: Error reading presentation.xml: {e}", file=sys.stderr)

    return referenced


def _get_referenced_media(ppt_dir: Path) -> set:
    """Get media filenames referenced in any .rels file."""
    referenced = set()

    for rels_file in ppt_dir.rglob("*.rels"):
        try:
            dom = minidom.parse(str(rels_file))
            for rel in dom.getElementsByTagName("Relationship"):
                target = rel.getAttribute("Target")
                if "media/" in target:
                    media_name = Path(target).name
                    referenced.add(media_name)
        except Exception:
            continue

    return referenced


def _clean_content_types(unpacked_dir: Path, referenced_slides: set) -> None:
    """Remove Content_Types entries for deleted slides."""
    ct_file = unpacked_dir / "[Content_Types].xml"
    if not ct_file.exists():
        return

    try:
        dom = minidom.parse(str(ct_file))
        modified = False

        for override in list(dom.getElementsByTagName("Override")):
            part_name = override.getAttribute("PartName")
            # Check if this is a slide override
            match = re.match(r"/ppt/slides/(slide\d+\.xml)", part_name)
            if match and match.group(1) not in referenced_slides:
                override.parentNode.removeChild(override)
                modified = True

        if modified:
            ct_file.write_bytes(dom.toxml(encoding="UTF-8"))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Clean orphaned files from unpacked PPTX")
    parser.add_argument("unpacked_dir", help="Unpacked PPTX directory")
    args = parser.parse_args()

    count, msg = clean(args.unpacked_dir)
    print(msg)


if __name__ == "__main__":
    main()
