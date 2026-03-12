"""Duplicate a slide or create a new one from a layout in an unpacked PPTX.

Updates the slide ID list and relationships. After using this script,
add the printed <p:sldId> element to <p:sldIdLst> in presentation.xml
at the desired position.

Usage:
    python add_slide.py <unpacked_dir> <source_slide_or_layout>

Examples:
    python add_slide.py unpacked/ slide2.xml        # Duplicate slide2
    python add_slide.py unpacked/ slideLayout2.xml  # Create from layout
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from xml.dom import minidom


def add_slide(unpacked_dir: str, source: str) -> tuple:
    """Duplicate a slide or create from layout.

    Args:
        unpacked_dir: Path to the unpacked PPTX directory
        source: Filename of source slide or layout (e.g. slide2.xml, slideLayout3.xml)

    Returns:
        Tuple of (new_slide_filename, sldId_element_string)
    """
    unpacked = Path(unpacked_dir)
    ppt_dir = unpacked / "ppt"
    slides_dir = ppt_dir / "slides"
    rels_dir = slides_dir / "_rels"

    if not slides_dir.exists():
        return None, "Error: ppt/slides/ directory not found"

    # Determine next slide number
    existing = sorted(slides_dir.glob("slide*.xml"))
    numbers = []
    for f in existing:
        match = re.match(r"slide(\d+)\.xml", f.name)
        if match:
            numbers.append(int(match.group(1)))

    next_num = max(numbers) + 1 if numbers else 1
    new_slide_name = f"slide{next_num}.xml"
    new_slide_path = slides_dir / new_slide_name

    # Copy source content
    if source.startswith("slideLayout"):
        # Create from layout
        source_path = ppt_dir / "slideLayouts" / source
        if not source_path.exists():
            return None, f"Error: Layout {source} not found"

        # Create a minimal slide referencing this layout
        slide_xml = _create_slide_from_layout(source)
        new_slide_path.write_text(slide_xml, encoding="utf-8")

    elif source.startswith("slide"):
        # Duplicate existing slide
        source_path = slides_dir / source
        if not source_path.exists():
            return None, f"Error: Slide {source} not found"

        shutil.copy2(source_path, new_slide_path)

        # Copy .rels file too
        source_rels = rels_dir / f"{source}.rels"
        if source_rels.exists():
            rels_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_rels, rels_dir / f"{new_slide_name}.rels")
    else:
        return None, f"Error: Source must be slideN.xml or slideLayoutN.xml"

    # Generate a unique slide ID
    pres_xml = ppt_dir / "presentation.xml"
    used_ids = set()
    if pres_xml.exists():
        try:
            dom = minidom.parse(str(pres_xml))
            for sld_id in dom.getElementsByTagName("p:sldId"):
                id_val = sld_id.getAttribute("id")
                if id_val:
                    used_ids.add(int(id_val))
        except Exception:
            pass

    new_id = 256
    while new_id in used_ids:
        new_id += 1

    # Add relationship to presentation.xml.rels
    pres_rels = ppt_dir / "_rels" / "presentation.xml.rels"
    r_id = _add_presentation_rel(pres_rels, new_slide_name)

    sld_id_element = f'<p:sldId id="{new_id}" r:id="{r_id}"/>'

    # Add to [Content_Types].xml
    _add_content_type(unpacked, new_slide_name)

    return new_slide_name, sld_id_element


def _create_slide_from_layout(layout_name: str) -> str:
    """Create minimal slide XML referencing a layout."""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr/>
    </p:spTree>
  </p:cSld>
</p:sld>"""


def _add_presentation_rel(rels_file: Path, slide_name: str) -> str:
    """Add a relationship entry for the new slide and return the rId."""
    if not rels_file.exists():
        return "rId1"

    try:
        dom = minidom.parse(str(rels_file))
        relationships = dom.getElementsByTagName("Relationships")[0]

        # Find next rId
        max_id = 0
        for rel in dom.getElementsByTagName("Relationship"):
            rid = rel.getAttribute("Id")
            match = re.match(r"rId(\d+)", rid)
            if match:
                max_id = max(max_id, int(match.group(1)))

        new_rid = f"rId{max_id + 1}"

        # Create new Relationship element
        new_rel = dom.createElement("Relationship")
        new_rel.setAttribute("Id", new_rid)
        new_rel.setAttribute(
            "Type",
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
        )
        new_rel.setAttribute("Target", f"slides/{slide_name}")
        relationships.appendChild(new_rel)

        rels_file.write_bytes(dom.toxml(encoding="UTF-8"))
        return new_rid

    except Exception as e:
        print(f"Warning: Could not update rels: {e}", file=sys.stderr)
        return "rIdNew"


def _add_content_type(unpacked_dir: Path, slide_name: str) -> None:
    """Add content type entry for the new slide."""
    ct_file = unpacked_dir / "[Content_Types].xml"
    if not ct_file.exists():
        return

    try:
        dom = minidom.parse(str(ct_file))
        types = dom.getElementsByTagName("Types")[0]

        override = dom.createElement("Override")
        override.setAttribute("PartName", f"/ppt/slides/{slide_name}")
        override.setAttribute(
            "ContentType",
            "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
        )
        types.appendChild(override)

        ct_file.write_bytes(dom.toxml(encoding="UTF-8"))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Add/duplicate slide in unpacked PPTX")
    parser.add_argument("unpacked_dir", help="Unpacked PPTX directory")
    parser.add_argument("source", help="Source slide (slideN.xml) or layout (slideLayoutN.xml)")
    args = parser.parse_args()

    new_slide, sld_id = add_slide(args.unpacked_dir, args.source)

    if new_slide:
        print(f"Created: {new_slide}")
        print(f"Add to <p:sldIdLst> at desired position:")
        print(f"  {sld_id}")
    else:
        print(sld_id)  # Error message
        sys.exit(1)


if __name__ == "__main__":
    main()
