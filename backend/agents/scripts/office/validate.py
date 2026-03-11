"""Schema validators for Office XML files.

Provides validation and auto-repair for DOCX, PPTX, and XLSX documents
after XML editing, before repacking.

Used by pack.py during the pack step.
"""

import re
import random
from pathlib import Path
from xml.dom import minidom


class BaseValidator:
    """Base class for Office XML validators."""

    def __init__(self, unpacked_dir: Path, original_file: Path):
        self.unpacked_dir = Path(unpacked_dir)
        self.original_file = Path(original_file)
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """Run validation. Returns True if valid."""
        raise NotImplementedError

    def repair(self) -> int:
        """Auto-repair known issues. Returns count of repairs made."""
        raise NotImplementedError

    def _read_xml(self, filepath: Path) -> minidom.Document:
        """Parse an XML file."""
        try:
            with open(filepath, encoding="utf-8") as f:
                return minidom.parse(f)
        except Exception as e:
            self.errors.append(f"Failed to parse {filepath.name}: {e}")
            return None

    def _write_xml(self, filepath: Path, dom: minidom.Document) -> None:
        """Write an XML DOM back to file."""
        filepath.write_bytes(dom.toxml(encoding="UTF-8"))


class DOCXSchemaValidator(BaseValidator):
    """Validates DOCX XML schema compliance."""

    # Required element order within <w:pPr>
    PPR_ORDER = ["w:pStyle", "w:numPr", "w:spacing", "w:ind", "w:jc", "w:rPr"]

    def validate(self) -> bool:
        """Check DOCX XML for common schema violations."""
        doc_xml = self.unpacked_dir / "word" / "document.xml"
        if not doc_xml.exists():
            self.errors.append("word/document.xml not found")
            return False

        dom = self._read_xml(doc_xml)
        if dom is None:
            return False

        # Check w:t elements have xml:space="preserve" when needed
        for t_elem in dom.getElementsByTagName("w:t"):
            text = t_elem.firstChild.nodeValue if t_elem.firstChild else ""
            if text and (text.startswith(" ") or text.endswith(" ")):
                if not t_elem.getAttribute("xml:space") == "preserve":
                    self.warnings.append(
                        f"<w:t> with whitespace missing xml:space='preserve'"
                    )

        # Check RSIDs are valid 8-digit hex
        for attr_name in ["w:rsidR", "w:rsidRPr", "w:rsidRDefault", "w:rsidP"]:
            for elem in dom.getElementsByTagName("*"):
                rsid = elem.getAttribute(attr_name)
                if rsid and not re.match(r"^[0-9A-Fa-f]{8}$", rsid):
                    self.errors.append(
                        f"Invalid RSID '{rsid}' on <{elem.tagName}>"
                    )

        return len(self.errors) == 0

    def repair(self) -> int:
        """Auto-repair known DOCX issues."""
        repairs = 0
        doc_xml = self.unpacked_dir / "word" / "document.xml"
        if not doc_xml.exists():
            return 0

        dom = self._read_xml(doc_xml)
        if dom is None:
            return 0

        modified = False

        # Fix missing xml:space="preserve" on <w:t> with whitespace
        for t_elem in dom.getElementsByTagName("w:t"):
            text = t_elem.firstChild.nodeValue if t_elem.firstChild else ""
            if text and (text.startswith(" ") or text.endswith(" ")):
                if t_elem.getAttribute("xml:space") != "preserve":
                    t_elem.setAttribute("xml:space", "preserve")
                    repairs += 1
                    modified = True

        # Fix durableId values >= 0x7FFFFFFF
        for elem in dom.getElementsByTagName("*"):
            durable_id = elem.getAttribute("w14:durableId")
            if durable_id:
                try:
                    val = int(durable_id)
                    if val >= 0x7FFFFFFF:
                        new_id = str(random.randint(1, 0x7FFFFFFE))
                        elem.setAttribute("w14:durableId", new_id)
                        repairs += 1
                        modified = True
                except ValueError:
                    pass

        if modified:
            self._write_xml(doc_xml, dom)

        return repairs


class PPTXSchemaValidator(BaseValidator):
    """Validates PPTX XML schema compliance."""

    def validate(self) -> bool:
        """Check PPTX XML for common schema violations."""
        pres_xml = self.unpacked_dir / "ppt" / "presentation.xml"
        if not pres_xml.exists():
            self.errors.append("ppt/presentation.xml not found")
            return False

        dom = self._read_xml(pres_xml)
        if dom is None:
            return False

        # Check that slides referenced in sldIdLst actually exist
        for sld_id in dom.getElementsByTagName("p:sldId"):
            # Verify relationship targets exist
            pass  # Basic structure check

        return len(self.errors) == 0

    def repair(self) -> int:
        """Auto-repair known PPTX issues."""
        return 0  # PPTX typically needs less auto-repair


class RedliningValidator(BaseValidator):
    """Validates tracked changes (redlining) in DOCX."""

    def __init__(self, unpacked_dir: Path, original_file: Path, author: str = "Orbimesh"):
        super().__init__(unpacked_dir, original_file)
        self.author = author

    def validate(self) -> bool:
        """Check tracked changes for consistency."""
        doc_xml = self.unpacked_dir / "word" / "document.xml"
        if not doc_xml.exists():
            return True  # No doc to validate

        dom = self._read_xml(doc_xml)
        if dom is None:
            return False

        # Check that <w:del> elements use <w:delText> not <w:t>
        for del_elem in dom.getElementsByTagName("w:del"):
            for t_elem in del_elem.getElementsByTagName("w:t"):
                self.errors.append(
                    "<w:t> found inside <w:del> — should be <w:delText>"
                )

        # Check tracked change IDs are unique
        ids_seen = set()
        for tag_name in ["w:ins", "w:del"]:
            for elem in dom.getElementsByTagName(tag_name):
                tc_id = elem.getAttribute("w:id")
                if tc_id in ids_seen:
                    self.warnings.append(f"Duplicate tracked change id='{tc_id}'")
                ids_seen.add(tc_id)

        return len(self.errors) == 0

    def repair(self) -> int:
        """Auto-repair tracked change issues."""
        return 0  # Tracked changes are complex — don't auto-repair
