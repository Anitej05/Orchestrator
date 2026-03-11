---
id: document_agent_lib
name: Document Agent
port: 8050
version: 1.1.0
description: >
  LLM-powered document processing for Word (.docx) and text files.
  Supports reading, creating, editing (including XML-level manipulation),
  summarizing, tracked changes, and comments for DOCX and TXT.
model: cerebras/llama-3.3-70b
context_strategy: standard
requires_auth: false
triggers:
  - document
  - docx
  - word
  - text file
  - summarize document
  - read document
  - create report
  - contract
  - resume
  - tracked changes
  - comment
capabilities:
  - read_document
  - create_document
  - edit_document
  - summarize_document
  - answer_questions
  - convert_format
  - accept_changes
  - add_comments
not_for:
  - spreadsheets (CSV, Excel)
  - emails
  - web pages
  - images without text
  - PDF files (use PDF Agent)
  - presentations (PPT, PPTX) (use PPT Agent)
---

# DOCX Creation, Editing, and Analysis

## Overview

A .docx file is a ZIP archive containing XML files. This agent supports
both high-level operations (python-docx) and low-level XML editing
(unpack → edit XML → repack).

## Quick Reference

| Task | Approach |
|------|----------|
| Read/analyze content | `pandoc` or `python-docx` |
| Create new document | Use `python-docx` — see Creating New Documents below |
| Edit existing document | Unpack → edit XML → repack — see Editing below |
| Accept tracked changes | `python scripts/accept_changes.py unpacked/` |
| Add comments | `python scripts/comment.py unpacked/ 0 "text"` |

---

## Creating New Documents

Using `python-docx` for creating new DOCX files:

```python
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Heading
doc.add_heading("Report Title", level=1)

# Paragraph with formatting
para = doc.add_paragraph()
run = para.add_run("Important text")
run.bold = True
run.font.size = Pt(14)

# Table
table = doc.add_table(rows=3, cols=3)
table.style = "Table Grid"
for i, row in enumerate(table.rows):
    for j, cell in enumerate(row.cells):
        cell.text = f"Row {i+1}, Col {j+1}"

# Image
doc.add_picture("image.png", width=Inches(4))

doc.save("output.docx")
```

### Lists
```python
# Bullet list
doc.add_paragraph("First item", style="List Bullet")
doc.add_paragraph("Second item", style="List Bullet")

# Numbered list
doc.add_paragraph("Step one", style="List Number")
doc.add_paragraph("Step two", style="List Number")
```

### Page Layout
```python
from docx.shared import Cm

section = doc.sections[0]
section.page_width = Cm(21)      # A4 width
section.page_height = Cm(29.7)   # A4 height
section.top_margin = Cm(2.54)    # 1 inch
section.bottom_margin = Cm(2.54)
section.left_margin = Cm(2.54)
section.right_margin = Cm(2.54)
```

---

## Editing Existing Documents (XML Workflow)

**Follow all 3 steps in order.**

### Step 1: Unpack
```bash
python -m agents.scripts.office.unpack document.docx unpacked/
```
Extracts XML, pretty-prints, merges adjacent runs, and converts smart quotes.

### Step 2: Edit XML
Edit files in `unpacked/word/`. See XML Reference below for patterns.

**Use "Orbimesh" as the author** for tracked changes and comments, unless the user requests otherwise.

**CRITICAL: Use smart quotes for new content:**
```xml
<w:t>Here&#x2019;s a quote: &#x201C;Hello&#x201D;</w:t>
```

| Entity | Character |
|--------|-----------|
| `&#x2018;` | ' (left single) |
| `&#x2019;` | ' (right single / apostrophe) |
| `&#x201C;` | " (left double) |
| `&#x201D;` | " (right double) |

**Adding comments:** Use `comment.py` to handle boilerplate:
```bash
python scripts/comment.py unpacked/ 0 "Comment text"
python scripts/comment.py unpacked/ 1 "Reply text" --parent 0
```

### Step 3: Pack
```bash
python -m agents.scripts.office.pack unpacked/ output.docx --original document.docx
```
Validates with auto-repair, condenses XML, and creates DOCX.

---

## XML Reference

### Tracked Changes — Insertion
```xml
<w:ins w:id="1" w:author="Orbimesh" w:date="2025-01-01T00:00:00Z">
  <w:r><w:t>inserted text</w:t></w:r>
</w:ins>
```

### Tracked Changes — Deletion
```xml
<w:del w:id="2" w:author="Orbimesh" w:date="2025-01-01T00:00:00Z">
  <w:r><w:delText>deleted text</w:delText></w:r>
</w:del>
```

**Inside `<w:del>`**: Use `<w:delText>` instead of `<w:t>`.

### Minimal Edits
Only mark what changes:
```xml
<w:r><w:t>The term is </w:t></w:r>
<w:del w:id="1" w:author="Orbimesh" w:date="...">
  <w:r><w:delText>30</w:delText></w:r>
</w:del>
<w:ins w:id="2" w:author="Orbimesh" w:date="...">
  <w:r><w:t>60</w:t></w:r>
</w:ins>
<w:r><w:t> days.</w:t></w:r>
```

### Comments
After running `comment.py`, add markers to document.xml:
```xml
<w:commentRangeStart w:id="0"/>
<w:r><w:t>commented text</w:t></w:r>
<w:commentRangeEnd w:id="0"/>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="0"/></w:r>
```

### Schema Rules
- **Element order in `<w:pPr>`**: `<w:pStyle>`, `<w:numPr>`, `<w:spacing>`, `<w:ind>`, `<w:jc>`, `<w:rPr>` last
- **Whitespace**: Add `xml:space="preserve"` to `<w:t>` with leading/trailing spaces
- **RSIDs**: Must be 8-digit hex (e.g., `00AB1234`)

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `scripts/accept_changes.py` | Accept all tracked changes |
| `scripts/comment.py` | Add comments and replies |
| Shared `office/unpack.py` | Extract DOCX XML |
| Shared `office/pack.py` | Repack with validation |
| Shared `office/soffice.py` | PDF conversion |

## Dependencies

- `python-docx` — Document manipulation
- `pandoc` — Text extraction
- LibreOffice (`soffice`) — PDF conversion
- Poppler (`pdftoppm`) — PDF to images
