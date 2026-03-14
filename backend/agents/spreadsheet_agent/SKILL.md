---
id: spreadsheet_agent
name: Spreadsheet Agent
port: 9000
version: 1.1.0
description: >
  Priority agent for CSV and Excel analysis, transformations,
  and natural-language data operations with LLM-powered task decomposition.
  Includes formula verification and professional output standards.
model: ollama/minimax-m2.5:cloud
context_strategy: standard
requires_auth: false
triggers:
  - csv
  - excel
  - spreadsheet
  - xlsx
  - xls
  - tabular data
  - rows
  - columns
  - cells
  - aggregate
  - pivot
  - filter rows
  - sort data
  - formula
capabilities:
  - load_file
  - process_data
  - filter
  - sort
  - aggregate
  - add_column
  - drop_column
  - export
  - verify_formulas
not_for:
  - PDF documents
  - Word documents
  - web scraping
  - emails
  - code editing
---

# XLSX Creation, Editing, and Analysis

## Requirements for All Excel Outputs

### Professional Font
- Use a consistent, professional font (e.g., Arial, Times New Roman) unless otherwise instructed

### Zero Formula Errors
- Every Excel model MUST be delivered with ZERO formula errors (`#REF!`, `#DIV/0!`, `#VALUE!`, `#N/A`, `#NAME?`)
- Run `scripts/recalc.py` to verify before delivery

### Preserve Existing Templates
- EXACTLY match existing format, style, and conventions when modifying files
- Never impose standardized formatting on files with established patterns

---

## Overview

**PRIORITY AGENT** for ALL CSV and Excel files (.csv, .xlsx, .xls).

## Quick Reference

| Task | Approach |
|------|----------|
| Load and analyze | `pandas.read_excel()` or `pandas.read_csv()` |
| Create new Excel | `openpyxl` for styled output |
| Edit existing | `openpyxl` for formatting-aware edits |
| Verify formulas | `python scripts/recalc.py file.xlsx` |

---

## CRITICAL: Use Formulas, Not Hardcoded Values

### ❌ WRONG — Hardcoding
```python
ws["C2"] = 150  # Sum of A2:B2
ws["C3"] = 200  # Sum of A3:B3
```

### ✅ CORRECT — Using Formulas
```python
ws["C2"] = "=SUM(A2:B2)"
ws["C3"] = "=SUM(A3:B3)"
```

---

## Reading and Analyzing Data

```python
import pandas as pd

# Read Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Read CSV
df = pd.read_csv("data.csv")

# Basic analysis
print(df.describe())
print(df.head())
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

## Creating New Excel Files

```python
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Report"

# Header styling
header_font = Font(name="Arial", bold=True, color="FFFFFF", size=12)
header_fill = PatternFill(start_color="1E2761", end_color="1E2761", fill_type="solid")

headers = ["Name", "Revenue", "Growth"]
for col, header in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col, value=header)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = Alignment(horizontal="center")

# Data with formulas
ws["A2"] = "Product A"
ws["B2"] = 50000
ws["C2"] = "=B2/B$2"  # Formula, not hardcoded!

# Column widths
for col in range(1, len(headers) + 1):
    ws.column_dimensions[get_column_letter(col)].width = 15

wb.save("output.xlsx")
```

## Editing Existing Files

```python
wb = openpyxl.load_workbook("existing.xlsx")
ws = wb.active

# Preserve existing formatting while updating values
for row in range(2, ws.max_row + 1):
    cell = ws.cell(row=row, column=3)
    if cell.value and isinstance(cell.value, str) and cell.value.startswith("="):
        pass  # Don't overwrite existing formulas
    else:
        cell.value = "=SUM(A{0}:B{0})".format(row)

wb.save("updated.xlsx")
```

---

## Recalculating Formulas

```bash
python scripts/recalc.py output.xlsx
python scripts/recalc.py output.xlsx --sheet "Financial Model" --verbose
```

### Interpreting Output
- **PASS**: All formulas valid, no error values detected
- **FAIL**: Found formula errors — fix and re-run

---

## Formula Verification Checklist

### Essential Verification
1. Run `recalc.py` — no formula errors allowed
2. Check totals match expected values
3. Verify cross-sheet references are intact

### Common Pitfalls
- ❌ Hardcoded totals that don't update when data changes
- ❌ `#REF!` from deleted rows/columns
- ❌ `#DIV/0!` from missing denominators
- ❌ Mixed cell references (missing `$` for absolute refs)

### Formula Testing Strategy
1. Change one input value → verify dependent cells update
2. Add a row → verify SUM ranges include it
3. Delete a row → verify no `#REF!` errors appear

---

## Code Style Guidelines

```python
# ✅ Good: Clear, descriptive variable names
revenue_by_region = df.groupby("Region")["Revenue"].sum()

# ❌ Bad: Cryptic names
x = df.groupby("a")["b"].sum()

# ✅ Good: Use openpyxl for styled output
wb = openpyxl.Workbook()
ws = wb.active

# ❌ Bad: Use pandas to_excel without styling
df.to_excel("output.xlsx")  # No formatting, unprofessional
```

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `scripts/recalc.py` | Verify formulas, check for errors |
| Shared `office/unpack.py` | Extract XLSX XML (advanced) |
| Shared `office/pack.py` | Repack XLSX (advanced) |

## Best Practices

### Library Selection
| Task | Best Library |
|------|-------------|
| Data analysis | pandas |
| Styled Excel creation | openpyxl |
| Large datasets (read) | pandas with `engine="openpyxl"` |
| Formula-heavy workbooks | openpyxl (preserves formulas) |

## Dependencies

- `openpyxl` — Excel read/write with formatting
- `pandas` — Data analysis and transformation
- `xlrd` — Legacy .xls format support
