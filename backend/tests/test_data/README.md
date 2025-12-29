# Test Data Directory

This directory contains test files for manual agent testing.

## File Structure

```
test_data/
├── sample_resume.pdf          # For document agent analysis tests
├── sample_document.docx       # For document agent edit tests
├── sales_data.csv             # For spreadsheet agent tests
├── financial_data.xlsx        # For spreadsheet agent tests
└── README.md                  # This file
```

## Recommended Test Files

### For Document Agent (`test_document_manual.py`)

1. **sample_resume.pdf** - A sample resume/CV
   - Used for: Document analysis, extracting skills, experience
   
2. **sample_document.docx** - Any Word document
   - Used for: Document editing, formatting tests
   
3. **sample_report.txt** - Plain text document
   - Used for: Text processing, conversion tests

### For Spreadsheet Agent (`test_spreadsheet_manual.py`)

1. **sales_data.csv** - Sample sales data with columns like:
   ```
   Date, Product, Category, Quantity, Price, Total
   2024-01-01, Widget A, Electronics, 5, 100, 500
   2024-01-02, Widget B, Home, 3, 50, 150
   ```

2. **financial_data.xlsx** - Excel file with financial data
   - Multiple sheets
   - Formulas, charts (optional)

3. **inventory.csv** - Inventory data
   ```
   SKU, Product Name, Quantity, Warehouse, Last Updated
   ```

## How to Use

1. **Add your test files** to this directory
2. **Run the test scripts**:
   ```bash
   # Document Agent Tests
   python backend/tests/document_agent/test_document_manual.py
   
   # Spreadsheet Agent Tests
   python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py
   
   # Browser Agent Tests
   python backend/tests/browser_agent/test_amazon_shoes.py
   ```

3. **Check results** - Outputs will be saved to `storage/` directories

## File Format Support

### Document Agent
- ✅ PDF (.pdf)
- ✅ Word (.docx)
- ✅ Text (.txt)

### Spreadsheet Agent
- ✅ CSV (.csv)
- ✅ Excel (.xlsx, .xls)

### Browser Agent
- ✅ No files needed (web-based)

## Sample Data Templates

If you don't have test files, you can:

1. **Create a sample CSV** in Excel or text editor:
   ```csv
   Date,Product,Amount
   2024-01-01,Product A,1000
   2024-01-02,Product B,1500
   2024-01-03,Product A,2000
   ```

2. **Use online resume templates** for PDF testing

3. **Create a simple Word document** for editing tests

## Notes

- Test files are **not committed to git** (in .gitignore)
- Keep test files **small** (< 5MB recommended)
- Use **realistic data** for better test results
- **Document agent** creates outputs in `storage/documents/`
- **Spreadsheet agent** creates outputs in `storage/spreadsheets/`
