# Orbimesh Test Suite

This directory contains all unit, integration, and end-to-end tests for the Orbimesh backend agents and orchestrator.

## Directory Structure

```
tests/
├── browser_agent/          # Browser automation agent tests
│   ├── test_browser_agent.py
│   ├── test_amazon_shoes.py
│   ├── test_amazon_headphones.py
│   ├── test_complex_vision.py
│   ├── test_google_doodle.py
│   ├── test_pdf_to_word.py
│   ├── test_screenshot.py
│   ├── test_vision_quick.py
│   ├── test_vnr_syllabus.py
│   ├── test_vnrvjiet_syllabus.py
│   └── *.log              # Test execution logs
│
├── document_agent/         # Document analysis agent tests
│   ├── test_file_processor.py
│   ├── test_text_extraction.py
│   ├── test_table_extraction.py
│   ├── test_metadata_extraction.py
│   ├── test_document_editing.py
│   ├── test_format_support.py
│   ├── test_integration.py
│   └── TESTING_PLAN.md    # Detailed testing plan
│
├── mail_agent/            # Mail agent tests
│   └── (tests to be added)
│
├── spreadsheet_agent/     # Spreadsheet agent tests
│   └── (tests to be added)
│
├── orchestrator/          # Orchestrator integration tests
│   └── test_orchestrator_integration.py
│
├── fixtures/              # Test data and fixtures
│   ├── test_data.csv
│   ├── sample_documents/
│   │   ├── sample.pdf
│   │   ├── sample.docx
│   │   ├── sample.txt
│   │   └── sample_table.xlsx
│   └── test_utils.py
│
└── README.md             # This file
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run tests for specific agent
```bash
# Browser agent
pytest tests/browser_agent/ -v

# Document agent
pytest tests/document_agent/ -v

# Orchestrator
pytest tests/orchestrator/ -v
```

### Run specific test file
```bash
pytest tests/browser_agent/test_browser_agent.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=agents --cov=orchestrator --cov-report=html
```

### Run with detailed output
```bash
pytest tests/ -vv -s
```

## Test Categories

### Browser Agent Tests
Tests for browser automation, web scraping, vision-based navigation, and form filling.
- **Functional**: End-to-end browser navigation tasks
- **Vision**: Vision model response parsing and action interpretation
- **Screenshot**: Screenshot capture and analysis
- **Integration**: Real browser interactions

### Document Agent Tests
Tests for document analysis, text extraction, table extraction, and document editing.
- **Unit**: Individual function testing (text extraction, parsing, etc.)
- **Integration**: Multi-step workflows (analyze → edit → save)
- **Format Support**: Different file formats (PDF, DOCX, images, etc.)
- **Edge Cases**: Large files, corrupted files, special characters, etc.

### Orchestrator Tests
Tests for agent selection, workflow routing, and multi-agent orchestration.
- **Agent Selection**: Correct agent selection based on query
- **Workflow**: Multi-step workflows across agents
- **Error Handling**: Fallback and error scenarios

## Writing New Tests

### Test File Template
```python
import pytest
from pathlib import Path

class TestFeatureName:
    """Test suite for feature_name"""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data"""
        yield data
    
    def test_basic_functionality(self, setup_data):
        """Test basic functionality"""
        assert result == expected
    
    def test_error_handling(self):
        """Test error scenarios"""
        with pytest.raises(ExceptionType):
            # code that should raise exception
            pass
```

### Test Naming Convention
- Test files: `test_<feature>.py`
- Test classes: `Test<FeatureName>`
- Test methods: `test_<specific_case>`

Example: `test_document_agent.py` → `TestDocumentAgent` → `test_pdf_extraction_with_tables()`

## CI/CD Integration

Tests are automatically run on:
- Pull requests
- Commits to main branch
- Manual GitHub Actions trigger

View test results in GitHub Actions workflow logs.

## Coverage Goals

- **Browser Agent**: 75%+ coverage
- **Document Agent**: 85%+ coverage
- **Orchestrator**: 80%+ coverage
- **Overall**: 80%+ coverage

## Known Issues

- Vision tests may be flaky due to model variability
- Browser tests require headless browser setup
- Document tests require sample files in fixtures/

## Contact

For issues or questions about tests, contact the development team.
