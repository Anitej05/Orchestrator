"""
Canvas Integration Tests -- Templates & On-The-Fly Generation

Tests both template-based and LLM-driven on-the-fly canvas generation 
for every agent in the Orbimesh orchestrator.

Run:
    cd d:\\Internship\\Orbimesh\\backend
    python tests/test_canvas_integration.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')



import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s → %(message)s")
logger = logging.getLogger("CanvasTest")

# ============================================================================
# TEST DATA — Representative outputs for each agent
# ============================================================================

CODING_AGENT_OUTPUT = """
Created file: utils/helper.py
Modified file: main.py

Summary: Added a new utility module with helper functions for data validation.
Tests passed: 3/3
"""

CODING_AGENT_FILE_CHANGES = [
    {
        "file": "utils/helper.py",
        "diff": "--- /dev/null\n+++ b/utils/helper.py\n@@ -0,0 +1,10 @@\n+def validate_email(email: str) -> bool:\n+    import re\n+    return bool(re.match(r'^[\\w.-]+@[\\w.-]+\\.\\w+$', email))\n+\n+def sanitize_input(text: str) -> str:\n+    return text.strip().replace('<', '&lt;').replace('>', '&gt;')\n",
        "language": "python",
        "status": "created",
    },
    {
        "file": "main.py", 
        "diff": "--- a/main.py\n+++ b/main.py\n@@ -1,3 +1,4 @@\n+from utils.helper import validate_email, sanitize_input\n import sys\n import os\n",
        "language": "python",
        "status": "modified",
    },
]

SPREADSHEET_AGENT_OUTPUT_TABLE = """
Loaded file: sales_data.csv
Shape: (150, 5)
Columns: ['Date', 'Product', 'Region', 'Units', 'Revenue']
Top 5 rows displayed.
"""

SPREADSHEET_AGENT_OUTPUT_SUMMARY = """
Aggregation Results:
- Total Revenue: $1,234,567
- Average Units per Day: 42
- Top Region: North America (45%)
- Top Product: Widget Pro ($567,890)
- Growth Rate: 12.3% MoM
"""

MAIL_AGENT_OUTPUT_DRAFT = {
    "to": ["team@company.com"],
    "subject": "Q4 Sprint Planning",
    "body": "Hi team,\n\nLet's schedule our Q4 sprint planning meeting for next week.\n\nBest,\nAI Assistant",
    "cc": ["manager@company.com"],
}

MAIL_AGENT_OUTPUT_SEARCH = """
Found 5 emails matching "project update":
1. [Nov 15] From: alice@team.com — "Project Alpha Update - Week 46"
2. [Nov 14] From: bob@team.com — "Re: Project Beta Status"
3. [Nov 13] From: carol@team.com — "Project Roadmap Q4"
4. [Nov 12] From: dave@team.com — "Weekly Project Digest"
5. [Nov 11] From: eve@team.com — "Project Meeting Notes"
"""

UNIVERSAL_AGENT_CODE_OUTPUT = """
Here's the Python code to solve the Fibonacci problem:

```python
def fibonacci(n: int) -> list:
    \"\"\"Generate the first n Fibonacci numbers.\"\"\"
    if n <= 0:
        return []
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib[:n]

# Example: first 10 Fibonacci numbers
result = fibonacci(10)
print(result)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```
"""

UNIVERSAL_AGENT_ANALYSIS_OUTPUT = """
## Market Analysis: AI Industry Trends 2025

### Key Findings

The AI industry continues to grow at an unprecedented rate, with several key trends emerging:

1. **Large Language Models (LLMs)** — The shift towards smaller, more efficient models is gaining momentum. Models under 10B parameters now match larger counterparts in many tasks.

2. **AI Agents** — Autonomous AI agents that can browse, code, and manage tasks are becoming mainstream. The agent framework market is expected to reach $15B by 2026.

3. **Edge AI** — On-device inference is becoming more practical with quantized models, enabling offline AI capabilities on smartphones and IoT devices.

4. **Regulatory Landscape** — The EU AI Act and similar regulations worldwide are shaping how AI companies deploy products, with a focus on transparency and safety.

5. **Multimodal AI** — Models that combine text, image, audio, and video understanding are the new standard, replacing single-modality approaches.

### Market Size
- 2024: $196B
- 2025 (est.): $268B
- Growth: 36.8% YoY

### Risks
- Regulatory compliance costs
- Talent shortage in ML engineering
- Data quality and bias concerns
"""

DOCUMENT_AGENT_OUTPUT_TEXT = """
# Project Specification Document

## Overview
This document outlines the technical specification for the Orbimesh Orchestrator, 
a multi-agent AI system for desktop automation and productivity.

## Architecture
The system uses a hub-spoke model where a central orchestrator routes tasks to 
specialized agents based on intent classification.

## Components
1. Orchestrator Core
2. Agent Registry
3. Canvas Service
4. Inference Service
5. Tool Registry
"""

BROWSER_AGENT_OUTPUT = """
✅ Task Completed (3/3 subtasks)

  ✓ Navigate to Amazon.com
  ✓ Search for "mechanical keyboard"
  ✓ Extract top 3 product names and prices

📋 **Extracted Information:**
  • product_1: {"name": "Keychron K2", "price": "$74.99", "rating": "4.7"}
  • product_2: {"name": "Corsair K70", "price": "$109.99", "rating": "4.5"}
  • product_3: {"name": "Ducky One 3", "price": "$129.00", "rating": "4.8"}
"""


# ============================================================================
# TESTS — Part 1: Template-Based Canvas
# ============================================================================

def test_template_spreadsheet():
    """Test spreadsheet_viewer template."""
    from services.canvas_service import CanvasService
    
    display = CanvasService.build_from_template(
        "spreadsheet_viewer",
        {
            "headers": ["Date", "Product", "Region", "Units", "Revenue"],
            "rows": [
                ["2025-01-01", "Widget A", "North", "100", "$5,000"],
                ["2025-01-02", "Widget B", "South", "75", "$3,750"],
                ["2025-01-03", "Widget A", "East", "120", "$6,000"],
            ],
            "filename": "sales_data.csv",
        },
        title="Sales Data",
    )
    assert display is not None, "spreadsheet_viewer template failed"
    assert display.canvas_type == "spreadsheet"
    assert display.canvas_title == "Sales Data"
    assert display.canvas_data is not None
    logger.info(f"  ✅ spreadsheet_viewer → type={display.canvas_type}, title={display.canvas_title}")
    return display


def test_template_email_preview():
    """Test email_preview template."""
    from services.canvas_service import CanvasService
    
    display = CanvasService.build_email_preview(
        to=["team@company.com"],
        subject="Sprint Planning",
        body="Hi team, let's plan our sprint.",
        cc=["manager@company.com"],
        requires_confirmation=True,
        confirmation_message="Send this email?",
    )
    assert display is not None, "email_preview failed"
    assert display.canvas_type == "email_preview"
    assert display.requires_confirmation is True
    logger.info(f"  ✅ email_preview → type={display.canvas_type}, confirmation={display.requires_confirmation}")
    return display


def test_template_document_viewer():
    """Test document_viewer template."""
    from services.canvas_service import CanvasService
    
    display = CanvasService.build_document_view(
        content=DOCUMENT_AGENT_OUTPUT_TEXT,
        title="Project Spec",
        file_path="docs/spec.md",
    )
    assert display is not None, "document_viewer failed"
    assert display.canvas_type == "markdown", f"Expected 'markdown', got '{display.canvas_type}'"
    logger.info(f"  ✅ document_viewer → type={display.canvas_type}, title={display.canvas_title}")
    return display


def test_template_code_viewer():
    """Test code_viewer template."""
    from services.canvas_service import CanvasService
    
    display = CanvasService.build_from_template(
        "code_viewer",
        {
            "code": "def hello():\n    print('Hello World')\n\nhello()",
            "language": "python",
            "filename": "hello.py",
        },
        title="Hello World Script",
    )
    assert display is not None, "code_viewer failed"
    assert display.canvas_type == "code"
    logger.info(f"  ✅ code_viewer → type={display.canvas_type}")
    return display


def test_template_code_diff_viewer():
    """Test code_diff_viewer template (coding agent's primary)."""
    from services.canvas_service import CanvasService
    
    display = CanvasService.build_from_template(
        "code_diff_viewer",
        {
            "diffs": CODING_AGENT_FILE_CHANGES,
            "files_modified": ["utils/helper.py", "main.py"],
            "file_count": 2,
            "summary": "Added utility module with validation helpers",
        },
        title="Code Changes (2 files)",
        requires_confirmation=True,
        confirmation_message="Apply these changes?",
    )
    assert display is not None, "code_diff_viewer failed"
    assert display.canvas_type == "code"
    assert display.requires_confirmation is True
    logger.info(f"  ✅ code_diff_viewer → type={display.canvas_type}, confirmation={display.requires_confirmation}")
    return display


def test_template_chart_bar():
    """Test chart_bar template."""
    from services.canvas_service import CanvasService
    
    display = CanvasService.build_from_template(
        "chart_bar",
        {
            "labels": ["North", "South", "East", "West"],
            "datasets": [{"label": "Revenue", "data": [450000, 320000, 280000, 184000]}],
        },
        title="Revenue by Region",
    )
    assert display is not None, "chart_bar failed"
    assert display.canvas_type == "chart"
    logger.info(f"  ✅ chart_bar → type={display.canvas_type}")
    return display


def test_template_json_tree():
    """Test json_tree template."""
    from services.canvas_service import CanvasService
    
    display = CanvasService.build_from_template(
        "json_tree",
        {
            "data": {
                "name": "Orbimesh",
                "version": "2.0",
                "agents": ["coding", "spreadsheet", "mail", "document", "browser", "universal"],
                "config": {"max_agents": 10, "debug": False}
            },
            "title": "System Config",
        },
        title="System Configuration",
    )
    assert display is not None, "json_tree failed"
    assert display.canvas_type == "json"
    logger.info(f"  ✅ json_tree → type={display.canvas_type}")
    return display


def test_template_markdown_viewer():
    """Test markdown_viewer template."""
    from services.canvas_service import CanvasService
    
    display = CanvasService.build_from_template(
        "markdown_viewer",
        {
            "content": "# Hello\n\nThis is **bold** and _italic_ text.\n\n- Item 1\n- Item 2",
            "title": "Sample Markdown",
        },
        title="Markdown Preview",
    )
    assert display is not None, "markdown_viewer failed"
    assert display.canvas_type == "markdown"
    logger.info(f"  ✅ markdown_viewer → type={display.canvas_type}")
    return display


# ============================================================================
# TESTS — Part 2: LLM-Driven On-The-Fly Canvas
# ============================================================================

async def test_llm_canvas_coding_agent():
    """Test LLM canvas decision for coding agent output (should pick code/code_diff)."""
    from services.canvas_service import CanvasService
    
    display = await CanvasService.decide_canvas_llm(
        output=CODING_AGENT_OUTPUT,
        agent_name="coding_agent",
        capability_name="code_task",
        file_changes=CODING_AGENT_FILE_CHANGES,
        files_modified=["utils/helper.py", "main.py"],
    )
    assert display is not None, "LLM canvas for coding_agent returned None"
    logger.info(f"  ✅ coding_agent LLM → type={display.canvas_type}, title={display.canvas_title}")
    return display


async def test_llm_canvas_spreadsheet_table():
    """Test LLM canvas for spreadsheet agent with tabular output."""
    from services.canvas_service import CanvasService
    
    display = await CanvasService.decide_canvas_llm(
        output=SPREADSHEET_AGENT_OUTPUT_TABLE,
        agent_name="spreadsheet_agent",
        capability_name="load_file",
        primary_canvas_type="spreadsheet",
    )
    assert display is not None, "LLM canvas for spreadsheet_agent (table) returned None"
    logger.info(f"  ✅ spreadsheet_agent (table) LLM → type={display.canvas_type}, title={display.canvas_title}")
    return display


async def test_llm_canvas_spreadsheet_summary():
    """Test LLM canvas for spreadsheet agent with aggregation summary (could be chart/markdown)."""
    from services.canvas_service import CanvasService
    
    display = await CanvasService.decide_canvas_llm(
        output=SPREADSHEET_AGENT_OUTPUT_SUMMARY,
        agent_name="spreadsheet_agent",
        capability_name="aggregate_data",
        primary_canvas_type="spreadsheet",
    )
    assert display is not None, "LLM canvas for spreadsheet_agent (summary) returned None"
    logger.info(f"  ✅ spreadsheet_agent (summary) LLM → type={display.canvas_type}, title={display.canvas_title}")
    return display


async def test_llm_canvas_mail_search():
    """Test LLM canvas for mail agent search results (non-email output)."""
    from services.canvas_service import CanvasService
    
    display = await CanvasService.decide_canvas_llm(
        output=MAIL_AGENT_OUTPUT_SEARCH,
        agent_name="mail_agent",
        capability_name="email_operations",
        primary_canvas_type="email_preview",
    )
    assert display is not None, "LLM canvas for mail_agent (search) returned None"
    logger.info(f"  ✅ mail_agent (search) LLM → type={display.canvas_type}, title={display.canvas_title}")
    return display


async def test_llm_canvas_universal_code():
    """Test LLM canvas for universal agent code output (should pick code_viewer)."""
    from services.canvas_service import CanvasService
    
    display = await CanvasService.decide_canvas_llm(
        output=UNIVERSAL_AGENT_CODE_OUTPUT,
        agent_name="universal_agent",
        capability_name="generate_code",
    )
    assert display is not None, "LLM canvas for universal_agent (code) returned None"
    logger.info(f"  ✅ universal_agent (code) LLM → type={display.canvas_type}, title={display.canvas_title}")
    return display


async def test_llm_canvas_universal_analysis():
    """Test LLM canvas for universal agent analysis (should pick markdown or document)."""
    from services.canvas_service import CanvasService
    
    display = await CanvasService.decide_canvas_llm(
        output=UNIVERSAL_AGENT_ANALYSIS_OUTPUT,
        agent_name="universal_agent",
        capability_name="analyze",
    )
    assert display is not None, "LLM canvas for universal_agent (analysis) returned None"
    logger.info(f"  ✅ universal_agent (analysis) LLM → type={display.canvas_type}, title={display.canvas_title}")
    return display


async def test_llm_canvas_document_text():
    """Test LLM canvas for document agent text display."""
    from services.canvas_service import CanvasService
    
    display = await CanvasService.decide_canvas_llm(
        output=DOCUMENT_AGENT_OUTPUT_TEXT,
        agent_name="document_agent",
        capability_name="display_document",
    )
    assert display is not None, "LLM canvas for document_agent returned None"
    logger.info(f"  ✅ document_agent LLM → type={display.canvas_type}, title={display.canvas_title}")
    return display


async def test_llm_canvas_browser_extracted():
    """Test LLM canvas for browser agent extracted data (structured products)."""
    from services.canvas_service import CanvasService
    
    display = await CanvasService.decide_canvas_llm(
        output=BROWSER_AGENT_OUTPUT,
        agent_name="browser_agent",
        capability_name="browse",
    )
    assert display is not None, "LLM canvas for browser_agent returned None"
    logger.info(f"  ✅ browser_agent LLM → type={display.canvas_type}, title={display.canvas_title}")
    return display


# ============================================================================
# TESTS — Part 3: Fallback Behavior
# ============================================================================

async def test_fallback_decision():
    """Test that _fallback_decision produces valid CanvasDecision when LLM is unavailable."""
    from services.canvas_llm import _fallback_decision
    
    # Test with file changes → should return code canvas
    decision = _fallback_decision(
        output="Some code output", 
        file_changes=CODING_AGENT_FILE_CHANGES,
        files_modified=["utils/helper.py", "main.py"],
        agent_name="coding_agent",
        capability_name="code_task",
    )
    assert decision.canvas_type == "code", f"Expected 'code', got '{decision.canvas_type}'"
    assert decision.requires_confirmation is True
    logger.info(f"  ✅ fallback (code changes) → type={decision.canvas_type}")
    
    # Test with spreadsheet primary → should return spreadsheet
    decision = _fallback_decision(
        output="Some data",
        agent_name="spreadsheet_agent",
        capability_name="load_file",
        primary_canvas_type="spreadsheet",
    )
    assert decision.canvas_type == "spreadsheet"
    logger.info(f"  ✅ fallback (spreadsheet primary) → type={decision.canvas_type}")
    
    # Test with mail primary → should return markdown
    decision = _fallback_decision(
        output="Search results here",
        agent_name="mail_agent",
        capability_name="search",
        primary_canvas_type="email_preview",
    )
    assert decision.canvas_type == "markdown"
    logger.info(f"  ✅ fallback (mail non-email) → type={decision.canvas_type}")
    
    # Test with generic → should return markdown
    decision = _fallback_decision(
        output="Hello world",
        agent_name="universal_agent",
        capability_name="execute_task",
    )
    assert decision.canvas_type == "markdown"
    logger.info(f"  ✅ fallback (generic) → type={decision.canvas_type}")


# ============================================================================
# TESTS — Part 4: Template Registry
# ============================================================================

def test_template_registry():
    """Test that all expected templates are registered."""
    from services.canvas_templates import get_template_ids, get_template, validate_template_data
    
    ids = get_template_ids()
    expected = [
        "spreadsheet_viewer", "spreadsheet_plan", "email_preview",
        "document_viewer", "pdf_viewer", "markdown_viewer",
        "chart_bar", "chart_line", "chart_pie",
        "code_viewer", "code_diff_viewer", "json_tree", "image_viewer",
    ]
    for tid in expected:
        assert tid in ids, f"Missing template: {tid}"
        tmpl = get_template(tid)
        assert tmpl is not None, f"get_template({tid}) returned None"
        assert "canvas_type" in tmpl
        assert "data_schema" in tmpl
    
    logger.info(f"  ✅ All {len(expected)} templates registered and valid")
    
    # Validate a template's data schema
    valid, err = validate_template_data("spreadsheet_viewer", {"headers": ["A"], "rows": [[1]]})
    assert valid, f"Validation failed: {err}"
    
    invalid, err = validate_template_data("spreadsheet_viewer", {})
    assert not invalid, "Validation should fail with empty data"
    
    logger.info(f"  ✅ Template data validation works correctly")


# ============================================================================
# RUNNER
# ============================================================================

async def run_all_tests():
    """Run all canvas integration tests."""
    results = {"passed": 0, "failed": 0, "errors": []}
    
    # === Template Tests ===
    template_tests = [
        ("Template: spreadsheet_viewer", test_template_spreadsheet),
        ("Template: email_preview", test_template_email_preview),
        ("Template: document_viewer", test_template_document_viewer),
        ("Template: code_viewer", test_template_code_viewer),
        ("Template: code_diff_viewer", test_template_code_diff_viewer),
        ("Template: chart_bar", test_template_chart_bar),
        ("Template: json_tree", test_template_json_tree),
        ("Template: markdown_viewer", test_template_markdown_viewer),
        ("Template: registry", test_template_registry),
    ]
    
    print("\n" + "=" * 72)
    print("  PART 1: TEMPLATE-BASED CANVAS TESTS")
    print("=" * 72)
    
    for name, test_fn in template_tests:
        try:
            test_fn()
            results["passed"] += 1
            print(f"  ✅ PASS: {name}")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append((name, str(e)))
            print(f"  ❌ FAIL: {name} — {e}")
    
    # === LLM On-The-Fly Tests ===
    llm_tests = [
        ("LLM: coding_agent (code diffs)", test_llm_canvas_coding_agent),
        ("LLM: spreadsheet_agent (table)", test_llm_canvas_spreadsheet_table),
        ("LLM: spreadsheet_agent (summary)", test_llm_canvas_spreadsheet_summary),
        ("LLM: mail_agent (search results)", test_llm_canvas_mail_search),
        ("LLM: universal_agent (code gen)", test_llm_canvas_universal_code),
        ("LLM: universal_agent (analysis)", test_llm_canvas_universal_analysis),
        ("LLM: document_agent (text)", test_llm_canvas_document_text),
        ("LLM: browser_agent (extracted)", test_llm_canvas_browser_extracted),
    ]
    
    print("\n" + "=" * 72)
    print("  PART 2: LLM-DRIVEN ON-THE-FLY CANVAS TESTS")
    print("=" * 72)
    
    for name, test_fn in llm_tests:
        try:
            display = await test_fn()
            results["passed"] += 1
            ctype = display.canvas_type if display else "N/A"
            title = display.canvas_title if display else "N/A"
            print(f"  ✅ PASS: {name} → type={ctype}, title=\"{title}\"")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append((name, str(e)))
            print(f"  ❌ FAIL: {name} — {e}")
    
    # === Fallback Tests ===
    print("\n" + "=" * 72)
    print("  PART 3: FALLBACK BEHAVIOR TESTS")
    print("=" * 72)
    
    try:
        await test_fallback_decision()
        results["passed"] += 1
        print(f"  ✅ PASS: Fallback decisions (4 scenarios)")
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(("Fallback decisions", str(e)))
        print(f"  ❌ FAIL: Fallback decisions — {e}")
    
    # === Summary ===
    total = results["passed"] + results["failed"]
    print("\n" + "=" * 72)
    print(f"  RESULTS: {results['passed']}/{total} passed")
    if results["errors"]:
        print(f"\n  Failures:")
        for name, err in results["errors"]:
            print(f"    ❌ {name}: {err[:120]}")
    print("=" * 72 + "\n")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(run_all_tests())
    sys.exit(0 if results["failed"] == 0 else 1)
