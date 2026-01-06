import asyncio
import logging
import sys
import pandas as pd
from pathlib import Path
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import spreadsheet agent components directly
from agents.spreadsheet_agent.llm_agent import SpreadsheetQueryAgent
from agents.spreadsheet_agent.code_generator import generate_modification_code
from agents.spreadsheet_agent.utils import load_dataframe
from agents.spreadsheet_agent.session import store_dataframe, get_dataframe

# Create global query agent instance
query_agent_instance = SpreadsheetQueryAgent()

# ----------------------------------------------------------------------------
# Test data path resolution helpers
# ----------------------------------------------------------------------------
def resolve_test_file_path(rel_path: str) -> Path:
    """
    Resolve a dataset path to the actual backend/tests/test_data location.

    Handles inputs like 'tests/test_data/retail_sales_dataset.xlsx' regardless
    of current working directory, and normalizes Windows paths.
    """
    original = Path(rel_path)

    # Base dir: backend/tests/test_data relative to this file
    base_dir = Path(__file__).parent.parent / "test_data"

    # If the provided path already exists as-is, return it
    if original.exists():
        return original

    # If the path includes tests/test_data, map to our base_dir with filename
    try:
        name = original.name
    except Exception:
        name = str(original).split("/")[-1]

    candidate1 = base_dir / name
    if candidate1.exists():
        return candidate1

    # Fallback: explicit backend/tests/test_data under repo root
    repo_root = Path(__file__).parent.parent.parent.parent  # go up to workspace root
    candidate2 = repo_root / "backend" / "tests" / "test_data" / name
    if candidate2.exists():
        return candidate2

    # If a bare filename was given, try in base_dir
    if not original.parent or str(original.parent) == ".":
        candidate3 = base_dir / str(original)
        if candidate3.exists():
            return candidate3

    # Last resort: return candidate1 (preferred location) even if it doesn't exist
    return candidate1

def list_available_test_files() -> List[str]:
    """Return a list of files present in backend/tests/test_data for diagnostics."""
    base_dir = Path(__file__).parent.parent / "test_data"
    if not base_dir.exists():
        return []
    return [str(p.name) for p in base_dir.iterdir() if p.is_file()]

# ============================================================================
# QUERY DATASETS
# ============================================================================

RETAIL_SALES_QUERIES = {
    "simple": [
        # Basic info
        "How many rows are in this dataset?",
        "What columns does this dataset have?",
        "How many transactions are there in total?",
        
        # Simple aggregations
        "What is the total Total Amount?",
        "What is the maximum Total Amount?",
        "What is the minimum Total Amount?",
        "What is the average Total Amount?",
        
        # Count operations
        "How many unique Customer ID are there?",
        "How many unique Product Category are there?",
        "How many Male customers are there?",
        "How many Female customers are there?",
        
        # Basic statistics
        "What is the average Age of customers?",
        "What is the average Quantity purchased?",
        "What is the highest Price per Unit?",
    ],
    
    "medium": [
        # Group by + aggregation
        "What is the total Total Amount for each Product Category?",
        "What is the average Total Amount by Gender?",
        "What is the total Total Amount by Product Category?",
        "What is the average Age by Gender?",
        "What is the total Quantity sold for each Product Category?",
        
        # Ranking queries
        "Which Product Category has the highest total sales?",
        "Which Gender spends more on average?",
        "Show the top 5 transactions by Total Amount",
        "Show the top 3 Product Category by total revenue",
        
        # Filtering + aggregation
        "How many transactions have Total Amount greater than 50?",
        "What is the average Total Amount for customers over 30 years old?",
        "How many transactions are for Beauty products?",
        "What is the total sales for Electronics category?",
        
        # Date-based queries
        "What is the total Total Amount by month?",
        "How many transactions happened in January 2023?",
        
        # Multiple column grouping
        "What is the average Total Amount by Product Category and Gender?",
    ],
    
    "hard": [
        # Percentage calculations
        "What percentage of total sales does each Product Category represent?",
        "What percentage of customers are Male vs Female?",
        
        # Multi-level aggregation
        "For each Product Category, show total sales, average price, and transaction count",
        "Calculate total revenue, average transaction value, and customer count by Gender",
        
        # Complex filtering
        "Which customers have Total Amount above the average and made more than 2 purchases?",
        "Show Product Categories where average Total Amount is greater than 100",
        
        # Statistical analysis
        "What is the standard deviation of Total Amount?",
        "Calculate the correlation between Age and Total Amount",
        
        # Temporal analysis
        "Calculate month-over-month growth in Total Amount",
        "Which month had the highest total sales?",
        
        # Multi-criteria analysis
        "Show the average Total Amount by Product Category for customers over 25 years old",
        "Which Gender spends more in the Electronics category?",
        
        # Edge cases
        "Are there any missing values in the Total Amount column?",
        "How many duplicate Customer ID entries exist?",
    ]
}

ZARA_SALES_QUERIES = {
    "simple": [
        # Basic info
        "How many rows are in this dataset?",
        "How many products are in this dataset?",
        "What columns are available?",
        
        # Simple aggregations
        "What is the total Sales Volume?",
        "What is the average price?",
        "What is the maximum price?",
        "What is the minimum price?",
        
        # Count operations
        "How many unique section are there?",
        "How many unique Product Position are there?",
        "How many products have Promotion as Yes?",
        "How many products have Promotion as No?",
        "How many products are marked as Seasonal?",
        
        # Basic categorical counts
        "How many products are in the MAN section?",
        "How many products are in the WOMAN section?",
        "How many products are displayed at End-cap position?",
    ],
    
    "medium": [
        # Group by + aggregation
        "What is the total Sales Volume for each section?",
        "What is the average price by Product Position?",
        "What is the total Sales Volume by Promotion status?",
        "What is the average price for each section?",
        
        # Ranking queries
        "Which section has the highest total Sales Volume?",
        "Which Product Position has the highest average price?",
        "Show the top 10 products by Sales Volume",
        "Show the top 5 most expensive products",
        
        # Filtering + aggregation
        "What is the average price for products on Promotion?",
        "How many products have Sales Volume greater than 100?",
        "What is the total Sales Volume for Seasonal products?",
        "How many products are priced above 50 dollars?",
        
        # Comparison queries
        "What is the average price difference between promoted and non-promoted products?",
        "Compare average Sales Volume between Aisle and End-cap positions",
        
        # Multi-column grouping
        "What is the total Sales Volume by section and Product Position?",
        "What is the average price by section and Promotion status?",
    ],
    
    "hard": [
        # Percentage calculations
        "What percentage of total Sales Volume does each section represent?",
        "What percentage of products are on Promotion?",
        "What is the promotion rate by section?",
        
        # Multi-level aggregation
        "For each section, show total Sales Volume, average price, and product count",
        "Calculate total sales, average price, and promotion rate by Product Position",
        
        # Complex filtering
        "Which sections have above-average Sales Volume and more than 20 products?",
        "Show products where price is above average AND Sales Volume is above average",
        
        # Statistical analysis
        "What is the standard deviation of price by section?",
        "Calculate the price range (max - min) for each section?",
        
        # Multi-criteria analysis
        "What is the average price for Seasonal products on Promotion by section?",
        "Compare Sales Volume between MAN and WOMAN sections for promoted products only",
        "Which Product Position generates the most Sales Volume for non-Seasonal items?",
        
        # Promotion effectiveness
        "Calculate the average Sales Volume uplift for promoted vs non-promoted products",
        "Which section benefits most from promotions in terms of Sales Volume?",
        
        # Edge cases
        "Are there any products with zero Sales Volume?",
        "How many products have missing price information?",
        "Identify products with the same price but different Sales Volume",
    ]
}

FINANCIALS_QUERIES = {
    "simple": [
        # Basic info
        "How many rows are in this dataset?",
        "How many financial records are there?",
        "What columns does this dataset have?",
        
        # Simple aggregations
        "What is the total value in January?",
        "What is the total value in December?",
        "What is the maximum value in any month?",
        "What is the average value across all months?",
        
        # Count operations
        "How many unique Account types are there?",
        "How many unique Business Unit are there?",
        "How many unique Year values are there?",
        "How many records are for Scenario Actuals?",
        "How many records are for Scenario Forecast?",
        
        # Basic categorical queries
        "List all unique Account types",
        "How many records are for the year 2012?",
    ],
    
    "medium": [
        # Group by + aggregation
        "What is the total value across all months for each Account?",
        "What is the average January value by Business Unit?",
        "What is the total annual value by Year?",
        "What is the total value by Scenario?",
        
        # Ranking queries
        "Which Account has the highest total across all months?",
        "Which Business Unit has the highest total revenue?",
        "Which month has the highest average value across all accounts?",
        
        # Quarterly analysis
        "What is the total for Q1 (Jan, Feb, Mar)?",
        "What is the total for Q2 (Apr, May, Jun)?",
        "What is the total for Q3 (Jul, Aug, Sep)?",
        "What is the total for Q4 (Oct, Nov, Dec)?",
        
        # Filtering + aggregation
        "What is the total value for Sales account only?",
        "What is the total Cost of Goods Sold across all months?",
        "Show the total values for all Expense accounts",
        
        # Time-based comparisons
        "What is the total for the first half (Jan-Jun) vs second half (Jul-Dec)?",
        "Compare Q1 and Q4 totals",
        
        # Multi-column grouping
        "What is the total value by Account and Business Unit?",
        "What is the average monthly value by Year and Scenario?",
    ],
    
    "hard": [
        # Percentage calculations
        "What percentage of total annual value does each Account represent?",
        "What is the percentage contribution of each month to the annual total?",
        
        # Multi-level aggregation
        "For each Account, calculate total value, average monthly value, and standard deviation",
        "Show total revenue, total costs, and profit margin by Business Unit",
        
        # Trend analysis
        "Calculate the month-over-month growth rate",
        "Which months show positive growth compared to the previous month?",
        "Identify the month with the highest growth rate",
        
        # Complex filtering
        "Show Accounts where total annual value exceeds 1 million",
        "Which Business Units have higher Actuals than Forecast?",
        
        # Scenario comparison
        "Compare Actuals vs Forecast by Account",
        "Calculate the variance between Budget and Actuals for each month",
        "Which Accounts have the largest Budget vs Actuals variance?",
        
        # Statistical analysis
        "Calculate the coefficient of variation (std/mean) for each Account",
        "Which Account has the most consistent monthly values?",
        
        # Multi-criteria analysis
        "For Sales accounts only, compare Actuals vs Forecast by Business Unit",
        "Calculate average quarterly values by Account and Scenario",
        
        # Year-over-year analysis
        "Compare total values between 2012 and 2013 by Account",
        "Which Accounts showed growth from 2012 to 2013?",
        
        # Edge cases
        "Are there any months with negative values?",
        "Identify any missing monthly values",
        "Which Account-Business Unit combinations have zero values across all months?",
    ]
}

SALES_10K_QUERIES = {
    "simple": [
        # Basic info
        "How many rows are in this dataset?",
        "How many sales records are there?",
        "What columns are available in this dataset?",
        
        # Simple aggregations
        "What is the total Total Revenue?",
        "What is the total Total Profit?",
        "What is the total Total Cost?",
        "What is the maximum Total Revenue?",
        "What is the minimum Total Profit?",
        "What is the average Unit Price?",
        "What is the average Units Sold?",
        
        # Count operations
        "How many unique Region are there?",
        "How many unique Country are there?",
        "How many unique Item Type are there?",
        "How many orders were placed Online?",
        "How many orders were placed Offline?",
        "How many unique Sales Channel are there?",
        
        # Basic categorical counts
        "How many orders have priority L?",
        "How many orders have priority C?",
        "How many orders are from Europe region?",
    ],
    
    "medium": [
        # Group by + aggregation
        "What is the total Total Revenue for each Region?",
        "What is the total Total Profit by Item Type?",
        "What is the average Total Revenue by Sales Channel?",
        "What is the total Units Sold by Region?",
        "What is the average Unit Price by Item Type?",
        
        # Ranking queries
        "Which Region has the highest total Total Revenue?",
        "Which Country has the highest total Total Profit?",
        "Which Item Type generates the most Total Revenue?",
        "Show the top 10 countries by Total Profit",
        "Show the top 5 Item Types by Units Sold",
        
        # Filtering + aggregation
        "What is the total Total Revenue for Online sales only?",
        "What is the average Total Profit for orders with priority H?",
        "How many orders have Total Revenue greater than 10000?",
        "What is the total Total Profit for Office Supplies?",
        "What is the average Total Revenue for Offline channel?",
        
        # Comparison queries
        "Compare total Total Revenue between Online and Offline channels",
        "What is the average Total Profit difference between priority levels?",
        "Compare average Unit Price across different Sales Channels",
        
        # Date-based queries
        "What is the total Total Revenue by month?",
        "How many orders were placed in 2012?",
        "What is the average Total Profit by quarter?",
        
        # Multi-column grouping
        "What is the total Total Revenue by Region and Sales Channel?",
        "What is the average Total Profit by Item Type and Order Priority?",
        "What is the total Units Sold by Region and Item Type?",
    ],
    
    "hard": [
        # Profit margin calculations
        "Calculate the profit margin (Total Profit / Total Revenue) for each Item Type",
        "Which Item Type has the highest profit margin?",
        "What is the average profit margin by Region?",
        "Calculate profit margin by Sales Channel and compare Online vs Offline",
        
        # Percentage calculations
        "What percentage of total Total Revenue does each Region represent?",
        "What percentage of orders are placed Online vs Offline?",
        "What is the revenue distribution across different Item Types?",
        
        # Multi-level aggregation
        "For each Region, show total Total Revenue, total Total Profit, average profit margin, and order count",
        "Calculate total revenue, total cost, total profit, and units sold by Item Type",
        "Show total sales, average order value, and profit margin by Sales Channel",
        
        # Complex filtering
        "Which Countries have Total Revenue above 1 million and profit margin above 20%?",
        "Show Item Types where average Total Profit exceeds 5000 and Units Sold is above 100",
        "Identify Regions with Total Revenue above the global average",
        
        # Statistical analysis
        "What is the standard deviation of Total Profit by Region?",
        "Calculate the coefficient of variation for Total Revenue by Item Type",
        "Which Item Type has the most consistent profit margins?",
        
        # Temporal analysis
        "Calculate month-over-month growth in Total Revenue",
        "Which quarter had the highest total Total Profit?",
        "Identify seasonal trends in Units Sold by Item Type",
        "Compare Year-over-Year Total Revenue growth",
        
        # Multi-criteria analysis
        "What is the average Total Profit for Online Beverages sales in Europe?",
        "Compare profit margins between High and Low priority orders by Region",
        "Which Sales Channel is most profitable for each Item Type?",
        "Show the top 3 Countries by Total Revenue in each Region",
        
        # Cost efficiency analysis
        "Calculate the cost-to-revenue ratio by Item Type",
        "Which Region has the lowest average Unit Cost?",
        "Compare Unit Price vs Unit Cost margins across Sales Channels",
        
        # Order fulfillment analysis
        "Calculate average shipping time (Ship Date - Order Date) by Region",
        "Which Order Priority level has the fastest average shipping time?",
        
        # Edge cases and data quality
        "Are there any orders with negative Total Profit?",
        "Identify orders where Total Cost exceeds Total Revenue",
        "How many orders have missing Ship Date?",
        "Find duplicate Order ID entries",
        "Are there any outliers in Total Revenue (values beyond 3 standard deviations)?",
    ]
}

# Dataset registry
DATASET_REGISTRY = {
    "retail": {
        "filename": "tests/test_data/retail_sales_dataset.xlsx",
        "queries": RETAIL_SALES_QUERIES,
        "description": "Retail Sales Dataset (varies rows)",
        "default_difficulty": "medium"
    },
    "zara": {
        "filename": "tests/test_data/zara.xlsx",
        "queries": ZARA_SALES_QUERIES,
        "description": "Zara Product Catalog (254 products)",
        "default_difficulty": "medium"
    },
    "financials": {
        "filename": "tests/test_data/Financials Sample Data.xlsx",
        "queries": FINANCIALS_QUERIES,
        "description": "Financials Dataset (353 records)",
        "default_difficulty": "medium"
    },
    "sales_10k": {
        "filename": "tests/test_data/10000 Sales Records.xlsx",
        "queries": SALES_10K_QUERIES,
        "description": "Sales 10K Dataset (10,000 records)",
        "default_difficulty": "medium"
    },
    "salary": {
        "filename": "tests/test_data/Salary_Data.xlsx",
        "queries": {
            "simple": [
                "How many rows are in this dataset?",
                "What columns are available?",
                "How many employees are there?",
                "What is the average salary?",
                "What is the highest salary?",
                "What is the lowest salary?",
                "How many unique departments are there?",
            ],
            "medium": [
                "What is the total salary by department?",
                "What is the average salary by department?",
                "Which department has the highest average salary?",
                "Show the top 5 employees by salary",
                "How many employees have salary greater than 50000?",
                "What is the average years of experience?",
            ],
            "hard": [
                "Calculate salary growth potential by years of experience",
                "What is the standard deviation of salary by department?",
                "Identify the correlation between years of experience and salary",
                "Which department has the most consistent salaries?",
                "Calculate percentile distribution of salaries",
            ]
        },
        "description": "Salary Data (varies rows)",
        "default_difficulty": "simple"
    }
}

# ============================================================================
# TEST RESULT TRACKING
# ============================================================================

class TestResult:
    """Track individual query test results"""
    def __init__(self, dataset: str, difficulty: str, query_index: int, query: str):
        self.dataset = dataset
        self.difficulty = difficulty
        self.query_index = query_index
        self.query = query
        self.success = False
        self.answer = ""
        self.error = ""
        self.parse_failures = 0
        self.provider_used = ""
        self.duration_ms = 0.0
        self.iterations = 0
        self.tokens_input = 0
        self.tokens_output = 0
        self.steps = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "dataset": self.dataset,
            "difficulty": self.difficulty,
            "query_index": self.query_index,
            "query": self.query,
            "success": self.success,
            "answer": self.answer[:200] if self.answer else "",
            "error": self.error[:200] if self.error else "",
            "parse_failures": self.parse_failures,
            "provider_used": self.provider_used,
            "duration_ms": round(self.duration_ms, 2),
            "iterations": self.iterations,
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": self.tokens_input + self.tokens_output
            }
        }


class TestSummary:
    """Aggregate results across all queries"""
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_duration_ms = 0.0
        self.total_tokens = 0

    def add_result(self, result: TestResult):
        self.results.append(result)
        self.total_queries += 1
        if result.success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        self.total_duration_ms += result.duration_ms
        self.total_tokens += result.tokens_input + result.tokens_output

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        success_rate = (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0
        return {
            "total_queries": self.total_queries,
            "successful": self.successful_queries,
            "failed": self.failed_queries,
            "success_rate": round(success_rate, 1),
            "avg_duration_ms": round(self.total_duration_ms / self.total_queries, 2) if self.total_queries > 0 else 0,
            "total_tokens": self.total_tokens,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "elapsed_time": round(time.time() - self.start_time, 2)
        }

    def print_summary(self):
        """Print formatted summary"""
        stats = self.get_stats()
        print("\n" + "="*70)
        print("TEST EXECUTION SUMMARY")
        print("="*70)
        print(f"\n[SUCCESSFUL]:  {self.successful_queries}/{self.total_queries} ({stats['success_rate']:.1f}%)")
        print(f"[FAILED]:      {self.failed_queries}/{self.total_queries}")
        print(f"Avg Duration: {stats['avg_duration_ms']:.2f} ms")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Elapsed Time: {stats['elapsed_time']:.2f}s")
        print("="*70)


async def run_dataset_tests(dataset_key: str, difficulty: str = None, output_format: str = "console") -> TestSummary:
    """Run tests on a specific dataset"""
    if dataset_key not in DATASET_REGISTRY:
        print(f"❌ Unknown dataset: {dataset_key}")
        print(f"   Available: {', '.join(DATASET_REGISTRY.keys())}")
        return TestSummary()

    dataset_info = DATASET_REGISTRY[dataset_key]
    test_file = dataset_info["filename"]
    resolved_path = resolve_test_file_path(test_file)
    difficulty = difficulty or dataset_info["default_difficulty"]

    if difficulty not in dataset_info["queries"]:
        print(f"❌ Unknown difficulty level: {difficulty}")
        print(f"   Available: {', '.join(dataset_info['queries'].keys())}")
        return TestSummary()

    # Check file exists (using resolved path)
    if not Path(resolved_path).exists():
        print(f"WARNING: File not found: {test_file}")
        print(f"Resolved location tried: {resolved_path}")
        # Show additional attempted canonical path
        alt_backend_path = Path(__file__).parent.parent / "test_data" / Path(test_file).name
        print(f"Alternate backend path: {alt_backend_path}")
        # List available files to aid selection
        available = list_available_test_files()
        if available:
            print(f"Available files in backend/tests/test_data: {', '.join(available)}")
        else:
            print("No files found in backend/tests/test_data")
        return TestSummary()

    # Load dataframe
    print("\n" + "*"*35)
    print(f"Testing Dataset: {dataset_key.upper()}")
    print(f"   Description: {dataset_info['description']}")
    print(f"   File: {resolved_path}")
    print(f"   Difficulty: {difficulty.upper()}")
    print("*"*35 + "\n")

    try:
        df = load_dataframe(str(resolved_path))
        print(f"Loaded: {len(df)} rows x {len(df.columns)} columns")
        print(f"   Columns: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}")
        print(f"   Dtypes: {', '.join([f'{col}({df[col].dtype})' for col in df.columns.tolist()[:3]])}{'...' if len(df.columns) > 3 else ''}")
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return TestSummary()

    # Run queries
    queries = dataset_info["queries"][difficulty]
    summary = TestSummary()
    thread_id = f"test-{dataset_key}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    for idx, query in enumerate(queries, 1):
        result = TestResult(dataset_key, difficulty, idx, query)
        print("\n" + "-"*70)
        print(f"[{idx}/{len(queries)}] {query}")
        print("-"*70)
        
        start_time = time.time()
        try:
            # Execute query
            llm_result = await query_agent_instance.query(
                df=df,
                question=query,
                max_iterations=5,
                session_context="",
                file_id=None,
                thread_id=thread_id
            )

            result.duration_ms = (time.time() - start_time) * 1000
            result.success = llm_result.success
            result.answer = llm_result.answer or ""
            result.error = llm_result.error or ""
            result.iterations = len(llm_result.steps_taken) if hasattr(llm_result, 'steps_taken') else 0

            # Extract metrics if available
            if hasattr(llm_result, 'execution_metrics') and llm_result.execution_metrics:
                metrics = llm_result.execution_metrics
                result.tokens_input = metrics.get('tokens_input', 0)
                result.tokens_output = metrics.get('tokens_output', 0)
                if hasattr(llm_result, 'provider_used'):
                    result.provider_used = llm_result.provider_used

            # Detailed step logging with code analysis
            if hasattr(llm_result, 'steps_taken') and llm_result.steps_taken:
                print(f"\nExecution Steps ({len(llm_result.steps_taken)} total):")
                for step_idx, step in enumerate(llm_result.steps_taken, 1):
                    step_type = step.get('step_type', 'unknown')
                    provider = step.get('provider', 'unknown')
                    print(f"\n   Step {step_idx}: {step_type} [{provider}]")
                    
                    # Show generated code with library analysis
                    if 'code' in step:
                        code = step['code']
                        # Detect libraries used
                        libraries = []
                        if 'df.' in code or 'pandas' in code:
                            libraries.append('pandas')
                        if 'np.' in code or 'numpy' in code:
                            libraries.append('numpy')
                        if 'plt.' in code or 'matplotlib' in code:
                            libraries.append('matplotlib')
                        if 'sql' in code.lower():
                            libraries.append('SQL')
                        
                        lib_str = f" [Libraries: {', '.join(libraries)}]" if libraries else ""
                        print(f"      Code{lib_str}:")
                        code_display = code[:200]
                        print(f"         {code_display}{'...' if len(code) > 200 else ''}")
                    
                    # Show result/output
                    if 'output' in step:
                        output = str(step['output'])[:100]
                        print(f"      Output: {output}{'...' if len(str(step.get('output', ''))) > 100 else ''}")
                    
                    # Show errors if any
                    if 'error' in step and step['error']:
                        error = step['error'][:150]
                        print(f"      ERROR: {error}{'...' if len(step.get('error', '')) > 150 else ''}")
                        if 'JSON' in step.get('error', ''):
                            result.parse_failures += 1
                    
                    # Show parse attempts if any
                    if 'parse_attempts' in step and step['parse_attempts'] > 0:
                        print(f"      Parse attempts: {step['parse_attempts']}")

            if result.success:
                print(f"\n[SUCCESS]")
                print(f"   Duration: {result.duration_ms:.0f}ms")
                print(f"   Iterations: {result.iterations}")
                print(f"   Provider: {result.provider_used if result.provider_used else 'default'}")
                print(f"   Tokens: {result.tokens_input + result.tokens_output:,} (in:{result.tokens_input} out:{result.tokens_output})")
                
                # Show answer with better formatting
                if result.answer:
                    print(f"\n   RESULT:")
                    # Try to format if it looks like structured data
                    answer_lines = result.answer.split('\n')
                    for line in answer_lines[:10]:  # Show first 10 lines
                        print(f"      {line}")
                    if len(answer_lines) > 10:
                        print(f"      ... ({len(answer_lines) - 10} more lines)")
            else:
                print(f"\n[FAILED]")
                print(f"   Duration: {result.duration_ms:.0f}ms")
                print(f"   Iterations: {result.iterations}")
                print(f"   Parse Failures: {result.parse_failures}")
                
                # Show primary error
                if result.error:
                    error_msg = result.error
                    print(f"\n   PRIMARY ERROR:")
                    error_lines = error_msg.split('\n')
                    for line in error_lines[:5]:
                        print(f"      {line}")
                    if len(error_lines) > 5:
                        print(f"      ... ({len(error_lines) - 5} more lines)")
                
                # Show failure context from steps with code analysis
                if hasattr(llm_result, 'steps_taken') and llm_result.steps_taken:
                    print(f"\n   EXECUTION STEPS:")
                    for step_idx, step in enumerate(llm_result.steps_taken, 1):
                        status = "[OK]" if not step.get('error') else "[FAIL]"
                        print(f"      {status} Step {step_idx}: {step.get('step_type', 'unknown')} [{step.get('provider', 'unknown')}]")
                        
                        # Show what code was attempted with library detection
                        if 'code' in step and step['code']:
                            code = step['code']
                            # Detect libraries
                            libraries = []
                            if 'df.' in code or 'pandas' in code:
                                libraries.append('pandas')
                            if 'np.' in code or 'numpy' in code:
                                libraries.append('numpy')
                            if 'sql' in code.lower():
                                libraries.append('SQL')
                            lib_str = f" [{', '.join(libraries)}]" if libraries else ""
                            print(f"         Code{lib_str}: {code[:150]}{'...' if len(code) > 150 else ''}")
                        
                        # Show step-level errors
                        if 'error' in step and step['error']:
                            step_error = step['error'][:120]
                            print(f"         Error: {step_error}{'...' if len(step.get('error', '')) > 120 else ''}")
                        
                        # Show output if available
                        if 'output' in step and step['output'] and not step.get('error'):
                            output = str(step['output'])[:80]
                            print(f"         Result: {output}{'...' if len(str(step['output'])) > 80 else ''}")
                    
                    # Highlight the failing step
                    failing_step = None
                    for step in reversed(llm_result.steps_taken):
                        if step.get('error'):
                            failing_step = step
                            break
                    
                    if failing_step:
                        print(f"\n   FAILURE POINT:")
                        print(f"      Step Type: {failing_step.get('step_type', 'unknown')}")
                        print(f"      Provider: {failing_step.get('provider', 'unknown')}")
                        if 'code' in failing_step:
                            code = failing_step['code']
                            # Detect libraries in failing code
                            libraries = []
                            if 'df.' in code:
                                libraries.append('pandas')
                            if 'np.' in code:
                                libraries.append('numpy')
                            lib_str = f" [{', '.join(libraries)}]" if libraries else ""
                            print(f"      Attempted Code{lib_str}: {code[:150]}...")
                        if 'error' in failing_step:
                            print(f"      Error: {failing_step['error'][:200]}...")
                else:
                    # No step history, show the full error
                    print(f"\n   ERROR DETAILS:")
                    error_lines = result.error.split('\n')
                    for line in error_lines[:10]:
                        print(f"      {line}")
                    if len(error_lines) > 10:
                        print(f"      ... ({len(error_lines) - 10} more lines)")

        except Exception as e:
            result.duration_ms = (time.time() - start_time) * 1000
            result.error = str(e)
            print(f"\n[EXCEPTION]")
            print(f"   Duration: {result.duration_ms:.0f}ms")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Error: {str(e)[:200]}")
            
            # Show full traceback for debugging
            import traceback
            print(f"\n   Traceback:")
            tb_lines = traceback.format_exc().split('\n')
            for line in tb_lines[-5:]:
                if line.strip():
                    print(f"      {line}")

        summary.add_result(result)

    # Print summary
    summary.print_summary()

    # Optionally save results
    if output_format == "json":
        report_file = Path(f"test_results_{dataset_key}_{difficulty}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        results_data = {
            "dataset": dataset_key,
            "difficulty": difficulty,
            "timestamp": datetime.now().isoformat(),
            "stats": summary.get_stats(),
            "results": [r.to_dict() for r in summary.results]
        }
        with open(report_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {report_file.absolute()}")

    return summary


def print_available_datasets():
    """List all available datasets and their queries"""
    print("\n" + "="*70)
    print("AVAILABLE DATASETS")
    print("="*70)
    for key, info in DATASET_REGISTRY.items():
        print(f"\n- {key.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   File: {info['filename']}")
        print(f"   Difficulties: {', '.join(info['queries'].keys())}")
        print(f"   Queries: {sum(len(v) for v in info['queries'].values())} total")


async def main():
    """Main entry point with CLI argument support"""
    parser = argparse.ArgumentParser(
        description="Test Spreadsheet Agent with multiple datasets and query complexities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_spreadsheet_manual.py --list
  python test_spreadsheet_manual.py --dataset zara --difficulty medium
  python test_spreadsheet_manual.py --dataset salary --difficulty simple --output json
  python test_spreadsheet_manual.py --dataset retail --difficulty hard --output json
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--dataset", type=str, help="Dataset to test (zara, retail, financials, sales_10k, salary)")
    parser.add_argument("--difficulty", type=str, choices=["simple", "medium", "hard"], help="Query difficulty level")
    parser.add_argument("--output", type=str, choices=["console", "json"], default="console", help="Output format")
    
    args = parser.parse_args()
    
    if args.list:
        print_available_datasets()
        return

    if not args.dataset:
        print("Please specify a dataset (--dataset) or use --list to see options")
        parser.print_help()
        return

    # Run tests
    summary = await run_dataset_tests(args.dataset, args.difficulty, args.output)

    # Display final message
    if summary.successful_queries > 0:
        print(f"\nCompleted {summary.successful_queries} successful queries!")
    else:
        print(f"\nNo successful queries. Check dataset file and LLM configuration.")


if __name__ == "__main__":
    asyncio.run(main())
