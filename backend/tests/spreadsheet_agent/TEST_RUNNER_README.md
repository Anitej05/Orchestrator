# Spreadsheet Agent - Parameterized Test Runner

## Overview
The refactored `test_spreadsheet_manual.py` provides a dataset-driven testing framework for the Spreadsheet Agent, supporting multiple datasets with categorized queries across three difficulty levels (simple, medium, hard).

## Features
✅ **5 Datasets**: Retail Sales, Zara Catalog, Financials, Sales 10K, Salary Data  
✅ **240+ Queries**: Categorized by difficulty and functionality  
✅ **Result Tracking**: Per-query metrics (duration, iterations, parse failures, tokens)  
✅ **Continue-on-Failure Mode**: Tests all queries and reports all failures at end  
✅ **Console & JSON Output**: Print to console or save results as JSON  
✅ **CLI Arguments**: Easy control via command-line options  

## Usage

### List Available Datasets
```bash
python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py --list
```

**Output:**
```
📚 AVAILABLE DATASETS
══════════════════════════════════════════════════════════════════════

🔹 RETAIL
   Description: Retail Sales Dataset (varies rows)
   File: backend/tests/test_data/retail_sales_dataset.xlsx
   Difficulties: simple, medium, hard
   Queries: 44 total

🔹 ZARA
   Description: Zara Product Catalog (254 products)
   File: backend/tests/test_data/zara.xlsx
   Difficulties: simple, medium, hard
   Queries: 48 total

🔹 FINANCIALS
   Description: Financials Dataset (353 records)
   File: backend/tests/test_data/Financials Sample Data.xlsx
   Difficulties: simple, medium, hard
   Queries: 53 total

🔹 SALES_10K
   Description: Sales 10K Dataset (10,000 records)
   File: backend/tests/test_data/10000 Sales Records.xlsx
   Difficulties: simple, medium, hard
   Queries: 77 total

🔹 SALARY
   Description: Salary Data (varies rows)
   File: backend/tests/test_data/Salary_Data.xlsx
   Difficulties: simple, medium, hard
   Queries: 18 total
```

### Run Tests on a Dataset

#### Test Zara Dataset (Medium Difficulty)
```bash
python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py --dataset zara --difficulty medium
```

#### Test Salary Dataset (Simple Difficulty)
```bash
python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py --dataset salary --difficulty simple
```

#### Test Retail Dataset (Hard Difficulty)
```bash
python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py --dataset retail --difficulty hard
```

### Save Results to JSON
```bash
python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py --dataset zara --difficulty simple --output json
```

**Output file location**: `test_results_zara_simple_20260102_172844.json`

**JSON format**:
```json
{
  "dataset": "zara",
  "difficulty": "simple",
  "timestamp": "2026-01-02T17:28:44.123456",
  "stats": {
    "total_queries": 15,
    "successful": 14,
    "failed": 1,
    "success_rate": 93.3,
    "avg_duration_ms": 645.32,
    "total_tokens": 28542,
    "total_duration_ms": 9679.8,
    "elapsed_time": 125.4
  },
  "results": [
    {
      "dataset": "zara",
      "difficulty": "simple",
      "query_index": 1,
      "query": "How many rows are in this dataset?",
      "success": true,
      "answer": "The dataset contains **252 rows**.",
      "error": "",
      "parse_failures": 0,
      "provider_used": "cerebras",
      "duration_ms": 1636.0,
      "iterations": 2,
      "tokens": {
        "input": 3730,
        "output": 368,
        "total": 4098
      }
    },
    ...
  ]
}
```

## Test Results Summary

After each test run, you'll see a summary:

```
══════════════════════════════════════════════════════════════════════
📊 TEST EXECUTION SUMMARY
══════════════════════════════════════════════════════════════════════

✅ Successful:  14/15 (93.3%)
❌ Failed:      1/15
⏱️  Avg Duration: 645.32 ms
📝 Total Tokens: 28,542
⏳ Elapsed Time: 125.40s
══════════════════════════════════════════════════════════════════════
```

## Query Categories

### Simple Queries
- **Purpose**: Validate basic spreadsheet operations  
- **Examples**: Row counts, column lists, basic aggregations  
- **Expected**: ~500ms per query, 0 retries  

### Medium Queries
- **Purpose**: Test grouping, ranking, filtering operations  
- **Examples**: Group-by aggregations, top-N results, conditional filtering  
- **Expected**: ~800ms per query, occasional retries  

### Hard Queries
- **Purpose**: Complex multi-step analysis and calculations  
- **Examples**: Statistical analysis, percentage calculations, multi-criteria filtering  
- **Expected**: ~1200ms per query, may require retries  

## Interpreting Results

### Success Indicators
✅ `success`: true  
✅ `duration_ms < 1500` (varies by query complexity)  
✅ `parse_failures == 0`  
✅ `iterations <= 3` (usually 1-2 for well-formed queries)  

### Failure Indicators
❌ `success`: false  
❌ `error` field contains error message  
❌ `parse_failures > 0` (JSON parsing issues)  
❌ `iterations == max_iterations` (query hit iteration limit)  

### Token Usage
- **Simple queries**: 1,800-2,500 tokens  
- **Medium queries**: 2,000-3,500 tokens  
- **Hard queries**: 2,500-5,000 tokens  

## Troubleshooting

### File Not Found
```
⚠️  File not found: backend/tests/test_data/zara.xlsx
📁 Expected location: C:\...\backend\tests\test_data\zara.xlsx
```
**Solution**: Verify files exist in `backend/tests/test_data/`

### All Queries Failed
Likely causes:
- LLM provider offline or API key invalid
- Dataset file corrupted or missing columns
- Memory/resource constraint

Check logs for detailed error messages in the console output.

### JSON Parse Errors
Indicates the LLM response wasn't valid JSON. The agent includes fallback parsing that extracts JSON from malformed responses, but some edge cases may still fail. This is tracked in `parse_failures`.

## Performance Expectations

| Dataset | Rows | Simple Avg | Medium Avg | Hard Avg |
|---------|------|-----------|-----------|----------|
| Zara | 254 | 580ms | 850ms | 1200ms |
| Retail | ~5000 | 600ms | 900ms | 1300ms |
| Financials | 353 | 620ms | 920ms | 1400ms |
| Sales 10K | 10000 | 700ms | 1000ms | 1500ms |
| Salary | ~100 | 550ms | 800ms | 1100ms |

## Configuration

### Available Datasets
Located in `DATASET_REGISTRY` (lines 420-490):
- `retail`: Retail Sales Dataset
- `zara`: Zara Product Catalog
- `financials`: Financial Records
- `sales_10k`: Large Sales Dataset (10K records)
- `salary`: Employee Salary Data

### Adding New Datasets
1. Create/add Excel file to `backend/tests/test_data/`
2. Add query dictionary (simple, medium, hard)
3. Register in `DATASET_REGISTRY`
4. Re-run with `--list` to verify

## Example Workflows

### Quick Validation (5 min)
```bash
# Test all simple queries on all datasets
python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py --dataset zara --difficulty simple
```

### Comprehensive Testing (30+ min)
```bash
# Test one dataset across all difficulties
python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py --dataset retail --difficulty hard --output json
```

### Batch Testing (Use shell script)
```bash
# Create a script to test all datasets
for dataset in retail zara financials sales_10k salary; do
  python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py \
    --dataset $dataset --difficulty simple --output json
done
```

## Output Files

When using `--output json`, results are saved as:
```
test_results_<dataset>_<difficulty>_<timestamp>.json
```

Example: `test_results_zara_simple_20260102_172844.json`

Files are created in the current working directory (project root when run from repo).

## Notes

- Tests continue on failure (mode B: report-all)
- Each query creates a new LLM conversation
- Results are tracked per-query with detailed metrics
- JSON output includes full results for post-analysis
- Console output is real-time, JSON is final summary

---

**Last Updated**: January 2, 2026  
**Test Runner Version**: 2.0 (Parameterized)
