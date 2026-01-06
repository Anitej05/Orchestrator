# Multi-File Operations Testing Guide

## Quick Start Testing

### Prerequisites
1. Backend running on port 8041
2. Orchestrator running
3. Two or more CSV/Excel files ready for upload

---

## Test Case 1: Compare Two Files

### Step 1: Upload Files
```bash
# Upload first file
curl -X POST http://localhost:8041/upload \
  -F "file=@sales_2023.csv"
# Response: {"success": true, "file_id": "uuid1", ...}

# Upload second file
curl -X POST http://localhost:8041/upload \
  -F "file=@sales_2024.csv"
# Response: {"success": true, "file_id": "uuid2", ...}
```

### Step 2: Compare Files (Direct API Call)
```bash
curl -X POST http://localhost:8041/compare \
  -H "Content-Type: application/json" \
  -d '{
    "file_ids": ["uuid1", "uuid2"],
    "comparison_mode": "schema_and_key",
    "output_format": "json"
  }'
```

### Expected Response
```json
{
  "success": true,
  "result": {
    "file_ids": ["uuid1", "uuid2"],
    "schema_diff": {
      "common_columns": ["Product", "Sales", "Date"],
      "unique_columns": {
        "uuid2": ["Region"]
      },
      "dtype_mismatches": {},
      "summary": "3 common columns, 1 new column in file2"
    },
    "row_diff": {
      "added": {"uuid2": 50},
      "removed": {"uuid1": 10},
      "changed": 5,
      "summary": "50 rows added, 10 removed, 5 changed"
    },
    "diff_artifact_id": "diff_report_uuid"
  },
  "canvas_display": {
    "canvas_type": "json",
    "canvas_data": {...}
  }
}
```

---

## Test Case 2: Merge Multiple Files

### Step 1: Upload Files
```bash
# Upload files with common key column
curl -X POST http://localhost:8041/upload -F "file=@customers_east.csv"
curl -X POST http://localhost:8041/upload -F "file=@customers_west.csv"
curl -X POST http://localhost:8041/upload -F "file=@customers_south.csv"
```

### Step 2: Merge Files (Direct API Call)
```bash
curl -X POST http://localhost:8041/merge \
  -H "Content-Type: application/json" \
  -d '{
    "file_ids": ["uuid1", "uuid2", "uuid3"],
    "merge_type": "union",
    "output_filename": "all_customers.csv"
  }'
```

### Expected Response
```json
{
  "success": true,
  "result": {
    "file_id": "merged_uuid",
    "filename": "all_customers.csv",
    "shape": [1500, 8],
    "summary": "Merged 3 files (union) → 1500 rows, 8 columns"
  },
  "canvas_display": {
    "canvas_type": "spreadsheet",
    "canvas_data": {
      "file_id": "merged_uuid",
      "preview": [...]
    }
  }
}
```

---

## Test Case 3: Orchestrator Auto-Injection

### Step 1: Upload Files via Orchestrator
```python
# In orchestrator conversation:
state["uploaded_files"] = [
  {"file_id": "uuid1", "file_type": "spreadsheet", "is_current_turn": True},
  {"file_id": "uuid2", "file_type": "spreadsheet", "is_current_turn": True}
]
```

### Step 2: Natural Language Request
User: "Compare these two files"

### Orchestrator Behavior
1. Detects instruction requires multi-file operation
2. Routes to `/compare` endpoint
3. **Auto-injects** `file_ids` from `uploaded_files` state
4. HTTP payload: `{"file_ids": ["uuid1", "uuid2"], "comparison_mode": "schema_and_key"}`

### Expected Log Output
```
✅ AUTO-INJECTED file_ids for multi-file operation: ['uuid1', 'uuid2']
Comparing 2 files: ['uuid1', 'uuid2']
Auto-detected key columns: ['ID']
Created diff report artifact: diff_report_uuid
```

---

## Test Case 4: Planner Integration

### Step 1: Use Natural Language with Planner
User: "Compare sales_2023.csv and sales_2024.csv, show me what changed"

### Step 2: Check Plan Proposal
```bash
curl -X POST http://localhost:8041/plan_operation \
  -F "file_id=uuid1" \
  -F "instruction=Compare sales_2023.csv and sales_2024.csv" \
  -F "stage=propose"
```

### Expected Plan Actions
```json
{
  "success": true,
  "result": {
    "plan_id": "plan_uuid",
    "actions": [
      {
        "action_type": "compare_files",
        "file_ids": ["uuid1", "uuid2"],
        "comparison_mode": "schema_and_key",
        "key_columns": null
      }
    ]
  },
  "canvas_display": {
    "requires_confirmation": true
  }
}
```

---

## Test Case 5: Merge with Key Columns

### Scenario: Join Two Customer Lists
**File 1**: `customers_basic.csv` (ID, Name, Email)  
**File 2**: `customers_orders.csv` (ID, Order_Count, Total_Spent)

### Merge Request
```bash
curl -X POST http://localhost:8041/merge \
  -H "Content-Type: application/json" \
  -d '{
    "file_ids": ["basic_uuid", "orders_uuid"],
    "merge_type": "join",
    "join_type": "inner",
    "key_columns": ["ID"]
  }'
```

### Expected Output
Merged file with columns: `ID, Name, Email, Order_Count, Total_Spent`

---

## Test Case 6: Error Handling

### 6A: Single File Provided (Should Fail)
```bash
curl -X POST http://localhost:8041/compare \
  -H "Content-Type: application/json" \
  -d '{"file_ids": ["uuid1"]}'
```

**Expected Error**:
```json
{
  "success": false,
  "error": "Comparison requires at least 2 files, got 1"
}
```

### 6B: Invalid File ID (Should Fail)
```bash
curl -X POST http://localhost:8041/compare \
  -H "Content-Type: application/json" \
  -d '{"file_ids": ["invalid_uuid", "uuid2"]}'
```

**Expected Error**:
```json
{
  "success": false,
  "error": "File invalid_uuid not found or not loaded"
}
```

### 6C: Merge with Mismatched Key Columns (Should Warn)
```bash
curl -X POST http://localhost:8041/merge \
  -H "Content-Type: application/json" \
  -d '{
    "file_ids": ["uuid1", "uuid2"],
    "merge_type": "join",
    "key_columns": ["CustomerID"]
  }'
# If "CustomerID" doesn't exist in one file:
```

**Expected Warning**:
```json
{
  "success": false,
  "error": "Key column 'CustomerID' not found in file uuid2"
}
```

---

## Performance Testing

### Large File Comparison (10K Rows)
```bash
# Measure threadpool effectiveness
time curl -X POST http://localhost:8041/compare \
  -H "Content-Type: application/json" \
  -d '{
    "file_ids": ["large_uuid1", "large_uuid2"],
    "comparison_mode": "full_diff"
  }'
```

**Expected**: Completes in <5s, API remains responsive to concurrent requests

### Concurrent Requests Test
```bash
# Run 5 comparison requests simultaneously
for i in {1..5}; do
  curl -X POST http://localhost:8041/compare \
    -H "Content-Type: application/json" \
    -d '{"file_ids": ["uuid1", "uuid2"]}' &
done
wait
```

**Expected**: All 5 requests complete without blocking each other

---

## Frontend Integration Testing

### Check Canvas Display Rendering

#### Compare Result (JSON Canvas)
```typescript
// Frontend should render:
const canvas_display = {
  canvas_type: "json",
  canvas_data: {
    file_ids: ["uuid1", "uuid2"],
    schema_diff: {...},
    row_diff: {...}
  }
};

// Render as:
// - Comparison table with schema differences
// - Row diff summary (added/removed/changed counts)
// - Link to download full diff report artifact
```

#### Merge Result (Spreadsheet Canvas)
```typescript
// Frontend should render:
const canvas_display = {
  canvas_type: "spreadsheet",
  canvas_data: {
    file_id: "merged_uuid",
    preview: [...],
    summary: "Merged 2 files"
  }
};

// Render as:
// - Data grid with merged preview
// - Download button for full merged file
// - Summary text at top
```

---

## Orchestrator Injection Testing

### Test Auto-Injection Logic

#### Python Test Script
```python
# Test orchestrator file_ids injection
state = {
    "uploaded_files": [
        {"file_id": "file1", "file_type": "spreadsheet", "is_current_turn": True},
        {"file_id": "file2", "file_type": "spreadsheet", "is_current_turn": True},
        {"file_id": "file3", "file_type": "document", "is_current_turn": True},
    ]
}

# Expected: Only file1 and file2 injected (file3 is not spreadsheet)
# pre_extracted_params["file_ids"] = ["file1", "file2"]
```

#### Check Logs
```bash
# Backend logs should show:
✅ AUTO-INJECTED file_ids for multi-file operation: ['file1', 'file2']
```

---

## Unit Tests Checklist

### `multi_file_ops.py` Functions

#### `compare_schemas()`
- [ ] 2 identical DataFrames → no differences
- [ ] Different dtypes → dtype_mismatches populated
- [ ] Different columns → unique_columns populated
- [ ] Different shapes → shape difference reported

#### `compare_by_keys()`
- [ ] Added rows → added_rows count correct
- [ ] Removed rows → removed_rows count correct
- [ ] Changed rows → changed_rows with details
- [ ] Mode="schema_and_key" → no value comparison
- [ ] Mode="full_diff" → value differences reported

#### `detect_key_columns()`
- [ ] "ID" column present → returns ["ID"]
- [ ] 90% unique column → detected as key
- [ ] No unique columns → returns first column
- [ ] Multiple unique columns → returns all

#### `merge_dataframes()`
- [ ] Join (inner) → correct row count
- [ ] Join (outer) → no data loss
- [ ] Union → only common columns
- [ ] Concat → all columns preserved
- [ ] Missing key columns → error raised

#### `generate_diff_report()`
- [ ] JSON format → valid JSON output
- [ ] CSV format → parseable CSV
- [ ] Markdown format → readable markdown

---

## Integration Tests Checklist

### Endpoints
- [ ] POST `/compare` with 2 files → success
- [ ] POST `/compare` with 5 files → success
- [ ] POST `/compare` with 1 file → error
- [ ] POST `/compare` with invalid file_id → error
- [ ] POST `/merge` with 2 files → success
- [ ] POST `/merge` with join=inner → correct result
- [ ] POST `/merge` with union → correct result
- [ ] POST `/merge` with concat → correct result
- [ ] POST `/merge` without key_columns (join) → error

### Orchestrator
- [ ] Orchestrator auto-injects file_ids from uploaded_files
- [ ] Orchestrator prioritizes is_current_turn=true files
- [ ] Orchestrator warns when <2 files available
- [ ] Orchestrator filters out non-spreadsheet files

### Planner
- [ ] Planner emits compare_files action for "compare X and Y"
- [ ] Planner emits merge_files action for "merge all files"
- [ ] Planner includes correct comparison_mode for "find differences"
- [ ] Planner includes correct merge_type for "join files"

---

## Debugging Tips

### Enable Debug Logging
```python
# In main.py:
logging.basicConfig(level=logging.DEBUG)
```

### Check Threadpool Execution
```python
# Logs should show:
# "Comparing 2 files: ['uuid1', 'uuid2']"
# "Auto-detected key columns: ['ID']"
# "Created diff report artifact: uuid"
```

### Verify File State
```python
# In Python REPL:
from agents.spreadsheet_agent.session import get_dataframe
df = get_dataframe("uuid1")
print(df.shape, df.columns.tolist())
```

### Check Artifact Registration
```bash
curl http://localhost:8041/stats
# Should show: "files_managed": 5 (including merged/diff files)
```

---

## Common Issues & Solutions

### Issue 1: "File not found"
**Cause**: File not loaded in session  
**Solution**: Call `/upload` first, or check `file_id` is correct

### Issue 2: "No key columns found"
**Cause**: No unique columns in DataFrame  
**Solution**: Provide `key_columns` explicitly in request

### Issue 3: Merge produces empty DataFrame
**Cause**: `join_type=inner` with no matching keys  
**Solution**: Use `join_type=outer` or check key column values

### Issue 4: Threadpool not working
**Cause**: `anyio` not installed  
**Solution**: `pip install anyio>=4.0.0`

### Issue 5: Orchestrator not injecting file_ids
**Cause**: Endpoint signature missing `file_ids` parameter  
**Solution**: Check endpoint definition in `Agent_entries/spreadsheet_agent.json`

---

## Success Criteria

✅ All test cases pass without errors  
✅ API remains responsive during large operations  
✅ Orchestrator correctly injects file_ids from state  
✅ Planner recognizes multi-file instructions  
✅ Canvas display renders correctly in frontend  
✅ Artifacts registered and downloadable  
✅ Error handling provides clear messages  

---

**Next Steps After Testing**:
1. Document any issues found
2. Add unit tests for all `multi_file_ops.py` functions
3. Load test with 50K+ row files
4. Frontend canvas rendering updates
5. Production deployment
