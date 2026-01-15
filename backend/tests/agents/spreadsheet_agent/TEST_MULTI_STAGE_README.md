# Multi-Stage Planning Test Suite

This directory contains comprehensive tests for the new multi-stage planning features.

## 📁 Test Files

### 1. `test_multi_stage_planning.py`
**Comprehensive unit and integration tests**

Tests all components of the multi-stage planning system:

- **Action Tests**: Filter, Sort, AddColumn, GroupBy, etc.
- **Parser Tests**: ActionParser and ActionExecutor
- **Planner Tests**: Propose, Revise, Simulate, Execute
- **LLM Tests**: Action generation via LLM
- **Performance Tests**: Large dataset handling
- **End-to-end Tests**: Complete 4-stage workflow

**Run:**
```bash
cd backend/tests/spreadsheet_agent
python test_multi_stage_planning.py
```

**Expected Output:**
```
🚀 MULTI-STAGE PLANNING TEST SUITE
================================================================================
Testing new 4-stage planning workflow...
Actions → Planner → LLM Integration
================================================================================

🧪 TEST: FilterAction
================================================================================
Original shape: (10, 5)
After filter (Price > 100): (4, 5)
✅ FilterAction tests passed

[... more tests ...]

📊 TEST SUMMARY
================================================================================
✅ Passed: 16
❌ Failed: 0
📈 Success Rate: 100.0%
```

---

### 2. `test_api_integration.py`
**HTTP API endpoint tests**

Tests the actual FastAPI endpoints via HTTP requests:

- `/upload` - File upload
- `/plan_operation` (stage=propose)
- `/plan_operation` (stage=revise)
- `/plan_operation` (stage=simulate)
- `/plan_operation` (stage=execute)
- `/simulate_operation` - Code simulation
- Error handling and edge cases

**Prerequisites:**
- Server must be running on `http://localhost:8041`

**Run:**
```bash
# Terminal 1: Start server
cd backend/agents/spreadsheet_agent
python main.py

# Terminal 2: Run tests
cd backend/tests/spreadsheet_agent
python test_api_integration.py
```

**Expected Output:**
```
🚀 MULTI-STAGE PLANNING API INTEGRATION TESTS
================================================================================
ℹ️  Checking if server is running...
✅ Server is running

ℹ️  Uploading test file...
✅ File uploaded: abc-123

Running: Complete Workflow (Propose→Simulate→Execute)
================================================================================
TEST: Propose Stage
================================================================================
✅ Plan proposed successfully
  Plan ID: 550e8400-e29b-41d4-a716-446655440000
  Actions: 2

[... more tests ...]

📊 TEST SUMMARY
================================================================================
Total Tests: 4
✅ Passed: 4
Success Rate: 100.0%
```

---

### 3. `test_quick.ps1`
**Quick PowerShell test script**

Fast curl-based test script for manual testing:

- Tests complete workflow
- Tests revision workflow
- Tests simulation endpoint
- Color-coded output
- Easy to read results

**Run:**
```powershell
cd backend\tests\spreadsheet_agent
.\test_quick.ps1
```

**Expected Output:**
```
================================================================================
Checking Server Status
================================================================================
✅ Server is running at http://localhost:8041

================================================================================
1. Uploading Test File
================================================================================
✅ File uploaded: file-123

================================================================================
2. STAGE 1: PROPOSE Plan
================================================================================
✅ Plan proposed: plan-456
ℹ️  Actions: 2

[... more stages ...]

📊 TEST SUMMARY
================================================================================
✅ All tests completed successfully!

Tested features:
  ✅ File upload
  ✅ Stage 1: Propose plan
  ✅ Stage 2: Simulate plan
  ✅ Stage 3: Execute plan
  ✅ Plan revision
  ✅ /simulate_operation endpoint
```

---

## 🎯 Test Coverage

### Action System
- ✅ FilterAction (numeric and string operators)
- ✅ SortAction (ascending/descending)
- ✅ AddColumnAction (formula parsing)
- ✅ RenameColumnAction
- ✅ DropColumnAction
- ✅ GroupByAction (with aggregates)
- ✅ FillNaAction (multiple methods)
- ✅ DropDuplicatesAction
- ✅ AddSerialNumberAction

### Parser & Executor
- ✅ ActionParser (JSON → action objects)
- ✅ ActionExecutor (validation + execution)
- ✅ Sequential action execution
- ✅ Error handling

### Multi-Stage Planner
- ✅ Stage 1: Propose (LLM + heuristic)
- ✅ Stage 2: Revise (with feedback)
- ✅ Stage 3: Simulate (on copy)
- ✅ Stage 4: Execute (on real data)
- ✅ Plan history tracking
- ✅ Error correction

### API Endpoints
- ✅ `/plan_operation` (all stages)
- ✅ `/simulate_operation`
- ✅ Error handling
- ✅ Request/response format

### Edge Cases
- ✅ Invalid column names
- ✅ Non-existent plan IDs
- ✅ Large datasets (10k+ rows)
- ✅ Complex action sequences
- ✅ LLM fallback scenarios

---

## 🚀 Quick Start

### Option 1: Run All Python Tests

```bash
# Install dependencies
pip install pandas requests

# Run comprehensive tests (no server needed)
cd backend/tests/spreadsheet_agent
python test_multi_stage_planning.py
```

### Option 2: Run API Tests

```bash
# Terminal 1: Start server
cd backend/agents/spreadsheet_agent
python main.py

# Terminal 2: Run API tests
cd backend/tests/spreadsheet_agent
python test_api_integration.py
```

### Option 3: Quick Manual Test

```powershell
# Start server first, then:
cd backend\tests\spreadsheet_agent
.\test_quick.ps1
```

---

## 📊 Test Data

Tests use data from:
- `backend/tests/test_data/sales_data.csv` (if exists)
- Auto-generated sample data (if not exists)

**Sample Data Structure:**
```csv
Date,Product,Amount
2024-01-01,Laptop,1200
2024-01-02,Mouse,25
2024-01-03,Keyboard,75
...
```

---

## 🔧 Troubleshooting

### Server Not Running
```
❌ Server not running at http://localhost:8041
```
**Fix:** Start the server:
```bash
cd backend/agents/spreadsheet_agent
python main.py
```

### Import Errors
```
ModuleNotFoundError: No module named 'agents'
```
**Fix:** Tests automatically add project root to path, but ensure you're running from the correct directory.

### LLM Tests Failing
```
⚠️  LLM action generation failed (may be expected if no API keys)
```
**Fix:** This is expected if LLM API keys are not configured. Core functionality will still work with heuristic fallback.

### Test Data Missing
```
❌ Test data not found
```
**Fix:** Tests will auto-create sample data, or ensure `backend/tests/test_data/sales_data.csv` exists.

---

## 🎓 Understanding Test Results

### ✅ All Green (100% Pass Rate)
- All features working correctly
- Multi-stage planning operational
- Ready for production use

### ⚠️ Some Yellow (Warnings)
- LLM tests may skip if no API keys
- Performance tests may vary by hardware
- Non-critical warnings in simulation

### ❌ Any Red (Failures)
- Core functionality broken
- Check error messages
- Review recent code changes
- Ensure server is running (for API tests)

---

## 📝 Adding New Tests

### Unit Test Template

```python
def test_new_feature():
    """Test description"""
    print("\n" + "="*80)
    print("🧪 TEST: New Feature")
    print("="*80)
    
    # Setup
    df = create_test_dataframe()
    
    # Test
    result = your_function(df)
    
    # Assertions
    assert result.success, "Should succeed"
    assert result.data is not None, "Should have data"
    
    print("✅ New feature test passed\n")
```

### API Test Template

```python
def test_new_endpoint(file_id: str) -> bool:
    """Test new endpoint"""
    print_header("TEST: New Endpoint")
    
    response = make_request("/new_endpoint", {
        "file_id": file_id,
        "param": "value"
    })
    
    if response["status_code"] != 200:
        print_error("Request failed")
        return False
    
    print_success("Endpoint working")
    return True
```

---

## 📚 Related Documentation

- [Multi-Stage Planning Architecture](../../../MULTI_STAGE_PLANNING.md)
- [Quick Start Guide](../../../MULTI_STAGE_QUICKSTART.md)
- [Spreadsheet Agent Improvements](../../../SPREADSHEET_AGENT_IMPROVEMENTS.md)
- [Testing Guide](../../../SPREADSHEET_AGENT_TESTING.md)

---

## 🎉 Success Criteria

Tests are considered successful when:

1. ✅ **Unit Tests**: All action classes work correctly
2. ✅ **Parser Tests**: JSON correctly converted to action objects
3. ✅ **Executor Tests**: Actions execute without errors
4. ✅ **Planner Tests**: All 4 stages complete successfully
5. ✅ **API Tests**: HTTP endpoints respond correctly
6. ✅ **Workflow Tests**: End-to-end scenarios work
7. ✅ **Error Tests**: Edge cases handled gracefully

**Target Pass Rate: 100%**

---

**Happy Testing! 🧪✨**
