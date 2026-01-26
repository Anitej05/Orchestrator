# Spreadsheet Agent v1 vs v2 Gap Analysis

## Purpose

This document analyzes what features exist in the legacy `spreadsheet_agent/` that may be missing from `spreadsheet_agent_v2/` to ensure safe migration.

---

## 1. Endpoint Comparison

### Old Agent Endpoints (main.py - 46 functions)

| Endpoint | v1 | v2 | Status |
|----------|:--:|:--:|--------|
| `POST /upload` | ✅ | via `/execute` | ✅ **Covered** |
| `POST /query` | ✅ | via `/execute` | ✅ **Covered** |
| `POST /transform` | ✅ | via `/execute` | ✅ **Covered** |
| `GET /summary` | ✅ | via `/execute` query | ✅ **Covered** |
| `GET /download/{file_id}` | ✅ | via `_step_export` | ✅ **Covered** |
| `POST /execute_pandas` | ✅ | via `_step_transform` | ✅ **Covered** |
| `GET /display_spreadsheet` | ✅ | via `canvas_display` | ✅ **Covered** |
| `GET /column_stats` | ✅ | via `/execute` query | ✅ **Covered** |
| `GET /health` | ✅ | ✅ | ✅ **Covered** |
| `POST /execute` | ✅ | ✅ | ✅ **Covered** |
| `POST /continue` | ✅ | ✅ | ✅ **Covered** |
| `GET /files` | ✅ | ✅ | ✅ **Covered** |
| `GET /files/{file_id}` | ✅ | ✅ | ✅ **Covered** |
| `DELETE /files/{file_id}` | ✅ | ❌ | ⚠️ **Missing** |
| `POST /reload_file` | ✅ | ❌ | 🔸 Not needed |
| `POST /cleanup_files` | ✅ | via session expiry | ✅ **Covered** |
| `POST /compare_files` | ✅ | ❌ | ⚠️ **Missing** |
| `POST /merge_files` | ✅ | ❌ | ⚠️ **Missing** |
| `POST /create_spreadsheet` | ✅ | ❌ | ⚠️ **Missing** |
| `POST /plan_operation` | ✅ | ❌ | 🔸 Replaced by LLM |
| `POST /simulate_operation` | ✅ | ❌ | 🔸 Replaced by retry |
| `GET /metrics` | ✅ | ❌ | 🔸 Nice-to-have |
| `GET /stats` | ✅ | via `/health` | ✅ **Covered** |
| `GET /performance_report` | ✅ | ❌ | 🔸 Nice-to-have |

### Summary

| Category | Count |
|----------|-------|
| ✅ Covered in v2 | 14 |
| ⚠️ Missing (may need) | 4 |
| 🔸 Not needed / Replaced | 5 |

---

## 2. Features Missing from v2

### 2.1 Critical Missing Features

| Feature | Description | Priority | Effort |
|---------|-------------|----------|--------|
| **compare_files** | Compare 2+ spreadsheets, find diffs | Medium | 2-3 hours |
| **merge_files** | Join/union multiple spreadsheets | Medium | 2-3 hours |
| **delete_file** | Remove file from session | Low | 30 mins |
| **create_spreadsheet** | Create new spreadsheet from scratch | Low | 1-2 hours |

### 2.2 Nice-to-Have Missing Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Metrics endpoint | Detailed API call metrics | Low |
| Performance report | Query/operation timing stats | Low |
| Simulate operation | Preview changes before apply | Low (LLM retry handles this) |
| Plan operation stages | propose/revise/simulate/execute workflow | Low (LLM handles automatically) |

---

## 3. Feature Coverage Analysis

### 3.1 Query Operations

| Operation | v1 Method | v2 Method |
|-----------|-----------|-----------|
| Natural language query | `natural_language_query()` | `_step_query()` |
| Pandas code execution | `execute_pandas()` | `_step_transform()` with sandboxed exec |
| Column statistics | `get_column_stats()` | Via query action |
| Aggregations | `_handle_aggregate()` | `_step_aggregate()` |
| Filtering | `_handle_filter()` | `_step_filter()` |
| Sorting | `_handle_sort()` | `_step_sort()` |

### 3.2 Transformation Operations

| Operation | v1 | v2 |
|-----------|----|----|
| Add column | `transform_data()` | `_step_add_column()` |
| Drop column | `transform_data()` | `_step_drop_column()` |
| Custom transform | `execute_pandas()` | `_step_transform()` |
| Rename columns | via transform | via LLM code generation |

### 3.3 File Operations

| Operation | v1 | v2 |
|-----------|----|----|
| Upload file | `upload_file()` | `_handle_file_upload()` |
| Load from path | via session | `_step_load_file()` |
| Export to file | `download_spreadsheet()` | `_step_export()` |
| List files | `list_files()` | `/files` endpoint |
| Get file info | `get_file_info()` | `/files/{file_id}` endpoint |
| Delete file | `delete_file_endpoint()` | ❌ **Missing** |

### 3.4 Multi-File Operations

| Operation | v1 | v2 |
|-----------|----|----|
| Compare files | `compare_files()` | ❌ **Missing** |
| Merge files | `merge_files()` | ❌ **Missing** |
| Cross-file queries | Limited support | Via LLM (can query multiple) |

---

## 4. Recommendation

### 4.1 Safe to Migrate Now

The v2 agent covers **all core functionality** for:
- Single-file operations (upload, query, transform, export)
- LLM-powered natural language queries
- Session management
- Canvas display for frontend
- Orchestrator integration

### 4.2 Features to Add Before Full Migration

If multi-file operations are required:

1. **Add `delete_file` endpoint** (~30 mins)
   - Simple session cleanup

2. **Add `compare_files` action** (~2 hours)
   - Can be done via LLM by loading both files

3. **Add `merge_files` action** (~2 hours)
   - Can be done via LLM transform step

### 4.3 Features NOT Needed

These v1 features are intentionally NOT in v2:

1. **Plan/Simulate stages** - LLM automatically handles planning with retry loop
2. **Metrics/Performance endpoints** - Can be added later if needed
3. **Reload file** - Files are automatically loaded on access
4. **Complex anomaly detection** - Removed for simplicity

---

## 5. Conclusion

| Verdict | Details |
|---------|---------|
| **Safe to migrate?** | ✅ Yes, for single-file workflows |
| **Multi-file support?** | ⚠️ Need to add compare/merge |
| **Blocking issues?** | None for typical usage |
| **Recommended action** | Migrate, add multi-file later if needed |

The v2 agent is a **complete replacement** for all common use cases. The legacy agent can be archived once multi-file operations are added (if required).
