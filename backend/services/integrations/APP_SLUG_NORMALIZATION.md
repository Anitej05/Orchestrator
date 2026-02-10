# App Slug Normalization Guide

## Overview

All Composio integration app slugs are now normalized to lowercase without underscores for consistency.

## Normalized App Slugs

| App | Normalized Slug | Deprecated Slug | Status |
|-----|----------------|-----------------|--------|
| Zoho Books | `zohobooks` | `zoho_books` | ⚠️ Deprecated |
| Gmail | `gmail` | - | ✅ Current |
| GitHub | `github` | - | ✅ Current |
| Slack | `slack` | - | ✅ Current |
| Notion | `notion` | - | ✅ Current |

## Migration Guide

### For Developers

**If you're using `zoho_books` anywhere in your code:**

```python
# ❌ OLD (deprecated)
auth_manager.start_auth_flow(user_id, "zoho_books")

# ✅ NEW (recommended)
auth_manager.start_auth_flow(user_id, "zohobooks")
```

**The system will:**
- Accept both `zoho_books` and `zohobooks`
- Log a deprecation warning when `zoho_books` is used
- Automatically normalize to `zohobooks` internally

### For Users/Deployers

**Environment variables remain unchanged:**

```bash
# .env file - no changes needed
COMPOSIO_AUTH_CONFIG_ZOHOBOOKS=ac_zohobooks_xxx
```

**Note:** The env var uses `ZOHOBOOKS` (no underscore) regardless of which slug you use in code.

## Why This Change?

1. **Consistency**: All Composio app slugs use lowercase without underscores (gmail, github, slack, notion)
2. **Simplification**: Removes confusion about which slug format to use
3. **Best Practice**: Aligns with Composio's official naming convention

## Deprecation Timeline

- **Current**: Both `zoho_books` and `zohobooks` work (with warning)
- **Next Version**: `zoho_books` will be removed entirely
- **Action Required**: Update all code to use `zohobooks`

## How to Update

1. Search your codebase for `"zoho_books"` (with underscore)
2. Replace with `"zohobooks"` (no underscore)
3. Test your Zoho Books integrations
4. Verify deprecation warnings are gone

## Support

If you see this warning in your logs:

```
⚠️ DEPRECATION: 'zoho_books' is deprecated. Please use 'zohobooks' instead.
Support for 'zoho_books' will be removed in a future version.
```

Simply update your code to use `"zohobooks"` instead of `"zoho_books"`.

---

**Last Updated:** February 10, 2026
