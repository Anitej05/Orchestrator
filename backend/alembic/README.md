# Alembic Migration System

## Overview

This project uses **Alembic** for database schema versioning with a single comprehensive migration file.

## Current System

- ✅ **Single migration file**: `versions/initial_complete_schema.py`
- ✅ **Automated via setup.py**: Run `python setup.py` to apply migrations
- ✅ **All tables and columns defined**: Complete schema in one migration
- ✅ **Simple workflow**: No manual `alembic` commands needed

## How It Works

### Automated Setup (Recommended)
```bash
cd backend
python setup.py  # Creates DB, enables pgvector, runs Alembic migrations
```

This runs `alembic upgrade head` automatically.

### Manual Migration Commands (Advanced)
```bash
# View current migration status
alembic current

# View migration history
alembic history

# Apply migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1
```

## Adding New Schema Changes

1. **Update models.py** with new columns/tables
2. **Generate migration**:
   ```bash
   alembic revision --autogenerate -m "add_new_feature"
   ```
3. **Review** generated file in `versions/`
4. **Apply**:
   ```bash
   python setup.py  # Runs migrations automatically
   ```

## Current Schema

All tables are defined in `versions/initial_complete_schema.py`:
- agents, agent_capabilities, agent_endpoints, endpoint_parameters
- agent_credentials, user_threads
- workflows, workflow_executions, workflow_schedules, workflow_webhooks
- conversation_plans, conversation_search, conversation_tags, conversation_tag_assignments
- conversation_analytics, agent_usage_analytics, user_activity_summary, workflow_execution_analytics

## Migration File Structure

```
alembic/
├── env.py                          # Alembic environment config
├── script.py.mako                  # Template for new migrations
└── versions/
    └── initial_complete_schema.py  # Single comprehensive migration
```

## Troubleshooting

### "No such column: request_format"
**Solution**: Migrations not applied
```bash
python setup.py
```

### "Alembic command not found"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Check Migration Status
```bash
alembic current  # Shows current revision
alembic history  # Shows all revisions
```

---

**Note**: This README was updated to reflect current system. Previous references to `database_init.py` have been removed as that file was consolidated into `setup.py`.
```bash
cd backend
python test_database_init.py
```

### Complete Database Reset (If Needed)
```python
# In Python shell or script
from database_init import reset_database
reset_database()  # WARNING: Deletes all data
```

## Master Migration

The file `versions/initial_complete_schema.py` contains the complete database schema. This serves as:
- Reference for fresh installations
- Documentation of current schema
- Baseline for future migrations

## Adding New Schema Changes

### Option 1: Automatic (Recommended)
1. Update `models.py` with your changes
2. Restart the app
3. Smart init detects and adds missing columns automatically!

### Option 2: Traditional Migration (If Preferred)
```bash
# Create new migration
alembic revision -m "add new feature"

# Edit the generated file
# Then run:
alembic upgrade head
```

## Why This Approach?

Traditional migrations work well for:
- Large teams with complex coordination
- Production databases needing granular version control
- Databases with complex data transformations

Smart initialization is better for:
- ✅ Development teams needing simplicity
- ✅ Self-healing production deployments
- ✅ Eliminating migration troubleshooting
- ✅ Working across multiple environments seamlessly

## Technical Details

The smart initialization (`database_init.py`):
1. Uses `Base.metadata.create_all()` to create missing tables
2. Uses SQLAlchemy inspector to detect existing schema
3. Compares with `models.py` to find missing columns
4. Executes `ALTER TABLE ADD COLUMN` for missing fields
5. Sets safe defaults (nullable=True, server_default where applicable)

## Alembic Commands (Still Available)

```bash
# Check current revision
alembic current

# View migration history
alembic history

# Create new migration (optional)
alembic revision -m "description"

# Upgrade to latest (now done automatically by app)
alembic upgrade head

# Downgrade (if needed)
alembic downgrade -1
```

## File Structure

```
alembic/
├── env.py                          # Alembic environment config
├── versions/
│   └── initial_complete_schema.py  # Master migration
└── README.md                       # This file

../
├── database_init.py                # Smart initialization (MAIN)
├── test_database_init.py           # Test script
├── DATABASE_MIGRATION_GUIDE.md     # Detailed documentation
└── models.py                       # Source of truth for schema
```

## Troubleshooting

**Problem**: "Column already exists" error  
**Solution**: Restart app - smart init will detect and skip existing columns

**Problem**: "Table doesn't exist" error  
**Solution**: Run `python test_database_init.py`

**Problem**: Want to reset everything  
**Solution**: Run `reset_database()` from `database_init.py`

## Migration History (Archived)

Previous migrations (now deleted):
1. initial_schema
2. add_agent_credential_fields
3. add_encrypted_credentials_column
4. add_conversation_thread_id
5. make_public_key_pem_nullable
6. add_is_active
7. add_request_format (was broken)

All functionality preserved in `initial_complete_schema.py`

## See Also

- `DATABASE_MIGRATION_GUIDE.md` - Comprehensive guide
- `database_init.py` - Smart initialization implementation
- `models.py` - Database schema definitions
