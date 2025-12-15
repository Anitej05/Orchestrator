"""Add request_format column to agent_endpoints

Revision ID: 172e9b4b297a
Revises: f42ecc999e66
Create Date: 2025-12-13 16:42:26.930714

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '172e9b4b297a'
down_revision: Union[str, Sequence[str], None] = 'f42ecc999e66'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add request_format column to agent_endpoints
    op.add_column('agent_endpoints', sa.Column('request_format', sa.String(), nullable=True))
    
    # Add created_at column to agents if it doesn't exist
    # Using batch mode to handle potential issues
    from sqlalchemy import inspect
    from sqlalchemy.engine import reflection
    
    bind = op.get_bind()
    inspector = inspect(bind)
    
    # Check if created_at column exists in agents table
    agents_columns = [col['name'] for col in inspector.get_columns('agents')]
    if 'created_at' not in agents_columns:
        op.add_column('agents', sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove the columns we added
    op.drop_column('agent_endpoints', 'request_format')
    
    # Optionally remove created_at from agents (if it was added)
    from sqlalchemy import inspect
    bind = op.get_bind()
    inspector = inspect(bind)
    
    agents_columns = [col['name'] for col in inspector.get_columns('agents')]
    if 'created_at' in agents_columns:
        op.drop_column('agents', 'created_at')
