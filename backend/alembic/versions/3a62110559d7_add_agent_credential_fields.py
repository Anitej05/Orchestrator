"""add_agent_credential_fields

Revision ID: 3a62110559d7
Revises: 888609832433
Create Date: 2025-12-04 10:13:47.684417

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3a62110559d7'
down_revision: Union[str, Sequence[str], None] = '888609832433'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add missing columns to agents table if they don't exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('agents')]
    
    if 'requires_credentials' not in columns:
        op.add_column('agents', sa.Column('requires_credentials', sa.Boolean(), nullable=True, server_default='false'))
    
    if 'credential_fields' not in columns:
        op.add_column('agents', sa.Column('credential_fields', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove columns if they exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('agents')]
    
    if 'credential_fields' in columns:
        op.drop_column('agents', 'credential_fields')
    
    if 'requires_credentials' in columns:
        op.drop_column('agents', 'requires_credentials')
