"""add_is_active_to_agent_credentials

Revision ID: f42ecc999e66
Revises: 257a4bfbd32c
Create Date: 2025-12-04 11:40:15.815163

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f42ecc999e66'
down_revision: Union[str, Sequence[str], None] = '257a4bfbd32c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Use raw SQL to check and add column if not exists
    from sqlalchemy import text
    conn = op.get_bind()
    
    # Check if column exists
    result = conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='agent_credentials' AND column_name='is_active'
    """))
    
    if result.fetchone() is None:
        # Column doesn't exist, add it
        conn.execute(text("""
            ALTER TABLE agent_credentials 
            ADD COLUMN is_active BOOLEAN DEFAULT true NOT NULL
        """))


def downgrade() -> None:
    """Downgrade schema."""
    # Use raw SQL to check and drop column if exists
    from sqlalchemy import text
    conn = op.get_bind()
    
    # Check if column exists
    result = conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='agent_credentials' AND column_name='is_active'
    """))
    
    if result.fetchone() is not None:
        # Column exists, drop it
        conn.execute(text("""
            ALTER TABLE agent_credentials 
            DROP COLUMN is_active
        """))
