"""add_encrypted_credentials_column

Revision ID: 6219b9bbb87a
Revises: 3a62110559d7
Create Date: 2025-12-04 10:37:45.050130

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6219b9bbb87a'
down_revision: Union[str, Sequence[str], None] = '3a62110559d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add encrypted_credentials column if it doesn't exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('agent_credentials')]
    
    if 'encrypted_credentials' not in columns:
        op.add_column('agent_credentials', sa.Column('encrypted_credentials', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('agent_credentials')]
    
    if 'encrypted_credentials' in columns:
        op.drop_column('agent_credentials', 'encrypted_credentials')
