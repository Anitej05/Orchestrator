"""add_conversation_thread_id_to_workflow_schedules

Revision ID: 8ccfe035d7b3
Revises: 6219b9bbb87a
Create Date: 2025-12-04 11:31:19.087053

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8ccfe035d7b3'
down_revision: Union[str, Sequence[str], None] = '6219b9bbb87a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Check if column already exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('workflow_schedules')]
    
    if 'conversation_thread_id' not in columns:
        op.add_column('workflow_schedules', 
            sa.Column('conversation_thread_id', sa.String(), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema."""
    # Check if column exists before dropping
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('workflow_schedules')]
    
    if 'conversation_thread_id' in columns:
        op.drop_column('workflow_schedules', 'conversation_thread_id')
