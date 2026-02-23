"""Add user connections tables

Revision ID: 42363728b514
Revises: e7a71c4bf948
Create Date: 2026-02-23 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '42363728b514'
down_revision: Union[str, Sequence[str], None] = 'e7a71c4bf948'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create user_connections and connection_logs tables."""
    from sqlalchemy import inspect as sa_inspect
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    existing_tables = inspector.get_table_names()

    def table_exists(name):
        return name in existing_tables

    # Create user_connections table
    if not table_exists('user_connections'):
        op.create_table('user_connections',
            sa.Column('id', sa.String(), nullable=False),
            sa.Column('user_id', sa.String(length=255), nullable=False),
            sa.Column('app_slug', sa.String(length=100), nullable=False),
            sa.Column('connection_id', sa.Text(), nullable=True),
            sa.Column('status', sa.String(length=50), nullable=True),
            sa.Column('app_metadata', sa.JSON(), nullable=True),
            sa.Column('auth_timestamp', sa.DateTime(), nullable=True),
            sa.Column('last_verified', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_user_connections_user_id'), 'user_connections', ['user_id'], unique=False)
        op.create_index(op.f('ix_user_connections_app_slug'), 'user_connections', ['app_slug'], unique=False)
    
    # Create connection_logs table
    if not table_exists('connection_logs'):
        op.create_table('connection_logs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.String(length=255), nullable=False),
            sa.Column('app_slug', sa.String(length=100), nullable=False),
            sa.Column('connection_id', sa.String(length=255), nullable=True),
            sa.Column('event_type', sa.String(length=50), nullable=False),
            sa.Column('status', sa.String(length=50), nullable=False),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_connection_logs_user_id'), 'connection_logs', ['user_id'], unique=False)
        op.create_index(op.f('ix_connection_logs_created_at'), 'connection_logs', ['created_at'], unique=False)


def downgrade() -> None:
    """Drop user_connections and connection_logs tables."""
    op.drop_index(op.f('ix_connection_logs_created_at'), table_name='connection_logs')
    op.drop_index(op.f('ix_connection_logs_user_id'), table_name='connection_logs')
    op.drop_table('connection_logs')
    
    op.drop_index(op.f('ix_user_connections_app_slug'), table_name='user_connections')
    op.drop_index(op.f('ix_user_connections_user_id'), table_name='user_connections')
    op.drop_table('user_connections')
