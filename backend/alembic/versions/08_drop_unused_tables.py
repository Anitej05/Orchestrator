"""Drop unused tables: endpoint_parameters, agent_endpoints, agent_credentials,
   conversation_analytics, conversation_search, connection_logs.

Revision ID: 08drop_unused_tables
Revises: 07add_integrations_sessions
Create Date: 2025-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

revision = '08drop_unused_tables'
down_revision = '07add_integrations_sessions'
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    return table_name in inspector.get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    if table_name not in inspector.get_table_names():
        return False
    return any(idx['name'] == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    # 1. endpoint_parameters — must go before agent_endpoints (FK dependency)
    if _table_exists('endpoint_parameters'):
        if _index_exists('endpoint_parameters', 'ix_endpoint_parameters_id'):
            op.drop_index('ix_endpoint_parameters_id', table_name='endpoint_parameters')
        op.drop_table('endpoint_parameters')

    # 2. agent_endpoints
    if _table_exists('agent_endpoints'):
        if _index_exists('agent_endpoints', 'ix_agent_endpoints_id'):
            op.drop_index('ix_agent_endpoints_id', table_name='agent_endpoints')
        op.drop_table('agent_endpoints')

    # 3. agent_credentials
    if _table_exists('agent_credentials'):
        if _index_exists('agent_credentials', 'ix_agent_credentials_user_id'):
            op.drop_index('ix_agent_credentials_user_id', table_name='agent_credentials')
        op.drop_table('agent_credentials')

    # 4. conversation_analytics
    if _table_exists('conversation_analytics'):
        if _index_exists('conversation_analytics', 'ix_conversation_analytics_thread_id'):
            op.drop_index('ix_conversation_analytics_thread_id', table_name='conversation_analytics')
        if _index_exists('conversation_analytics', 'ix_conversation_analytics_user_id'):
            op.drop_index('ix_conversation_analytics_user_id', table_name='conversation_analytics')
        op.drop_table('conversation_analytics')

    # 5. conversation_search
    if _table_exists('conversation_search'):
        if _index_exists('conversation_search', 'ix_conversation_search_thread_id'):
            op.drop_index('ix_conversation_search_thread_id', table_name='conversation_search')
        if _index_exists('conversation_search', 'ix_conversation_search_user_id'):
            op.drop_index('ix_conversation_search_user_id', table_name='conversation_search')
        op.drop_table('conversation_search')

    # 6. connection_logs
    if _table_exists('connection_logs'):
        if _index_exists('connection_logs', 'ix_connection_logs_user_id'):
            op.drop_index('ix_connection_logs_user_id', table_name='connection_logs')
        if _index_exists('connection_logs', 'ix_connection_logs_created_at'):
            op.drop_index('ix_connection_logs_created_at', table_name='connection_logs')
        op.drop_table('connection_logs')


def downgrade():
    # Recreate connection_logs
    op.create_table(
        'connection_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('app_slug', sa.String(length=100), nullable=False),
        sa.Column('connection_id', sa.String(length=255), nullable=True),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_connection_logs_user_id', 'connection_logs', ['user_id'])
    op.create_index('ix_connection_logs_created_at', 'connection_logs', ['created_at'])

    # Recreate conversation_search
    op.create_table(
        'conversation_search',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('thread_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('message_index', sa.Integer(), nullable=False),
        sa.Column('message_content', sa.Text(), nullable=False),
        sa.Column('message_role', sa.String(length=50), nullable=True),
        sa.Column('message_timestamp', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_conversation_search_thread_id', 'conversation_search', ['thread_id'])
    op.create_index('ix_conversation_search_user_id', 'conversation_search', ['user_id'])

    # Recreate conversation_analytics
    op.create_table(
        'conversation_analytics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('thread_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('total_messages', sa.Integer(), nullable=True),
        sa.Column('total_agents_used', sa.Integer(), nullable=True),
        sa.Column('plan_attempts', sa.Integer(), nullable=True),
        sa.Column('successful_plans', sa.Integer(), nullable=True),
        sa.Column('total_execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('failed_executions', sa.Integer(), nullable=True),
        sa.Column('avg_response_time_ms', sa.Float(), nullable=True),
        sa.Column('conversation_duration_seconds', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('thread_id')
    )
    op.create_index('ix_conversation_analytics_thread_id', 'conversation_analytics', ['thread_id'])
    op.create_index('ix_conversation_analytics_user_id', 'conversation_analytics', ['user_id'])

    # Recreate agent_credentials
    op.create_table(
        'agent_credentials',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('agent_id', sa.String(), nullable=False),
        sa.Column('encrypted_credentials', sa.JSON(), nullable=True),
        sa.Column('auth_type', sa.String(), nullable=True),
        sa.Column('encrypted_access_token', sa.Text(), nullable=True),
        sa.Column('encrypted_refresh_token', sa.Text(), nullable=True),
        sa.Column('auth_header_name', sa.String(), nullable=True),
        sa.Column('token_expires_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_agent_credentials_user_id', 'agent_credentials', ['user_id'])

    # Recreate agent_endpoints
    op.create_table(
        'agent_endpoints',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_id', sa.String(), nullable=False),
        sa.Column('endpoint', sa.String(), nullable=False),
        sa.Column('http_method', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('request_format', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_agent_endpoints_id', 'agent_endpoints', ['id'])

    # Recreate endpoint_parameters
    op.create_table(
        'endpoint_parameters',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('endpoint_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('param_type', sa.String(), nullable=False),
        sa.Column('required', sa.Boolean(), nullable=True),
        sa.Column('default_value', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['endpoint_id'], ['agent_endpoints.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_endpoint_parameters_id', 'endpoint_parameters', ['id'])
