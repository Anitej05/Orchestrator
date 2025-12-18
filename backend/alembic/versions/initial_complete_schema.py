"""Initial complete schema

Revision ID: initial_complete_schema
Revises: 
Create Date: 2025-12-15

This is a complete master migration that creates all tables.
Future schema changes will be added as new migrations on top of this.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'initial_complete_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables from models.py"""
    
    # Ensure pgvector extension exists
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Agents table
    op.create_table('agents',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('owner_id', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('capabilities', postgresql.JSON(astext_type=sa.Text()), nullable=False),
    sa.Column('price_per_call_usd', sa.Float(), nullable=False),
    sa.Column('status', sa.Enum('active', 'inactive', 'deprecated', name='statusenum'), nullable=False),
    sa.Column('rating', sa.Float(), nullable=True),
    sa.Column('rating_count', sa.Integer(), nullable=False),
    sa.Column('public_key_pem', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('agent_type', sa.String(), nullable=True),
    sa.Column('connection_config', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('requires_credentials', sa.Boolean(), nullable=True),
    sa.Column('credential_fields', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agents_id'), 'agents', ['id'], unique=False)
    op.create_index(op.f('ix_agents_owner_id'), 'agents', ['owner_id'], unique=False)
    op.create_index(op.f('ix_agents_status'), 'agents', ['status'], unique=False)
    
    # Agent capabilities table
    op.create_table('agent_capabilities',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('agent_id', sa.String(), nullable=False),
    sa.Column('capability_text', sa.String(), nullable=False),
    sa.Column('embedding', Vector(768), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_capabilities_id'), 'agent_capabilities', ['id'], unique=False)
    
    # Agent endpoints table
    op.create_table('agent_endpoints',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('agent_id', sa.String(), nullable=False),
    sa.Column('endpoint', sa.String(), nullable=False),
    sa.Column('http_method', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('request_format', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_endpoints_id'), 'agent_endpoints', ['id'], unique=False)
    
    # Endpoint parameters table
    op.create_table('endpoint_parameters',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('endpoint_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('param_type', sa.String(), nullable=False),
    sa.Column('required', sa.Boolean(), nullable=True),
    sa.Column('default_value', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['endpoint_id'], ['agent_endpoints.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_endpoint_parameters_id'), 'endpoint_parameters', ['id'], unique=False)
    
    # User threads table
    op.create_table('user_threads',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('thread_id', sa.String(), nullable=False),
    sa.Column('title', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_threads_id'), 'user_threads', ['id'], unique=False)
    op.create_index(op.f('ix_user_threads_thread_id'), 'user_threads', ['thread_id'], unique=True)
    op.create_index(op.f('ix_user_threads_user_id'), 'user_threads', ['user_id'], unique=False)
    
    # Workflows table
    op.create_table('workflows',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('workflow_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('blueprint', postgresql.JSON(astext_type=sa.Text()), nullable=False),
    sa.Column('plan_graph', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('version', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflows_id'), 'workflows', ['id'], unique=False)
    op.create_index(op.f('ix_workflows_user_id'), 'workflows', ['user_id'], unique=False)
    op.create_index(op.f('ix_workflows_workflow_id'), 'workflows', ['workflow_id'], unique=True)
    
    # Workflow executions table
    op.create_table('workflow_executions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('execution_id', sa.String(), nullable=False),
    sa.Column('workflow_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('inputs', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('outputs', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['workflow_id'], ['workflows.workflow_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflow_executions_execution_id'), 'workflow_executions', ['execution_id'], unique=True)
    op.create_index(op.f('ix_workflow_executions_id'), 'workflow_executions', ['id'], unique=False)
    op.create_index(op.f('ix_workflow_executions_user_id'), 'workflow_executions', ['user_id'], unique=False)
    
    # Workflow schedules table
    op.create_table('workflow_schedules',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('schedule_id', sa.String(), nullable=False),
    sa.Column('workflow_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('cron_expression', sa.String(), nullable=False),
    sa.Column('input_template', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('conversation_thread_id', sa.String(), nullable=True),
    sa.Column('last_run_at', sa.DateTime(), nullable=True),
    sa.Column('next_run_at', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['workflow_id'], ['workflows.workflow_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflow_schedules_id'), 'workflow_schedules', ['id'], unique=False)
    op.create_index(op.f('ix_workflow_schedules_schedule_id'), 'workflow_schedules', ['schedule_id'], unique=True)
    op.create_index(op.f('ix_workflow_schedules_user_id'), 'workflow_schedules', ['user_id'], unique=False)
    
    # Workflow webhooks table
    op.create_table('workflow_webhooks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('webhook_id', sa.String(), nullable=False),
    sa.Column('workflow_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('webhook_token', sa.String(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['workflow_id'], ['workflows.workflow_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflow_webhooks_id'), 'workflow_webhooks', ['id'], unique=False)
    op.create_index(op.f('ix_workflow_webhooks_user_id'), 'workflow_webhooks', ['user_id'], unique=False)
    op.create_index(op.f('ix_workflow_webhooks_webhook_id'), 'workflow_webhooks', ['webhook_id'], unique=True)
    
    # Agent credentials table
    op.create_table('agent_credentials',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('agent_id', sa.String(), nullable=False),
    sa.Column('encrypted_credentials', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('auth_type', sa.String(), nullable=True),
    sa.Column('encrypted_access_token', sa.Text(), nullable=True),
    sa.Column('encrypted_refresh_token', sa.Text(), nullable=True),
    sa.Column('auth_header_name', sa.String(), nullable=True),
    sa.Column('token_expires_at', sa.DateTime(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_credentials_user_id'), 'agent_credentials', ['user_id'], unique=False)
    
    # Conversation plans table
    op.create_table('conversation_plans',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('plan_id', sa.String(length=255), nullable=False),
    sa.Column('thread_id', sa.String(length=255), nullable=False),
    sa.Column('user_id', sa.String(length=255), nullable=False),
    sa.Column('plan_version', sa.Integer(), nullable=True),
    sa.Column('task_agent_pairs', postgresql.JSON(astext_type=sa.Text()), nullable=False),
    sa.Column('task_plan', postgresql.JSON(astext_type=sa.Text()), nullable=False),
    sa.Column('plan_graph', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=True),
    sa.Column('result', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('execution_time_ms', sa.Integer(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversation_plans_plan_id'), 'conversation_plans', ['plan_id'], unique=True)
    op.create_index(op.f('ix_conversation_plans_thread_id'), 'conversation_plans', ['thread_id'], unique=False)
    op.create_index(op.f('ix_conversation_plans_user_id'), 'conversation_plans', ['user_id'], unique=False)
    
    # Conversation search table
    op.create_table('conversation_search',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('thread_id', sa.String(length=255), nullable=False),
    sa.Column('user_id', sa.String(length=255), nullable=False),
    sa.Column('message_index', sa.Integer(), nullable=False),
    sa.Column('message_content', sa.Text(), nullable=False),
    sa.Column('message_role', sa.String(length=50), nullable=True),
    sa.Column('message_timestamp', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversation_search_thread_id'), 'conversation_search', ['thread_id'], unique=False)
    op.create_index(op.f('ix_conversation_search_user_id'), 'conversation_search', ['user_id'], unique=False)
    
    # Conversation tags table
    op.create_table('conversation_tags',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('tag_id', sa.String(length=255), nullable=False),
    sa.Column('user_id', sa.String(length=255), nullable=False),
    sa.Column('tag_name', sa.String(length=100), nullable=False),
    sa.Column('tag_color', sa.String(length=7), nullable=True),
    sa.Column('tag_description', sa.Text(), nullable=True),
    sa.Column('is_system', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversation_tags_tag_id'), 'conversation_tags', ['tag_id'], unique=True)
    op.create_index(op.f('ix_conversation_tags_user_id'), 'conversation_tags', ['user_id'], unique=False)
    
    # Conversation tag assignments table
    op.create_table('conversation_tag_assignments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('thread_id', sa.String(length=255), nullable=False),
    sa.Column('tag_id', sa.String(length=255), nullable=False),
    sa.Column('assigned_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['tag_id'], ['conversation_tags.tag_id'], ),
    sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversation_tag_assignments_tag_id'), 'conversation_tag_assignments', ['tag_id'], unique=False)
    op.create_index(op.f('ix_conversation_tag_assignments_thread_id'), 'conversation_tag_assignments', ['thread_id'], unique=False)
    
    # Conversation analytics table
    op.create_table('conversation_analytics',
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
    sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversation_analytics_thread_id'), 'conversation_analytics', ['thread_id'], unique=True)
    op.create_index(op.f('ix_conversation_analytics_user_id'), 'conversation_analytics', ['user_id'], unique=False)
    
    # Agent usage analytics table
    op.create_table('agent_usage_analytics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('analytics_id', sa.String(length=255), nullable=False),
    sa.Column('user_id', sa.String(length=255), nullable=False),
    sa.Column('agent_id', sa.String(length=255), nullable=False),
    sa.Column('execution_count', sa.Integer(), nullable=True),
    sa.Column('success_count', sa.Integer(), nullable=True),
    sa.Column('failure_count', sa.Integer(), nullable=True),
    sa.Column('avg_execution_time_ms', sa.Float(), nullable=True),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_usage_analytics_analytics_id'), 'agent_usage_analytics', ['analytics_id'], unique=True)
    op.create_index(op.f('ix_agent_usage_analytics_user_id'), 'agent_usage_analytics', ['user_id'], unique=False)
    
    # User activity summary table
    op.create_table('user_activity_summary',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.String(length=255), nullable=False),
    sa.Column('activity_date', sa.String(length=10), nullable=False),
    sa.Column('total_conversations_started', sa.Integer(), nullable=True),
    sa.Column('total_workflows_executed', sa.Integer(), nullable=True),
    sa.Column('total_plans_created', sa.Integer(), nullable=True),
    sa.Column('successful_executions', sa.Integer(), nullable=True),
    sa.Column('failed_executions', sa.Integer(), nullable=True),
    sa.Column('total_execution_time_ms', sa.Integer(), nullable=True),
    sa.Column('agents_used', sa.Integer(), nullable=True),
    sa.Column('api_calls_made', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_activity_summary_user_id'), 'user_activity_summary', ['user_id'], unique=False)
    
    # Workflow execution analytics table
    op.create_table('workflow_execution_analytics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('execution_id', sa.String(length=255), nullable=False),
    sa.Column('user_id', sa.String(length=255), nullable=False),
    sa.Column('workflow_id', sa.String(length=255), nullable=False),
    sa.Column('total_steps', sa.Integer(), nullable=True),
    sa.Column('completed_steps', sa.Integer(), nullable=True),
    sa.Column('failed_steps', sa.Integer(), nullable=True),
    sa.Column('total_duration_ms', sa.Integer(), nullable=True),
    sa.Column('retry_count', sa.Integer(), nullable=True),
    sa.Column('error_type', sa.String(length=100), nullable=True),
    sa.Column('success_rate', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['execution_id'], ['workflow_executions.execution_id'], ),
    sa.ForeignKeyConstraint(['workflow_id'], ['workflows.workflow_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflow_execution_analytics_execution_id'), 'workflow_execution_analytics', ['execution_id'], unique=True)
    op.create_index(op.f('ix_workflow_execution_analytics_user_id'), 'workflow_execution_analytics', ['user_id'], unique=False)
    op.create_index(op.f('ix_workflow_execution_analytics_workflow_id'), 'workflow_execution_analytics', ['workflow_id'], unique=False)


def downgrade() -> None:
    """Drop all tables"""
    op.drop_index(op.f('ix_workflow_execution_analytics_workflow_id'), table_name='workflow_execution_analytics')
    op.drop_index(op.f('ix_workflow_execution_analytics_user_id'), table_name='workflow_execution_analytics')
    op.drop_index(op.f('ix_workflow_execution_analytics_execution_id'), table_name='workflow_execution_analytics')
    op.drop_table('workflow_execution_analytics')
    op.drop_index(op.f('ix_user_activity_summary_user_id'), table_name='user_activity_summary')
    op.drop_table('user_activity_summary')
    op.drop_index(op.f('ix_agent_usage_analytics_user_id'), table_name='agent_usage_analytics')
    op.drop_index(op.f('ix_agent_usage_analytics_analytics_id'), table_name='agent_usage_analytics')
    op.drop_table('agent_usage_analytics')
    op.drop_index(op.f('ix_conversation_analytics_user_id'), table_name='conversation_analytics')
    op.drop_index(op.f('ix_conversation_analytics_thread_id'), table_name='conversation_analytics')
    op.drop_table('conversation_analytics')
    op.drop_index(op.f('ix_conversation_tag_assignments_thread_id'), table_name='conversation_tag_assignments')
    op.drop_index(op.f('ix_conversation_tag_assignments_tag_id'), table_name='conversation_tag_assignments')
    op.drop_table('conversation_tag_assignments')
    op.drop_index(op.f('ix_conversation_tags_user_id'), table_name='conversation_tags')
    op.drop_index(op.f('ix_conversation_tags_tag_id'), table_name='conversation_tags')
    op.drop_table('conversation_tags')
    op.drop_index(op.f('ix_conversation_search_user_id'), table_name='conversation_search')
    op.drop_index(op.f('ix_conversation_search_thread_id'), table_name='conversation_search')
    op.drop_table('conversation_search')
    op.drop_index(op.f('ix_conversation_plans_user_id'), table_name='conversation_plans')
    op.drop_index(op.f('ix_conversation_plans_thread_id'), table_name='conversation_plans')
    op.drop_index(op.f('ix_conversation_plans_plan_id'), table_name='conversation_plans')
    op.drop_table('conversation_plans')
    op.drop_index(op.f('ix_agent_credentials_user_id'), table_name='agent_credentials')
    op.drop_table('agent_credentials')
    op.drop_index(op.f('ix_workflow_webhooks_webhook_id'), table_name='workflow_webhooks')
    op.drop_index(op.f('ix_workflow_webhooks_user_id'), table_name='workflow_webhooks')
    op.drop_index(op.f('ix_workflow_webhooks_id'), table_name='workflow_webhooks')
    op.drop_table('workflow_webhooks')
    op.drop_index(op.f('ix_workflow_schedules_user_id'), table_name='workflow_schedules')
    op.drop_index(op.f('ix_workflow_schedules_schedule_id'), table_name='workflow_schedules')
    op.drop_index(op.f('ix_workflow_schedules_id'), table_name='workflow_schedules')
    op.drop_table('workflow_schedules')
    op.drop_index(op.f('ix_workflow_executions_user_id'), table_name='workflow_executions')
    op.drop_index(op.f('ix_workflow_executions_id'), table_name='workflow_executions')
    op.drop_index(op.f('ix_workflow_executions_execution_id'), table_name='workflow_executions')
    op.drop_table('workflow_executions')
    op.drop_index(op.f('ix_workflows_workflow_id'), table_name='workflows')
    op.drop_index(op.f('ix_workflows_user_id'), table_name='workflows')
    op.drop_index(op.f('ix_workflows_id'), table_name='workflows')
    op.drop_table('workflows')
    op.drop_index(op.f('ix_user_threads_user_id'), table_name='user_threads')
    op.drop_index(op.f('ix_user_threads_thread_id'), table_name='user_threads')
    op.drop_index(op.f('ix_user_threads_id'), table_name='user_threads')
    op.drop_table('user_threads')
    op.drop_index(op.f('ix_endpoint_parameters_id'), table_name='endpoint_parameters')
    op.drop_table('endpoint_parameters')
    op.drop_index(op.f('ix_agent_endpoints_id'), table_name='agent_endpoints')
    op.drop_table('agent_endpoints')
    op.drop_index(op.f('ix_agent_capabilities_id'), table_name='agent_capabilities')
    op.drop_table('agent_capabilities')
    op.drop_index(op.f('ix_agents_status'), table_name='agents')
    op.drop_index(op.f('ix_agents_owner_id'), table_name='agents')
    op.drop_index(op.f('ix_agents_id'), table_name='agents')
    op.drop_table('agents')
    op.execute('DROP TYPE IF EXISTS statusenum')
