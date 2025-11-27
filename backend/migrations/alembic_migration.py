"""Add plan_graph and analytics tables

Revision ID: 001_add_plan_graph_analytics
Revises: None
Create Date: 2025-11-25 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_add_plan_graph_analytics'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Apply migration"""
    
    # 1. Add plan_graph column to workflows table
    op.add_column('workflows', sa.Column('plan_graph', postgresql.JSON(), nullable=True))
    
    # 2. Create conversation_plans table
    op.create_table(
        'conversation_plans',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('plan_id', sa.String(length=255), nullable=False),
        sa.Column('thread_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('plan_version', sa.Integer(), nullable=False),
        sa.Column('task_agent_pairs', postgresql.JSON(), nullable=False),
        sa.Column('task_plan', postgresql.JSON(), nullable=False),
        sa.Column('plan_graph', postgresql.JSON(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('result', postgresql.JSON(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('plan_id')
    )
    op.create_index('idx_conversation_plans_user', 'conversation_plans', ['user_id'])
    op.create_index('idx_conversation_plans_thread', 'conversation_plans', ['thread_id'])
    op.create_index('idx_conversation_plans_status', 'conversation_plans', ['status'])
    op.create_index('idx_conversation_plans_created', 'conversation_plans', ['user_id', 'created_at'])
    
    # 3. Create conversation_search table
    op.create_table(
        'conversation_search',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('thread_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('message_index', sa.Integer(), nullable=False),
        sa.Column('message_content', sa.Text(), nullable=False),
        sa.Column('message_role', sa.String(length=50), nullable=True),
        sa.Column('message_timestamp', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_conversation_search_user', 'conversation_search', ['user_id'])
    op.create_index('idx_conversation_search_thread', 'conversation_search', ['thread_id'])
    op.create_index('idx_conversation_search_role', 'conversation_search', ['message_role'])
    
    # 4. Create conversation_tags table
    op.create_table(
        'conversation_tags',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tag_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('tag_name', sa.String(length=100), nullable=False),
        sa.Column('tag_color', sa.String(length=7), nullable=False),
        sa.Column('tag_description', sa.Text(), nullable=True),
        sa.Column('is_system', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tag_id')
    )
    op.create_index('idx_conversation_tags_user', 'conversation_tags', ['user_id'])
    op.create_index('unique_user_tag', 'conversation_tags', ['user_id', 'tag_name'], unique=True)
    
    # 5. Create conversation_tag_assignments table
    op.create_table(
        'conversation_tag_assignments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('thread_id', sa.String(length=255), nullable=False),
        sa.Column('tag_id', sa.String(length=255), nullable=False),
        sa.Column('assigned_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tag_id'], ['conversation_tags.tag_id'], ),
        sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('thread_id', 'tag_id')
    )
    op.create_index('idx_tag_assignments_tag', 'conversation_tag_assignments', ['tag_id'])
    
    # 6. Create conversation_analytics table
    op.create_table(
        'conversation_analytics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('thread_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('total_messages', sa.Integer(), nullable=False),
        sa.Column('total_agents_used', sa.Integer(), nullable=False),
        sa.Column('plan_attempts', sa.Integer(), nullable=False),
        sa.Column('successful_plans', sa.Integer(), nullable=False),
        sa.Column('total_execution_time_ms', sa.Integer(), nullable=False),
        sa.Column('failed_executions', sa.Integer(), nullable=False),
        sa.Column('avg_response_time_ms', sa.Float(), nullable=False),
        sa.Column('conversation_duration_seconds', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['thread_id'], ['user_threads.thread_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('thread_id')
    )
    op.create_index('idx_conversation_analytics_user', 'conversation_analytics', ['user_id'])
    op.create_index('idx_conversation_analytics_updated', 'conversation_analytics', ['user_id', 'updated_at'])
    
    # 7. Create agent_usage_analytics table
    op.create_table(
        'agent_usage_analytics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('analytics_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('agent_id', sa.String(length=255), nullable=False),
        sa.Column('execution_count', sa.Integer(), nullable=False),
        sa.Column('success_count', sa.Integer(), nullable=False),
        sa.Column('failure_count', sa.Integer(), nullable=False),
        sa.Column('avg_execution_time_ms', sa.Float(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('analytics_id')
    )
    op.create_index('idx_agent_usage_user', 'agent_usage_analytics', ['user_id'])
    op.create_index('idx_agent_usage_popularity', 'agent_usage_analytics', ['user_id', 'execution_count'])
    op.create_index('unique_user_agent', 'agent_usage_analytics', ['user_id', 'agent_id'], unique=True)
    
    # 8. Create user_activity_summary table
    op.create_table(
        'user_activity_summary',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('activity_date', sa.String(length=10), nullable=False),
        sa.Column('total_conversations_started', sa.Integer(), nullable=False),
        sa.Column('total_workflows_executed', sa.Integer(), nullable=False),
        sa.Column('total_plans_created', sa.Integer(), nullable=False),
        sa.Column('successful_executions', sa.Integer(), nullable=False),
        sa.Column('failed_executions', sa.Integer(), nullable=False),
        sa.Column('total_execution_time_ms', sa.Integer(), nullable=False),
        sa.Column('agents_used', sa.Integer(), nullable=False),
        sa.Column('api_calls_made', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'activity_date')
    )
    op.create_index('idx_user_activity_user', 'user_activity_summary', ['user_id'])
    op.create_index('idx_user_activity_date', 'user_activity_summary', ['activity_date'])
    
    # 9. Create workflow_execution_analytics table
    op.create_table(
        'workflow_execution_analytics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('execution_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('workflow_id', sa.String(length=255), nullable=False),
        sa.Column('total_steps', sa.Integer(), nullable=False),
        sa.Column('completed_steps', sa.Integer(), nullable=False),
        sa.Column('failed_steps', sa.Integer(), nullable=False),
        sa.Column('total_duration_ms', sa.Integer(), nullable=False),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('error_type', sa.String(length=100), nullable=True),
        sa.Column('success_rate', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['workflow_id'], ['workflows.workflow_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('execution_id')
    )
    op.create_index('idx_workflow_execution_user', 'workflow_execution_analytics', ['user_id'])
    op.create_index('idx_workflow_execution_workflow', 'workflow_execution_analytics', ['workflow_id'])
    op.create_index('idx_workflow_execution_created', 'workflow_execution_analytics', ['user_id', 'created_at'])
    
    # 10. Enhance user_threads table
    op.add_column('user_threads', sa.Column('preview_text', sa.Text(), nullable=True))
    op.add_column('user_threads', sa.Column('last_message_at', sa.DateTime(), nullable=True))
    op.add_column('user_threads', sa.Column('agent_ids', postgresql.JSON(), nullable=True))
    op.add_column('user_threads', sa.Column('execution_status', sa.String(length=50), nullable=True))
    op.add_column('user_threads', sa.Column('message_count', sa.Integer(), nullable=False))
    op.add_column('user_threads', sa.Column('has_plan', sa.Boolean(), nullable=False))
    
    # Add indexes for user_threads enhancements
    op.create_index('idx_user_threads_last_message', 'user_threads', ['user_id', 'last_message_at'])
    op.create_index('idx_user_threads_execution_status', 'user_threads', ['user_id', 'execution_status'])
    op.create_index('idx_user_threads_has_plan', 'user_threads', ['user_id', 'has_plan'])


def downgrade():
    """Revert migration"""
    
    # Drop indexes
    op.drop_index('idx_user_threads_has_plan', 'user_threads')
    op.drop_index('idx_user_threads_execution_status', 'user_threads')
    op.drop_index('idx_user_threads_last_message', 'user_threads')
    
    # Drop new columns from user_threads
    op.drop_column('user_threads', 'has_plan')
    op.drop_column('user_threads', 'message_count')
    op.drop_column('user_threads', 'execution_status')
    op.drop_column('user_threads', 'agent_ids')
    op.drop_column('user_threads', 'last_message_at')
    op.drop_column('user_threads', 'preview_text')
    
    # Drop workflows enhancement
    op.drop_column('workflows', 'plan_graph')
    
    # Drop new tables
    op.drop_table('workflow_execution_analytics')
    op.drop_table('user_activity_summary')
    op.drop_table('agent_usage_analytics')
    op.drop_table('conversation_analytics')
    op.drop_table('conversation_tag_assignments')
    op.drop_table('conversation_tags')
    op.drop_table('conversation_search')
    op.drop_table('conversation_plans')
