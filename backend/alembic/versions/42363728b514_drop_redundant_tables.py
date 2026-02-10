"""Drop redundant tables

Revision ID: 42363728b514
Revises: e7a71c4bf948
Create Date: 2026-02-10 23:19:43.376461

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '42363728b514'
down_revision: Union[str, Sequence[str], None] = 'e7a71c4bf948'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop redundant tables that are not being used."""
    # Drop conversation-related tables
    op.drop_index(op.f("ix_conversation_tag_assignments_thread_id"), table_name="conversation_tag_assignments")
    op.drop_index(op.f("ix_conversation_tag_assignments_tag_id"), table_name="conversation_tag_assignments")
    op.drop_table("conversation_tag_assignments")

    op.drop_index(op.f("ix_conversation_tags_user_id"), table_name="conversation_tags")
    op.drop_index(op.f("ix_conversation_tags_tag_id"), table_name="conversation_tags")
    op.drop_table("conversation_tags")

    op.drop_index(op.f("ix_conversation_search_user_id"), table_name="conversation_search")
    op.drop_index(op.f("ix_conversation_search_thread_id"), table_name="conversation_search")
    op.drop_table("conversation_search")

    op.drop_index(op.f("ix_conversation_plans_user_id"), table_name="conversation_plans")
    op.drop_index(op.f("ix_conversation_plans_thread_id"), table_name="conversation_plans")
    op.drop_index(op.f("ix_conversation_plans_plan_id"), table_name="conversation_plans")
    op.drop_table("conversation_plans")

    # Drop agent-related redundant tables
    op.drop_index(op.f("ix_agent_credentials_user_id"), table_name="agent_credentials")
    op.drop_table("agent_credentials")

    op.drop_index(op.f("ix_agent_capabilities_id"), table_name="agent_capabilities")
    op.drop_table("agent_capabilities")

    # Drop workflow webhooks table
    op.drop_index(op.f("ix_workflow_webhooks_webhook_id"), table_name="workflow_webhooks")
    op.drop_index(op.f("ix_workflow_webhooks_user_id"), table_name="workflow_webhooks")
    op.drop_index(op.f("ix_workflow_webhooks_id"), table_name="workflow_webhooks")
    op.drop_table("workflow_webhooks")


def downgrade() -> None:
    """Recreate redundant tables if needed."""
    # Recreate workflow_webhooks
    op.create_table(
        "workflow_webhooks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("webhook_id", sa.String(), nullable=False),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("webhook_token", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.workflow_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_workflow_webhooks_id"), "workflow_webhooks", ["id"], unique=False)
    op.create_index(op.f("ix_workflow_webhooks_user_id"), "workflow_webhooks", ["user_id"], unique=False)
    op.create_index(op.f("ix_workflow_webhooks_webhook_id"), "workflow_webhooks", ["webhook_id"], unique=True)

    # Recreate agent_capabilities
    op.create_table(
        "agent_capabilities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("capability_text", sa.String(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_agent_capabilities_id"), "agent_capabilities", ["id"], unique=False)

    # Recreate agent_credentials
    op.create_table(
        "agent_credentials",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("encrypted_credentials", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("auth_type", sa.String(), nullable=True),
        sa.Column("encrypted_access_token", sa.Text(), nullable=True),
        sa.Column("encrypted_refresh_token", sa.Text(), nullable=True),
        sa.Column("auth_header_name", sa.String(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_agent_credentials_user_id"), "agent_credentials", ["user_id"], unique=False)

    # Recreate conversation_plans
    op.create_table(
        "conversation_plans",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plan_id", sa.String(length=255), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("plan_version", sa.Integer(), nullable=True),
        sa.Column("task_agent_pairs", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("task_plan", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("plan_graph", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("result", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("execution_time_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["thread_id"], ["user_threads.thread_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversation_plans_plan_id"), "conversation_plans", ["plan_id"], unique=True)
    op.create_index(op.f("ix_conversation_plans_thread_id"), "conversation_plans", ["thread_id"], unique=False)
    op.create_index(op.f("ix_conversation_plans_user_id"), "conversation_plans", ["user_id"], unique=False)

    # Recreate conversation_search
    op.create_table(
        "conversation_search",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("message_index", sa.Integer(), nullable=False),
        sa.Column("message_content", sa.Text(), nullable=False),
        sa.Column("message_role", sa.String(length=50), nullable=True),
        sa.Column("message_timestamp", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["thread_id"], ["user_threads.thread_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversation_search_thread_id"), "conversation_search", ["thread_id"], unique=False)
    op.create_index(op.f("ix_conversation_search_user_id"), "conversation_search", ["user_id"], unique=False)

    # Recreate conversation_tags
    op.create_table(
        "conversation_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tag_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("tag_name", sa.String(length=100), nullable=False),
        sa.Column("tag_color", sa.String(length=7), nullable=True),
        sa.Column("tag_description", sa.Text(), nullable=True),
        sa.Column("is_system", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversation_tags_tag_id"), "conversation_tags", ["tag_id"], unique=True)
    op.create_index(op.f("ix_conversation_tags_user_id"), "conversation_tags", ["user_id"], unique=False)

    # Recreate conversation_tag_assignments
    op.create_table(
        "conversation_tag_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("tag_id", sa.String(length=255), nullable=False),
        sa.Column("assigned_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["tag_id"], ["conversation_tags.tag_id"]),
        sa.ForeignKeyConstraint(["thread_id"], ["user_threads.thread_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversation_tag_assignments_tag_id"), "conversation_tag_assignments", ["tag_id"], unique=False)
    op.create_index(op.f("ix_conversation_tag_assignments_thread_id"), "conversation_tag_assignments", ["thread_id"], unique=False)

