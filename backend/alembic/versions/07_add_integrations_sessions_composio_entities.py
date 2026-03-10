"""Add integrations_sessions and composio_entities tables; align user_connections schema

Revision ID: 07add_integrations_sessions
Revises: 48cf81f098d2
Create Date: 2026-03-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '07add_integrations_sessions'
down_revision: Union[str, Sequence[str], None] = '48cf81f098d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create integrations_sessions and composio_entities tables.
    Also backfill new columns on user_connections.
    """
    from sqlalchemy import inspect as sa_inspect
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    existing_tables = inspector.get_table_names()

    def table_exists(name: str) -> bool:
        return name in existing_tables

    def column_exists(table: str, col: str) -> bool:
        if not table_exists(table):
            return False
        return col in [c["name"] for c in inspector.get_columns(table)]

    # ------------------------------------------------------------------
    # 1. integrations_sessions
    # ------------------------------------------------------------------
    if not table_exists("integrations_sessions"):
        op.create_table(
            "integrations_sessions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("user_id", sa.String(length=255), nullable=False),
            sa.Column("app_slug", sa.String(length=100), nullable=False),
            sa.Column("session_id", sa.String(length=255), nullable=False),
            sa.Column("last_context", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("last_used", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "app_slug", name="uq_integrations_sessions_user_app"),
        )
        op.create_index(
            "ix_integrations_sessions_user_id", "integrations_sessions", ["user_id"], unique=False
        )

    # ------------------------------------------------------------------
    # 2. composio_entities
    # ------------------------------------------------------------------
    if not table_exists("composio_entities"):
        op.create_table(
            "composio_entities",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("internal_user_id", sa.String(length=255), nullable=False),
            sa.Column("composio_entity_id", sa.String(length=255), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("internal_user_id", name="uq_composio_entities_internal_user"),
            sa.UniqueConstraint("composio_entity_id", name="uq_composio_entities_entity_id"),
        )
        op.create_index(
            "ix_composio_entities_internal_user_id",
            "composio_entities",
            ["internal_user_id"],
            unique=False,
        )
        op.create_index(
            "ix_composio_entities_composio_entity_id",
            "composio_entities",
            ["composio_entity_id"],
            unique=False,
        )

    # ------------------------------------------------------------------
    # 3. Align user_connections schema (add new canonical columns)
    # ------------------------------------------------------------------
    if table_exists("user_connections"):
        if not column_exists("user_connections", "internal_user_id"):
            op.add_column(
                "user_connections", sa.Column("internal_user_id", sa.String(length=255), nullable=True)
            )
            op.create_index(
                "ix_user_connections_internal_user_id",
                "user_connections",
                ["internal_user_id"],
                unique=False,
            )
        if not column_exists("user_connections", "composio_entity_id"):
            op.add_column(
                "user_connections",
                sa.Column("composio_entity_id", sa.String(length=255), nullable=True),
            )
            op.create_index(
                "ix_user_connections_composio_entity_id",
                "user_connections",
                ["composio_entity_id"],
                unique=False,
            )
        if not column_exists("user_connections", "app_name"):
            op.add_column(
                "user_connections", sa.Column("app_name", sa.String(length=100), nullable=True)
            )


def downgrade() -> None:
    """Reverse the migration."""
    from sqlalchemy import inspect as sa_inspect
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    existing_tables = inspector.get_table_names()

    def table_exists(name: str) -> bool:
        return name in existing_tables

    def column_exists(table: str, col: str) -> bool:
        if not table_exists(table):
            return False
        return col in [c["name"] for c in inspector.get_columns(table)]

    # Remove columns from user_connections (SQLite doesn't support DROP COLUMN)
    dialect = bind.dialect.name
    if dialect != "sqlite":
        for col in ["app_name", "composio_entity_id", "internal_user_id"]:
            if column_exists("user_connections", col):
                op.drop_column("user_connections", col)

    if table_exists("composio_entities"):
        op.drop_index("ix_composio_entities_composio_entity_id", table_name="composio_entities")
        op.drop_index("ix_composio_entities_internal_user_id", table_name="composio_entities")
        op.drop_table("composio_entities")

    if table_exists("integrations_sessions"):
        op.drop_index("ix_integrations_sessions_user_id", table_name="integrations_sessions")
        op.drop_table("integrations_sessions")
