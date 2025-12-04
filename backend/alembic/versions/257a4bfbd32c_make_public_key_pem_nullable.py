"""make_public_key_pem_nullable

Revision ID: 257a4bfbd32c
Revises: 8ccfe035d7b3
Create Date: 2025-12-04 11:34:33.853565

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '257a4bfbd32c'
down_revision: Union[str, Sequence[str], None] = '8ccfe035d7b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Make public_key_pem nullable
    op.alter_column('agents', 'public_key_pem',
                    existing_type=sa.Text(),
                    nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    # Make public_key_pem NOT NULL again
    op.alter_column('agents', 'public_key_pem',
                    existing_type=sa.Text(),
                    nullable=False)
