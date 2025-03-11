"""rename metadata_id to doc_metadata_id

Revision ID: 1e5a233dc4cf
Revises: 
Create Date: 2025-03-11 00:15:19.535800

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1e5a233dc4cf'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename the column
    op.alter_column(
        'documents',
        'metadata_id',
        new_column_name='doc_metadata_id',
        existing_type=sa.Integer(),
        existing_nullable=False
    )

    # Drop and recreate the foreign key constraint
    op.drop_constraint(
        'documents_metadata_id_fkey',
        'documents',
        type_='foreignkey'
    )
    op.create_foreign_key(
        'documents_doc_metadata_id_fkey',
        'documents',
        'document_metadata',
        ['doc_metadata_id'],
        ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    # Drop and recreate the foreign key constraint
    op.drop_constraint(
        'documents_doc_metadata_id_fkey',
        'documents',
        type_='foreignkey'
    )
    op.create_foreign_key(
        'documents_metadata_id_fkey',
        'documents',
        'document_metadata',
        ['metadata_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # Rename the column back
    op.alter_column(
        'documents',
        'doc_metadata_id',
        new_column_name='metadata_id',
        existing_type=sa.Integer(),
        existing_nullable=False
    )
