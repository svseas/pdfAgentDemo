"""Add summary_embedding column to document_summaries

Revision ID: add_summary_embedding
Revises: 1e5a233dc4cf
Create Date: 2024-03-14 00:23:22.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'add_summary_embedding'
down_revision: Union[str, None] = '1e5a233dc4cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create pgvector extension if it doesn't exist
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Add embedding column with vector type
    op.add_column('document_summaries', sa.Column('embedding', Vector(768), nullable=True))
    
    # Create cosine similarity search index
    op.execute(
        'CREATE INDEX ix_document_summaries_embedding_cosine ON document_summaries '
        'USING ivfflat (embedding vector_cosine_ops)'
    )


def downgrade() -> None:
    # Remove the index and column
    op.execute('DROP INDEX IF EXISTS ix_document_summaries_embedding_cosine')
    op.drop_column('document_summaries', 'embedding')