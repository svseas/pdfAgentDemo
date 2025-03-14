"""Update sub-query embeddings to use vector type

Revision ID: update_sub_query_embeddings
Revises: update_query_embeddings
Create Date: 2025-03-14 15:35:09.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = 'update_sub_query_embeddings'
down_revision = 'update_query_embeddings'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Drop existing column
    op.drop_column('sub_queries', 'sub_query_embedding')
    
    # Add vector column
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.add_column('sub_queries',
        sa.Column('sub_query_embedding', Vector(768), nullable=True)
    )
    
    # Create index for similarity search
    op.execute(
        'CREATE INDEX ix_sub_queries_embedding_cosine ON sub_queries '
        'USING ivfflat (sub_query_embedding vector_cosine_ops)'
    )

def downgrade() -> None:
    # Drop vector column and index
    op.drop_index('ix_sub_queries_embedding_cosine', table_name='sub_queries')
    op.drop_column('sub_queries', 'sub_query_embedding')
    
    # Add back JSON column
    op.add_column('sub_queries',
        sa.Column('sub_query_embedding', sa.JSON(), nullable=True)
    )