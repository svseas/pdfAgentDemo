"""Update query embeddings to use vector type

Revision ID: update_query_embeddings
Revises: add_original_queries
Create Date: 2025-03-14 15:05:07.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'update_query_embeddings'
down_revision = 'add_original_queries'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Drop existing embedding columns
    op.drop_column('original_user_queries', 'query_embedding')
    op.drop_column('user_queries', 'query_embedding')
    
    # Add vector columns
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Add vector columns using raw SQL since SQLAlchemy doesn't have direct pgvector support
    op.execute('ALTER TABLE original_user_queries ADD COLUMN query_embedding vector(768)')
    op.execute('ALTER TABLE user_queries ADD COLUMN query_embedding vector(768)')
    
    # Create index for similarity search on original queries
    op.execute(
        'CREATE INDEX ix_original_user_queries_embedding_cosine ON original_user_queries '
        'USING ivfflat (query_embedding vector_cosine_ops)'
    )

def downgrade() -> None:
    # Drop vector columns and index
    op.drop_index('ix_original_user_queries_embedding_cosine', table_name='original_user_queries')
    op.drop_column('original_user_queries', 'query_embedding')
    op.drop_column('user_queries', 'query_embedding')
    
    # Add back JSON columns
    op.add_column('original_user_queries',
        sa.Column('query_embedding', sa.JSON(), nullable=True)
    )
    op.add_column('user_queries',
        sa.Column('query_embedding', sa.JSON(), nullable=True)
    )