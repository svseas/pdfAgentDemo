"""Add original queries table

Revision ID: add_original_queries
Revises: add_summary_embedding
Create Date: 2025-03-14 14:44:43.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_original_queries'
down_revision = 'add_summary_embedding'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Check if original_user_queries table exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()
    
    if 'original_user_queries' not in tables:
        # Create original_user_queries table
        op.create_table(
            'original_user_queries',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('query_text', sa.String(), nullable=False),
            sa.Column('query_embedding', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )

    # Check if original_query_id column exists in user_queries
    columns = [c['name'] for c in inspector.get_columns('user_queries')]
    if 'original_query_id' not in columns:
        # Add original_query_id to user_queries table
        op.add_column('user_queries', sa.Column('original_query_id', sa.Integer(), nullable=True))
        op.create_foreign_key(
            'fk_user_queries_original_query',
            'user_queries',
            'original_user_queries',
            ['original_query_id'],
            ['id']
        )

        # Copy existing queries to original_user_queries and update references
        op.execute("""
            INSERT INTO original_user_queries (query_text, query_embedding, created_at, updated_at)
            SELECT DISTINCT ON (query_text) query_text, query_embedding, created_at, created_at
            FROM user_queries;
            
            UPDATE user_queries uq
            SET original_query_id = ouq.id
            FROM original_user_queries ouq
            WHERE uq.query_text = ouq.query_text;
        """)

def downgrade() -> None:
    op.drop_constraint('fk_user_queries_original_query', 'user_queries', type_='foreignkey')
    op.drop_column('user_queries', 'original_query_id')
    op.drop_table('original_user_queries')