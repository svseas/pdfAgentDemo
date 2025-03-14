"""Fix sub_queries original_query reference

Revision ID: fix_sub_queries_original_query_reference
Revises: update_sub_query_embeddings
Create Date: 2025-03-14 18:13:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'fix_sub_queries_ref'
down_revision = 'update_sub_query_embeddings'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Drop the existing foreign key constraint
    op.drop_constraint('sub_queries_original_query_id_fkey', 'sub_queries', type_='foreignkey')
    
    # Update sub_queries to point to the original query from original_user_queries
    # by going through user_queries -> original_user_queries relationship
    op.execute("""
        WITH RECURSIVE query_chain AS (
            -- Start with sub_queries and their immediate user_queries
            SELECT
                sq.id as sub_query_id,
                sq.original_query_id as current_query_id,
                1 as depth
            FROM sub_queries sq
            
            UNION ALL
            
            -- Follow the chain through user_queries until we hit original_user_queries
            SELECT
                qc.sub_query_id,
                uq.original_query_id,
                qc.depth + 1
            FROM query_chain qc
            JOIN user_queries uq ON uq.id = qc.current_query_id
            WHERE uq.original_query_id IS NOT NULL
            AND qc.depth < 10  -- Prevent infinite recursion
        )
        -- Take the final mapping and update sub_queries
        UPDATE sub_queries sq
        SET original_query_id = (
            SELECT uq.original_query_id
            FROM query_chain qc
            JOIN user_queries uq ON uq.id = qc.current_query_id
            WHERE qc.sub_query_id = sq.id
            ORDER BY qc.depth DESC
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1
            FROM query_chain qc
            JOIN user_queries uq ON uq.id = qc.current_query_id
            WHERE qc.sub_query_id = sq.id
        );
    """)

    # Add the new foreign key constraint
    op.create_foreign_key(
        'sub_queries_original_query_id_fkey',
        'sub_queries',
        'original_user_queries',
        ['original_query_id'],
        ['id']
    )

def downgrade() -> None:
    # Drop the new foreign key constraint
    op.drop_constraint('sub_queries_original_query_id_fkey', 'sub_queries', type_='foreignkey')
    
    # Restore the original foreign key constraint
    op.create_foreign_key(
        'sub_queries_original_query_id_fkey',
        'sub_queries',
        'user_queries',
        ['original_query_id'],
        ['id']
    )