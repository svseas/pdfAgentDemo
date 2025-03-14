"""add_context_sets_table

Revision ID: be905c821acd
Revises: fix_sub_queries_ref
Create Date: 2025-03-14 22:48:06.278927

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'be905c821acd'
down_revision: Union[str, None] = 'fix_sub_queries_ref'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Import JSONB type
    from sqlalchemy.dialects.postgresql import JSONB
    
    # Create context_sets table
    op.create_table(
        'context_sets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('workflow_run_id', sa.Integer(), nullable=False),
        sa.Column('original_query_id', sa.Integer(), nullable=False),
        sa.Column('context_data', JSONB, nullable=False, comment='Complete context data including chunks'),
        sa.Column('context_metadata', JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"),
                 comment='Context metadata like total chunks, tokens, etc.'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['workflow_run_id'], ['workflow_runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['original_query_id'], ['original_user_queries.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('original_query_id', name='uq_context_sets_original_query_id')
    )

    # Add indexes
    op.create_index(
        'ix_context_sets_workflow_run_id',
        'context_sets',
        ['workflow_run_id']
    )
    op.create_index(
        'ix_context_sets_original_query_id',
        'context_sets',
        ['original_query_id']
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('ix_context_sets_original_query_id')
    op.drop_index('ix_context_sets_workflow_run_id')
    
    # Drop table
    op.drop_table('context_sets')
