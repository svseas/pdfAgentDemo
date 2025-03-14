"""Remove workflow-related tables.

Revision ID: remove_workflow_tables
Revises: update_sub_query_embeddings
Create Date: 2025-03-15 03:45:22.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'remove_workflow_tables'
down_revision = 'update_sub_query_embeddings'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # First remove all foreign key constraints
    op.drop_constraint('context_results_agent_step_id_fkey', 'context_results', type_='foreignkey')
    op.drop_constraint('sub_queries_workflow_run_id_fkey', 'sub_queries', type_='foreignkey')
    op.drop_constraint('response_citations_workflow_run_id_fkey', 'response_citations', type_='foreignkey')
    op.drop_constraint('context_sets_workflow_run_id_fkey', 'context_sets', type_='foreignkey')
    op.drop_constraint('agent_steps_workflow_run_id_fkey', 'agent_steps', type_='foreignkey')
    op.drop_constraint('workflow_runs_user_query_id_fkey', 'workflow_runs', type_='foreignkey')
    op.drop_constraint('user_queries_original_query_id_fkey', 'user_queries', type_='foreignkey')

    # Then remove columns
    op.drop_column('context_results', 'agent_step_id')
    op.drop_column('sub_queries', 'workflow_run_id')
    op.drop_column('response_citations', 'workflow_run_id')
    op.drop_column('context_sets', 'workflow_run_id')

    # Finally drop tables
    op.drop_table('agent_steps')
    op.drop_table('workflow_runs')
    op.drop_table('user_queries')

def downgrade() -> None:
    # Recreate user_queries table
    op.create_table('user_queries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('query_text', sa.String(), nullable=False),
        sa.Column('original_query_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_foreign_key(
        'user_queries_original_query_id_fkey',
        'user_queries', 'original_user_queries',
        ['original_query_id'], ['id']
    )

    # Recreate workflow_runs table
    op.create_table('workflow_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_query_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('final_answer', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_foreign_key(
        'workflow_runs_user_query_id_fkey',
        'workflow_runs', 'user_queries',
        ['user_query_id'], ['id']
    )

    # Recreate agent_steps table
    op.create_table('agent_steps',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('workflow_run_id', sa.Integer(), nullable=False),
        sa.Column('agent_type', sa.String(), nullable=False),
        sa.Column('input_data', sa.JSON(), nullable=True),
        sa.Column('output_data', sa.JSON(), nullable=True),
        sa.Column('error', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_foreign_key(
        'agent_steps_workflow_run_id_fkey',
        'agent_steps', 'workflow_runs',
        ['workflow_run_id'], ['id']
    )

    # Add columns back
    op.add_column('context_sets',
        sa.Column('workflow_run_id', sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        'context_sets_workflow_run_id_fkey',
        'context_sets', 'workflow_runs',
        ['workflow_run_id'], ['id']
    )

    op.add_column('response_citations',
        sa.Column('workflow_run_id', sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        'response_citations_workflow_run_id_fkey',
        'response_citations', 'workflow_runs',
        ['workflow_run_id'], ['id']
    )

    op.add_column('sub_queries',
        sa.Column('workflow_run_id', sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        'sub_queries_workflow_run_id_fkey',
        'sub_queries', 'workflow_runs',
        ['workflow_run_id'], ['id']
    )

    op.add_column('context_results',
        sa.Column('agent_step_id', sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        'context_results_agent_step_id_fkey',
        'context_results', 'agent_steps',
        ['agent_step_id'], ['id']
    )