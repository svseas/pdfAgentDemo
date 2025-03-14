"""Workflow tracking middleware for API routes."""
from typing import Callable, Any, Optional
from functools import wraps
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from datetime import datetime

from src.api.dependencies import get_db
from src.models.workflow import WorkflowRun, UserQuery, OriginalUserQuery
from src.repositories.enums import WorkflowStatus, QueryType

def track_workflow(
    query_param: str = "query",
    is_system_query: bool = False
) -> Callable:
    """Decorator to track workflow for API routes.
    
    Args:
        query_param: Name of the parameter containing the query text
        is_system_query: Whether this is a system-generated query
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(
            *args: Any,
            db: AsyncSession = Depends(get_db),
            **kwargs: Any
        ) -> Any:
            # Extract query text from request
            request_data = kwargs.get(query_param)
            if not request_data:
                return await func(*args, **kwargs)

            try:
                # Handle different request types
                if hasattr(request_data, "query"):
                    query_text = request_data.query
                elif hasattr(request_data, "query_text"):
                    query_text = request_data.query_text
                elif isinstance(request_data, str):
                    query_text = request_data
                else:
                    # For other request types, create a descriptive query text
                    query_text = str(request_data)

                # Check if original query exists (case-insensitive)
                result = await db.execute(
                    select(OriginalUserQuery).where(
                        text("LOWER(query_text) = LOWER(:query_text)")
                    ).params(query_text=query_text)
                )
                original_query = result.scalar_one_or_none()

                # Create original query if it doesn't exist
                # Always create original query since sub-queries need it
                if not original_query:
                    original_query = OriginalUserQuery(
                        query_text=query_text,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    db.add(original_query)
                    await db.flush()

                # Create user query
                query = UserQuery(
                    query_text=query_text,
                    original_query_id=original_query.id,  # Always link to original
                    created_at=datetime.now()
                )
                db.add(query)
                await db.flush()
                query_id = query.id

                # Create workflow run
                workflow = WorkflowRun(
                    user_query_id=query_id,
                    status=WorkflowStatus.RUNNING
                )
                db.add(workflow)
                await db.flush()
                workflow_run_id = workflow.id

                # Add workflow context to kwargs
                kwargs["workflow_run_id"] = workflow_run_id
                kwargs["query_id"] = query_id
                kwargs["original_query_id"] = original_query.id

                try:
                    # Execute route handler
                    result = await func(*args, db=db, **kwargs)

                    # Update workflow status on success
                    workflow.status = WorkflowStatus.COMPLETED
                    await db.commit()

                    return result

                except Exception as e:
                    # Update workflow status on failure
                    workflow.status = WorkflowStatus.FAILED
                    await db.commit()
                    raise

            except Exception as e:
                await db.rollback()
                raise

        return wrapper
    return decorator