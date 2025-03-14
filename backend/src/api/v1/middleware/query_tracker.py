"""Query tracking middleware for API routes."""
from typing import Callable, Any
from functools import wraps
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from datetime import datetime

from src.api.dependencies import get_db
from src.models.document_processing import OriginalUserQuery

def track_query(
    query_param: str = "query",
    is_system_query: bool = False
) -> Callable:
    """Decorator to track queries for API routes.
    
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
                if not original_query:
                    original_query = OriginalUserQuery(
                        query_text=query_text,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    db.add(original_query)
                    await db.flush()

                # Add query context to kwargs
                kwargs["original_query_id"] = original_query.id

                try:
                    # Execute route handler
                    result = await func(*args, db=db, **kwargs)
                    await db.commit()
                    return result

                except Exception as e:
                    await db.rollback()
                    raise

            except Exception as e:
                await db.rollback()
                raise

        return wrapper
    return decorator