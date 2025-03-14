"""Error handling middleware for API routes."""
from typing import Callable, Any
from functools import wraps
from fastapi import HTTPException
import logging

from src.domain.exceptions import (
    AgentError,
    ContextBuilderError,
    ChunkRetrievalError,
    ContextStorageError,
    RepositoryError
)

logger = logging.getLogger(__name__)

def handle_errors(func: Callable) -> Callable:
    """Decorator to handle common errors in API routes."""
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except AgentError as e:
            logger.error(f"Agent error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except ContextBuilderError as e:
            logger.error(f"Context builder error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except ChunkRetrievalError as e:
            logger.error(f"Chunk retrieval error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except ContextStorageError as e:
            logger.error(f"Context storage error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except RepositoryError as e:
            logger.error(f"Repository error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred"
            )
    return wrapper