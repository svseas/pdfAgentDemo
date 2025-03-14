"""Query synthesis routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from typing import Optional

from src.api.dependencies import (
    get_query_synthesizer_agent,
    get_db
)
from src.api.v1.middleware.error_handler import handle_errors
from src.api.v1.middleware.query_tracker import track_query
from src.api.v1.common.response_builder import ResponseBuilder
from src.domain.agents.query_synthesizer_agent import QuerySynthesizerAgent
from src.models.document_processing import OriginalUserQuery
from src.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/synthesize")
@handle_errors
@track_query(query_param="request")
async def synthesize_query(
    request: dict,
    synthesizer: QuerySynthesizerAgent = Depends(get_query_synthesizer_agent),
    db: AsyncSession = Depends(get_db),
    original_query_id: Optional[int] = None
) -> dict:
    """Synthesize final answer using analyzed query and context.
    
    Args:
        request: Synthesis parameters
        synthesizer: Agent for answer synthesis
        db: Database session
        original_query_id: ID of the original query (added by middleware)
        
    Returns:
        Dict containing synthesized answer
        
    Raises:
        HTTPException: If synthesis fails
    """
    try:
        # Get original query
        result = await db.execute(
            select(OriginalUserQuery).where(
                OriginalUserQuery.id == original_query_id
            )
        )
        original_query = result.scalar_one_or_none()
        if not original_query:
            raise HTTPException(
                status_code=404,
                detail=f"Original query {original_query_id} not found"
            )

        # Process request
        input_data = {
            "original_query_id": original_query_id,
            "query_text": original_query.query_text,
            "context": request.get("context", []),
            "citations": request.get("citations", []),
            "language": request.get("language", "vietnamese"),
            "temperature": request.get("temperature", 0.7)
        }
        
        result = await synthesizer.process(input_data)
        
        return ResponseBuilder.success(
            data={
                "answer": result["answer"],
                "original_query": original_query.query_text
            },
            metadata={
                "language": input_data["language"],
                "temperature": input_data["temperature"],
                "context_chunks": len(input_data["context"]),
                "citations_used": len(input_data["citations"])
            }
        )

    except Exception as e:
        logger.error(f"Error synthesizing answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error synthesizing answer: {str(e)}"
        )