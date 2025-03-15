"""Document summarization routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from typing import Optional

from src.api.dependencies import (
    get_summarization_agent,
    get_db
)
from src.api.v1.middleware.error_handler import handle_errors
from src.api.v1.middleware.query_tracker import track_query
from src.api.v1.common.response_builder import ResponseBuilder
from src.domain.agents.recursive_summarization_agent import RecursiveSummarizationAgent
from src.schemas.rag import SummarizeRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/summarize")
@handle_errors
async def summarize_document(
    request: SummarizeRequest,
    summarization_agent: RecursiveSummarizationAgent = Depends(get_summarization_agent),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Generate a recursive summary of a document.
    
    Args:
        request: Summarization parameters
        summarization_agent: Agent for document summarization
        db: Database session
        
    Returns:
        Dict containing summaries at different levels
        
    Raises:
        HTTPException: If summarization fails
    """
    try:
        # Process request
        input_data = {
            "document_id": request.document_id,
            "language": request.language,
            "max_length": request.max_length
        }
        
        result = await summarization_agent.process(input_data)
        
        return ResponseBuilder.success(
            data={
                "document_id": input_data["document_id"],
                "chunk_summaries": result.get("chunk_summaries", []),
                "intermediate_summaries": result.get("intermediate_summaries", []),
                "final_summary": result.get("final_summary", "")
            },
            metadata={
                "language": request.language,
                "max_length": request.max_length,
                "chunk_count": len(result.get("chunk_summaries", [])),
                "intermediate_count": len(result.get("intermediate_summaries", [])),
                "has_final_summary": bool(result.get("final_summary")),
                **result.get("metadata", {})
            }
        )

    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error summarizing document: {str(e)}"
        )