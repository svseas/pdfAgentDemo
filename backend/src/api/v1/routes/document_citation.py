"""Document citation extraction and management routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from typing import Optional

from src.api.dependencies import (
    get_citation_agent,
    get_query_synthesizer_agent,
    get_db
)
from src.api.v1.middleware.error_handler import handle_errors
from src.api.v1.middleware.query_tracker import track_query
from src.api.v1.common.response_builder import ResponseBuilder
from src.domain.agents.citation_agent import CitationAgent
from src.domain.agents.query_synthesizer_agent import QuerySynthesizerAgent
from src.schemas.rag import (
    CitationRequest,
    SynthesisRequest
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/extract-citations")
@handle_errors
@track_query(query_param="request")
async def extract_citations(
    request: CitationRequest,
    citation_agent: CitationAgent = Depends(get_citation_agent),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Extract relevant citations from document.
    
    Args:
        request: Citation extraction parameters
        citation_agent: Agent for citation extraction
        db: Database session
        original_query_id: ID of the original query (added by middleware)
        
    Returns:
        Dict containing extracted citations
        
    Raises:
        HTTPException: If citation extraction fails
    """
    try:
        # Process request
        input_data = {
            "original_query_id": request.original_query_id,
            "document_id": int(request.document_id),
            "query_text": request.query,
            "language": request.language
        }
        
        result = await citation_agent.process(input_data)
        
        return ResponseBuilder.success(
            data={"citations": result.get("citations", [])},
            metadata={
                "document_id": request.document_id,
                "language": request.language,
                "total_citations": len(result.get("citations", []))
            }
        )

    except Exception as e:
        logger.error(f"Error extracting citations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting citations: {str(e)}"
        )

@router.post("/synthesize")
@handle_errors
async def synthesize_answer(
    request: SynthesisRequest,
    synthesizer: QuerySynthesizerAgent = Depends(get_query_synthesizer_agent)
) -> dict:
    """Synthesize final answer using analyzed query, context and citations.
    
    Args:
        request: Synthesis parameters
        synthesizer: Agent for answer synthesis
        
    Returns:
        Dict containing synthesized answer
        
    Raises:
        HTTPException: If synthesis fails
    """
    try:
        answer = await synthesizer.synthesize(
            query=request.query,
            analyzed_query=request.analyzed_query,
            context=request.context,
            citations=request.citations,
            language=request.language,
            temperature=request.temperature
        )
        
        return ResponseBuilder.success(
            data={"answer": answer},
            metadata={
                "language": request.language,
                "temperature": request.temperature,
                "context_chunks": len(request.context),
                "citations_used": len(request.citations)
            }
        )

    except Exception as e:
        logger.error(f"Error synthesizing answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error synthesizing answer: {str(e)}"
        )