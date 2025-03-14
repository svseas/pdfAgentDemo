"""Query analysis routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import logging
from typing import Optional

from src.api.dependencies import (
    get_query_analyzer_agent,
    get_embedding_generator,
    get_db
)
from src.api.v1.middleware.error_handler import handle_errors
from src.api.v1.middleware.query_tracker import track_query
from src.api.v1.common.response_builder import ResponseBuilder
from src.domain.agents.query_analyzer_agent import QueryAnalyzerAgent
from src.domain.embedding_generator import EmbeddingGenerator
from src.models.document_processing import OriginalUserQuery
from src.schemas.rag import QueryAnalysisRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/analyze-query")
@handle_errors
@track_query(query_param="request")
async def analyze_query(
    request: QueryAnalysisRequest,
    query_analyzer: QueryAnalyzerAgent = Depends(get_query_analyzer_agent),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Analyze query using stepback prompting.
    
    Args:
        request: Query analysis parameters
        query_analyzer: Agent for query analysis
        embedding_generator: Service for generating embeddings
        db: Database session
        original_query_id: ID of the original query (added by middleware)
        
    Returns:
        Dict containing query analysis results
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Generate embedding asynchronously
        logger.info("Generating embedding for query: %s", request.query)
        query_embedding = await embedding_generator.generate_embedding(request.query)
        logger.info(
            "Generated embedding shape: %s",
            query_embedding.shape if query_embedding is not None else None
        )

        # Update original query embedding
        if query_embedding is not None and request.original_query_id:
            original_query = await db.get(OriginalUserQuery, request.original_query_id)
            if original_query:
                original_query.query_embedding = query_embedding
                original_query.updated_at = datetime.now()
                await db.commit()
                logger.info("Successfully updated query embedding")
            else:
                logger.warning("Original query not found for embedding update")
        else:
            logger.warning("No embedding generated for query")
        
        # Process query
        input_data = {
            "query_id": request.original_query_id,
            "query_text": request.query,
            "language": request.language
        }
        
        # Let agent handle database operations
        analysis = await query_analyzer.process(input_data)
        
        return ResponseBuilder.success(
            data={
                "analysis": analysis,
                "original_query": request.query,
                "language": request.language
            },
            metadata={
                "has_embedding": query_embedding is not None,
                "sub_queries_generated": len(analysis.get("sub_queries", [])) if analysis else 0
            }
        )

    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing query: {str(e)}"
        )