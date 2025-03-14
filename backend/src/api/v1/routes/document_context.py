"""Document context building and management routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import logging
from typing import Optional

from src.api.dependencies import (
    get_query_processor,
    get_context_builder_agent,
    get_db
)
from src.api.v1.middleware.error_handler import handle_errors
from src.api.v1.middleware.query_tracker import track_query
from src.api.v1.common.response_builder import ResponseBuilder
from src.domain.query_processor import QueryProcessor
from src.domain.agents.context_builder_agent import ContextBuilderAgent
from src.models.document_processing import (
    OriginalUserQuery,
    SubQuery
)
from src.repositories.query_repository import QueryRepository
from src.schemas.rag import (
    ContextRequest,
    BuildQueryContextRequest,
    ContextBuilderResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/build-context")
@handle_errors
async def build_context(
    request: ContextRequest,
    query_service: QueryProcessor = Depends(get_query_processor)
) -> dict:
    """Build context from relevant chunks for LLM.
    
    Args:
        request: Context building parameters
        query_service: Service for query processing
        
    Returns:
        Dict containing generated response
    """
    response = await query_service.generate_response(
        request.query,
        request.relevant_chunks,
        request.temperature
    )
    return ResponseBuilder.success(
        data={"response": response},
        metadata={
            "chunks_used": len(request.relevant_chunks),
            "temperature": request.temperature
        }
    )

@router.post("/build-query-context", response_model=ContextBuilderResponse)
@handle_errors
@track_query(query_param="request", is_system_query=True)
async def build_query_context(
    request: BuildQueryContextRequest,
    context_builder: ContextBuilderAgent = Depends(get_context_builder_agent),
    db: AsyncSession = Depends(get_db),
    original_query_id: Optional[int] = None
) -> ContextBuilderResponse:
    """Build context from original query and its sub-queries.
    
    Args:
        request: Context building parameters
        context_builder: Agent for building context
        db: Database session
        original_query_id: ID of the original query (added by middleware)
        
    Returns:
        Complete context results including chunks and metadata
        
    Raises:
        HTTPException: If queries not found or context building fails
    """
    try:
        # Get original query
        result = await db.execute(
            select(OriginalUserQuery).where(
                OriginalUserQuery.id == request.original_query_id
            )
        )
        original_query = result.scalar_one_or_none()
        if not original_query:
            raise HTTPException(
                status_code=404,
                detail=f"Original query {request.original_query_id} not found"
            )

        # Get existing sub-queries
        result = await db.execute(
            select(SubQuery).where(
                SubQuery.original_query_id == request.original_query_id
            )
        )
        sub_queries = result.scalars().all()
        if not sub_queries:
            raise HTTPException(
                status_code=404,
                detail=f"No sub-queries found for original query {request.original_query_id}"
            )

        # Process each sub-query (including original)
        all_chunks = []
        last_context_set_id = None
        
        for sub_query in sub_queries:
            result = await context_builder.process({
                "sub_query_id": sub_query.id,
                "query_text": sub_query.sub_query_text,
                "is_original": sub_query.id == sub_queries[0].id,
                "top_k": request.top_k
            })
            
            # Store context set ID from first result
            if last_context_set_id is None:
                last_context_set_id = result.get("context_set_id")
            
            # Add chunks to combined list
            if "context" in result and "chunks" in result["context"]:
                all_chunks.extend(result["context"]["chunks"])
        
        # Deduplicate chunks by ID
        seen_chunk_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["id"] not in seen_chunk_ids:
                seen_chunk_ids.add(chunk["id"])
                unique_chunks.append(chunk)
        
        # Sort by document ID and chunk index
        unique_chunks.sort(key=lambda x: (x["document_id"], x["chunk_index"]))

        # Create final response
        return ResponseBuilder.success(
            data={
                "context_set_id": last_context_set_id,
                "original_query": original_query.query_text,
                "sub_queries": [{
                    "id": sq.id,
                    "text": sq.sub_query_text,
                    "is_original": sq.id == sub_queries[0].id
                } for sq in sub_queries],
                "context": {
                    "total_chunks": len(unique_chunks),
                    "total_tokens": sum(len(chunk["text"].split()) for chunk in unique_chunks),
                    "chunks": unique_chunks
                }
            },
            metadata={
                "sub_queries_processed": len(sub_queries),
                "unique_chunks": len(unique_chunks)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building query context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))