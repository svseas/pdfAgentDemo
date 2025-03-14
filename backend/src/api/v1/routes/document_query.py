"""Document querying and search routes."""
from fastapi import APIRouter, Depends
import logging
from typing import Optional

from src.api.dependencies import (
    get_query_processor,
    get_document_repository,
    get_embedding_generator
)
from src.api.v1.middleware.error_handler import handle_errors
from src.api.v1.middleware.query_tracker import track_query
from src.api.v1.common.response_builder import ResponseBuilder
from src.domain.query_processor import QueryProcessor
from src.domain.embedding_generator import EmbeddingGenerator
from src.repositories.document_repository import DocumentRepository
from src.schemas.rag import (
    QueryRequest,
    VectorizeRequest,
    VectorResponse,
    SearchRequest
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/query")
@handle_errors
@track_query(query_param="request")
async def query_documents(
    request: QueryRequest,
    query_service: QueryProcessor = Depends(get_query_processor),
    doc_repository: DocumentRepository = Depends(get_document_repository)
) -> dict:
    """Complete RAG workflow: search documents and generate response.
    
    Args:
        request: Query parameters including text and search settings
        query_service: Service for query processing
        doc_repository: Repository for document operations
        original_query_id: ID of the original query (added by middleware)
        
    Returns:
        Dict containing generated response and relevant chunks
    """
    try:
        # Get query embedding
        query_embedding = query_service.embedding_generator.generate_embedding(
            request.query
        )
        
        # Get similar chunks from repository
        similar_chunks = await doc_repository.get_similar_chunks(
            query_embedding=query_embedding.tolist(),
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        if not similar_chunks:
            return ResponseBuilder.query_response(
                response="Không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
                relevant_chunks=[],
                metadata={"chunks_found": 0}
            )
        
        # Sort chunks by similarity score for better context building
        sorted_chunks = sorted(
            similar_chunks,
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )
        
        # Generate response using the chunks
        response = await query_service.generate_response(
            request.query,
            sorted_chunks,
            request.temperature
        )
        
        return ResponseBuilder.query_response(
            response=response,
            relevant_chunks=sorted_chunks,
            metadata={
                "chunks_found": len(sorted_chunks),
                "temperature": request.temperature
            }
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return ResponseBuilder.error(
            message=f"Có lỗi xảy ra khi xử lý câu hỏi: {str(e)}",
            details={"relevant_chunks": []}
        )

@router.post("/vectorize", response_model=VectorResponse)
@handle_errors
async def vectorize_text(
    request: VectorizeRequest,
    embedding_service: EmbeddingGenerator = Depends(get_embedding_generator)
) -> VectorResponse:
    """Convert text to vector form using embeddings.
    
    Args:
        request: Text to vectorize
        embedding_service: Service for generating embeddings
        
    Returns:
        Vector representation of input text
    """
    embedding = embedding_service.generate_embedding(request.text)
    return VectorResponse(vector=embedding.tolist())

@router.post("/search")
@handle_errors
@track_query(query_param="request")
async def semantic_search(
    request: SearchRequest,
    query_service: QueryProcessor = Depends(get_query_processor),
    doc_repository: DocumentRepository = Depends(get_document_repository),
    original_query_id: Optional[int] = None
) -> dict:
    """Find relevant document chunks using semantic search with GRAG enhancement.
    
    Args:
        request: Search parameters
        query_service: Service for query processing
        doc_repository: Repository for document operations
        original_query_id: ID of the original query (added by middleware)
        
    Returns:
        Dict containing relevant chunks and search metadata
    """
    try:
        logger.info(f"Processing search request for query: {request.query}")
        
        # Get query embedding
        query_embedding = query_service.embedding_generator.generate_embedding(
            request.query
        )
        
        # Get initial chunks from repository
        initial_chunks = await doc_repository.get_similar_chunks(
            query_embedding=query_embedding.tolist(),
            top_k=request.top_k * 2,  # Get more chunks for GRAG to work with
            similarity_threshold=request.similarity_threshold
        )
        
        if not initial_chunks:
            logger.info("No relevant chunks found")
            return ResponseBuilder.chunk_response(
                chunks=[],
                total_chunks=0,
                metadata={"grag_enabled": request.use_grag}
            )
            
        # Use QueryProcessor's get_relevant_chunks for GRAG enhancement
        logger.info("Applying GRAG reranking to initial chunks")
        reranked_chunks = query_service.get_relevant_chunks(
            query=request.query,
            doc_chunks=initial_chunks,
            top_k=request.top_k,
            use_grag=request.use_grag
        )
        
        logger.info(f"Reranking complete, returning {len(reranked_chunks)} chunks")
        return ResponseBuilder.chunk_response(
            chunks=reranked_chunks,
            total_chunks=len(reranked_chunks),
            metadata={
                "grag_enabled": request.use_grag,
                "initial_chunks": len(initial_chunks)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise