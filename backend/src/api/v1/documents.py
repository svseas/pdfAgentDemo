from fastapi import APIRouter, Depends
import numpy as np
import logging

from src.domain.embedding_generator import EmbeddingGenerator
from src.domain.query_processor import QueryProcessor
from src.api.dependencies import get_embedding_generator, get_query_processor, get_document_repository
from src.repositories.document_repository import DocumentRepository
from src.schemas.rag import (
    VectorizeRequest,
    VectorResponse,
    SearchRequest,
    ContextRequest,
    QueryRequest
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/query", response_model=dict)
async def query_documents(
    request: QueryRequest,
    query_service: QueryProcessor = Depends(get_query_processor),
    doc_repository: DocumentRepository = Depends(get_document_repository)
) -> dict:
    """Complete RAG workflow: search documents and generate response"""
    try:
        # Get query embedding
        query_embedding = query_service.embedding_generator.generate_embedding(request.query)
        
        # Get similar chunks from repository
        similar_chunks = await doc_repository.get_similar_chunks(
            query_embedding=query_embedding.tolist(),
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        if not similar_chunks:
            return {
                "response": "Không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
                "relevant_chunks": []
            }
        
        # Sort chunks by similarity score for better context building
        sorted_chunks = sorted(similar_chunks, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Generate response using the chunks
        response = await query_service.generate_response(
            request.query,
            sorted_chunks,
            request.temperature
        )
        
        return {
            "response": response,
            "relevant_chunks": sorted_chunks,  # Include sorted chunks for transparency
            "total_chunks": len(sorted_chunks)
        }
    except Exception as e:
        return {
            "response": f"Có lỗi xảy ra khi xử lý câu hỏi: {str(e)}",
            "relevant_chunks": [],
            "error": str(e)
        }

@router.post("/vectorize", response_model=VectorResponse)
async def vectorize_text(
    request: VectorizeRequest,
    embedding_service: EmbeddingGenerator = Depends(get_embedding_generator)
) -> VectorResponse:
    """Convert text to vector form using embeddings"""
    embedding = embedding_service.generate_embedding(request.text)
    return VectorResponse(vector=embedding.tolist())

@router.post("/search", response_model=dict)
async def semantic_search(
    request: SearchRequest,
    query_service: QueryProcessor = Depends(get_query_processor),
    doc_repository: DocumentRepository = Depends(get_document_repository)
) -> dict:
    """Find relevant document chunks using semantic search with GRAG enhancement"""
    try:
        logger.info(f"Processing search request for query: {request.query}")
        
        # Get query embedding
        query_embedding = query_service.embedding_generator.generate_embedding(request.query)
        
        # Get initial chunks from repository
        initial_chunks = await doc_repository.get_similar_chunks(
            query_embedding=query_embedding.tolist(),
            top_k=request.top_k * 2,  # Get more chunks for GRAG to work with
            similarity_threshold=request.similarity_threshold
        )
        
        if not initial_chunks:
            logger.info("No relevant chunks found")
            return {"relevant_chunks": []}
            
        # Use QueryProcessor's get_relevant_chunks for GRAG enhancement
        logger.info("Applying GRAG reranking to initial chunks")
        reranked_chunks = query_service.get_relevant_chunks(
            query=request.query,
            doc_chunks=initial_chunks,
            top_k=request.top_k,
            use_grag=True  # Explicitly enable GRAG
        )
        
        logger.info(f"Reranking complete, returning {len(reranked_chunks)} chunks")
        return {"relevant_chunks": reranked_chunks}
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise

@router.post("/build-context", response_model=dict)
async def build_context(
    request: ContextRequest,
    query_service: QueryProcessor = Depends(get_query_processor)
) -> dict:
    """Build context from relevant chunks for LLM"""
    response = await query_service.generate_response(
        request.query,
        request.relevant_chunks,
        request.temperature
    )
    return {"response": response}
