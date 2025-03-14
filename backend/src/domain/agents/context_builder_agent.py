"""Context building and retrieval agent."""
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.context_repository import ContextRepository
from src.repositories.query_repository import QueryRepository
from src.repositories.document_repository import DocumentRepository
from src.models.document_processing import SubQuery
from src.domain.query_processor import QueryProcessor
from src.domain.exceptions import (
    ContextBuilderError,
    ChunkRetrievalError,
    ContextStorageError
)
from src.domain.agents.base_agent import BaseAgent
from src.schemas.rag import ContextBuilderRequest, ContextBuilderResponse

logger = logging.getLogger(__name__)

class ContextBuilderAgent(BaseAgent):
    """Agent that builds context for queries using vector search.
    
    This agent is responsible for:
    - Retrieving relevant context using pre-computed query embeddings
    - Adding surrounding context for better coherence
    - Tracking and storing context results
    - Formatting context for LLM consumption
    
    The agent expects queries and sub-queries to already have embeddings stored
    in the database. It does not generate embeddings itself.
    """
    def __init__(
        self,
        session: AsyncSession,
        context_repo: ContextRepository,
        query_repo: QueryRepository,
        doc_repo: DocumentRepository,
        query_processor: QueryProcessor,
        *args,
        **kwargs
    ):
        """Initialize context builder agent.
        
        Args:
            session: Database session
            context_repo: Repository for context operations
            query_repo: Repository for query operations
            doc_repo: Repository for document operations
            query_processor: Processor for finding relevant chunks
            *args, **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(session, *args, **kwargs)
        self.query_processor = query_processor
        self.context_repo = context_repo
        self.query_repo = query_repo
        self.doc_repo = doc_repo

    async def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of context building logic.
        
        Args:
            input_data: Must contain:
                - sub_query_id: ID of sub-query to process
                - query_text: Text of query to process
                - is_original: Whether this is the original query
                - top_k: Optional number of results to return (default 5)
                
        Returns:
            Dict containing context building results matching ContextBuilderResponse schema
                
        Raises:
            ContextBuilderError: If context building fails
        """
        try:
            sub_query_id = input_data.get("sub_query_id")
            if not sub_query_id:
                raise ContextBuilderError("No sub-query ID provided")

            query_text = input_data.get("query_text")
            if not query_text:
                raise ContextBuilderError("No query text provided")

            # Get query embedding from database
            sub_query = await self._get_sub_query(sub_query_id)
            if not sub_query:
                raise ContextBuilderError(f"Sub-query {sub_query_id} not found")
                
            if sub_query.sub_query_embedding is None:
                raise ContextBuilderError(f"No embedding found for sub-query {sub_query_id}")

            # Get relevant chunks using query processor
            context_results = await self._get_relevant_chunks(
                query_text=query_text,
                query_embedding=sub_query.sub_query_embedding,
                top_k=input_data.get("top_k", 5)
            )

            # Add surrounding context
            context_results = await self._add_surrounding_context(context_results)
            
            # Format context data
            context_data = {
                "total_chunks": len(context_results),
                "total_tokens": self._estimate_tokens(context_results),
                "chunks": [self._format_chunk(chunk) for chunk in context_results]
            }
            
            # Store complete context in context_sets table
            context_set_id = await self.context_repo.create_context_set(
                original_query_id=sub_query.original_query_id,
                context_data=context_data,
                context_metadata={
                    "total_chunks": len(context_results),
                    "total_tokens": self._estimate_tokens(context_results),
                    "has_direct_matches": any(not chunk.get("is_context", False) for chunk in context_results),
                    "average_relevance": sum(chunk.get("relevance", 0.0) for chunk in context_results) / len(context_results) if context_results else 0.0
                }
            )
            
            # Format response
            return {
                "status": "success",
                "context_set_id": context_set_id,
                "original_query": query_text if input_data.get("is_original") else None,
                "context": context_data
            }
            
        except ContextBuilderError as e:
            raise e
        except Exception as e:
            raise ContextBuilderError(f"Context building failed: {str(e)}") from e

    async def _get_sub_query(self, sub_query_id: int) -> Optional[Any]:
        """Get sub-query with its embedding from database.
        
        Args:
            sub_query_id: ID of the sub-query
            
        Returns:
            Sub-query object with embedding if found, None otherwise
            
        Raises:
            ContextBuilderError: If query retrieval fails
        """
        try:
            result = await self.session.execute(
                select(SubQuery).where(SubQuery.id == sub_query_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise ContextBuilderError(f"Failed to get sub-query: {str(e)}") from e

    async def _get_relevant_chunks(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Get relevant chunks using pre-computed query embedding.
        
        Args:
            query_text: Original query text for logging
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
            
        Raises:
            ChunkRetrievalError: If chunk retrieval fails
        """
        try:
            # Use PostgreSQL vector similarity search
            similar_chunks = await self.doc_repo.get_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=0.5
            )
            
            # Convert similarity scores to relevance scores
            for chunk in similar_chunks:
                chunk["relevance"] = chunk.pop("similarity", 0.0)
            
            return similar_chunks
        except Exception as e:
            raise ChunkRetrievalError(f"Failed to get relevant chunks: {str(e)}") from e

    async def _add_surrounding_context(
        self,
        chunks: List[Dict[str, Any]],
        window_size: int = 1
    ) -> List[Dict[str, Any]]:
        """Add surrounding chunks for better context.
        
        Args:
            chunks: List of retrieved chunks
            window_size: Number of chunks to add on each side
            
        Returns:
            List of chunks with surrounding context added
            
        Raises:
            ChunkRetrievalError: If context expansion fails
        """
        try:
            expanded_chunks = []
            seen_chunks = set()

            for chunk in chunks:
                # Skip if we've already included this chunk
                if chunk["id"] in seen_chunks:
                    continue
                seen_chunks.add(chunk["id"])
                expanded_chunks.append(chunk)

                # Get surrounding chunks
                surrounding = await self.doc_repo.get_surrounding_chunks(
                    doc_metadata_id=chunk["doc_metadata_id"],
                    chunk_index=chunk["chunk_index"],
                    window_size=window_size
                )

                # Add surrounding chunks if not already included
                for ctx_chunk in surrounding:
                    if ctx_chunk.id not in seen_chunks:
                        seen_chunks.add(ctx_chunk.id)
                        expanded_chunks.append({
                            "id": ctx_chunk.id,
                            "content": ctx_chunk.content,
                            "chunk_index": ctx_chunk.chunk_index,
                            "doc_metadata_id": ctx_chunk.doc_metadata_id,
                            "is_context": True,
                            "relevance": 0.0  # Context chunks get 0 relevance
                        })

            # Sort by document ID and chunk index
            expanded_chunks.sort(key=lambda x: (x["doc_metadata_id"], x["chunk_index"]))
            return expanded_chunks

        except Exception as e:
            raise ChunkRetrievalError(f"Failed to add surrounding context: {str(e)}") from e

    def _format_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Format chunk data according to schema.
        
        Args:
            chunk: Raw chunk data
            
        Returns:
            Formatted chunk data
        """
        return {
            "id": chunk["id"],
            "document_id": chunk["doc_metadata_id"],
            "document_name": chunk.get("filename", "Unknown"),
            "chunk_index": chunk["chunk_index"],
            "is_direct_match": not chunk.get("is_context", False),
            "relevance_score": chunk.get("relevance", 0.0),
            "text": chunk["content"]
        }

    def _estimate_tokens(self, chunks: List[Dict[str, Any]]) -> int:
        """Estimate token count for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 4 chars per token
        total_chars = sum(len(chunk["content"]) for chunk in chunks)
        return total_chars // 4