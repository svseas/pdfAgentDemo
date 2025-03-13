"""Query processing implementation."""
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from pathlib import Path
import os
from src.core.config import settings
from src.core.llm_service import LLMService
from src.domain.interfaces import QueryProcessorInterface, LLMInterface, EmbeddingGeneratorInterface
from src.domain.exceptions import QueryProcessingError, LLMError

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """Calculate similarity between embeddings."""
    
    @staticmethod
    def calculate(
        query_embedding: np.ndarray,
        doc_embedding: np.ndarray,
        chunk_index: int
    ) -> float:
        """
        Calculate similarity score between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embedding: Document embedding vector
            chunk_index: Index of the chunk for position bias
            
        Returns:
            Similarity score
        """
        try:
            # Convert to numpy arrays if needed
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            if not isinstance(doc_embedding, np.ndarray):
                doc_embedding = np.array(doc_embedding, dtype=np.float32)
            
            # Ensure arrays are 1-dimensional
            query_embedding = query_embedding.flatten()
            doc_embedding = doc_embedding.flatten()
            
            # Check if shapes match
            if query_embedding.shape[0] != doc_embedding.shape[0]:
                logger.error(f"Shape mismatch: query {query_embedding.shape} vs doc {doc_embedding.shape}")
                return 0.0
                
            # Calculate cosine similarity
            query_norm = float(np.linalg.norm(query_embedding))
            doc_norm = float(np.linalg.norm(doc_embedding))
            
            if query_norm < 1e-10 or doc_norm < 1e-10:
                return 0.0
                
            cosine_sim = float(np.dot(query_embedding, doc_embedding)) / (query_norm * doc_norm)
            
            # Apply position bias
            position_weight = 1.0 / (1.0 + 0.1 * chunk_index)
            
            return cosine_sim * position_weight
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

class GRAGReranker:
    """GRAG-based document reranking."""
    
    def __init__(self):
        """Initialize GRAG reranker."""
        self.service = None
        
    def initialize(self) -> bool:
        """Initialize GRAG service."""
        try:
            from src.domain.grag.service import GRAGService
            
            # Use local model path
            cwd = os.getcwd()
            model_path = Path(cwd) / "models" / "amrbart"
            logger.info(f"Looking for model at: {model_path}")
            
            if not model_path.exists():
                logger.error(f"AMR model not found at {model_path}")
                return False
                
            self.service = GRAGService(
                embedding_model_name=settings.EMBEDDING_MODEL,
                amr_model_name=str(model_path)
            )
            logger.info("GRAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GRAG service: {e}")
            return False
            
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using GRAG.
        
        Args:
            query: User query
            documents: List of document chunks
            top_k: Number of top results to return
            
        Returns:
            Reranked document chunks
        """
        if not self.service:
            if not self.initialize():
                return documents[:top_k]
                
        try:
            original_order = [doc["id"] for doc in documents]
            
            reranked_docs = self.service.rerank(
                question=query,
                documents=documents,
                top_k=top_k
            )
            
            new_order = [doc["id"] for doc in reranked_docs]
            logger.info(f"GRAG reranking: {original_order} -> {new_order}")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"GRAG reranking error: {e}")
            return documents[:top_k]

class QueryProcessor(QueryProcessorInterface):
    """Process queries and generate responses."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGeneratorInterface,
        llm_service: LLMInterface,
        top_k: int = 3,
        use_grag: bool = False
    ):
        """
        Initialize query processor.
        
        Args:
            embedding_generator: Service for generating embeddings
            llm_service: Service for LLM interactions
            top_k: Number of top matches to return
            use_grag: Whether to use GRAG reranking
        """
        self.embedding_generator = embedding_generator
        self.llm_service = llm_service
        self.top_k = top_k
        self.use_grag = use_grag
        self.similarity_calculator = SimilarityCalculator()
        self.reranker = GRAGReranker() if use_grag else None
        
    def get_relevant_chunks(
        self,
        query: str,
        doc_chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get most relevant document chunks for query.
        
        Args:
            query: User query
            doc_chunks: List of document chunks with embeddings
            top_k: Optional override for number of results
            
        Returns:
            List of relevant chunks
            
        Raises:
            QueryProcessingError: If processing fails
        """
        if top_k is None:
            top_k = self.top_k

        try:
            logger.info(f"Processing query: {query}")
            logger.info(f"Total chunks: {len(doc_chunks)}")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Calculate similarities
            chunk_similarities = []
            for i, chunk in enumerate(doc_chunks):
                try:
                    if not chunk.get("embedding"):
                        logger.warning(f"Chunk {i} has no embedding")
                        continue
                    
                    similarity = self.similarity_calculator.calculate(
                        query_embedding,
                        chunk["embedding"],
                        i
                    )
                    
                    if similarity > 0:
                        chunk_similarities.append((chunk, similarity, i))
                        logger.debug(f"Similarity for chunk {i}: {similarity}")
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
            
            # Sort by similarity
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k and sort by position
            top_chunks = chunk_similarities[:top_k] if chunk_similarities else []
            top_chunks.sort(key=lambda x: x[2])
            
            result_chunks = [chunk for chunk, _, _ in top_chunks]
            logger.info(f"Selected {len(result_chunks)} chunks")
            
            # Apply GRAG reranking if enabled
            if self.use_grag and len(result_chunks) > 1:
                result_chunks = self.reranker.rerank(query, result_chunks, top_k)
            
            return result_chunks
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to get relevant chunks: {str(e)}")
            
    async def generate_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        language: str = "vi"
    ) -> str:
        """
        Generate response using relevant chunks.
        
        Args:
            query: User query
            relevant_chunks: Relevant document chunks
            temperature: LLM temperature
            language: Response language
            
        Returns:
            Generated response
            
        Raises:
            QueryProcessingError: If generation fails
        """
        try:
            # Build context from chunks
            context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
            logger.info(f"Context length: {len(context)}")
            logger.info(f"Chunks used: {len(relevant_chunks)}")
            
            # Get system prompt
            system_message = self.llm_service.get_prompt("system", language)
            
            # Build user message
            user_message = f"""Nội dung văn bản:

{context}

Yêu cầu: {query}

Hãy trả lời dựa trên nội dung văn bản được cung cấp. Nếu nội dung không đủ thông tin để trả lời chính xác, hãy nêu rõ điều này.""" if language == "vi" else f"""Document content:

{context}

Question: {query}

Please answer based on the provided document content. If the content lacks sufficient information for an accurate answer, please clearly state this."""
            
            # Generate response
            response = await self.llm_service.generate_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature
            )
            
            return response
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to generate response: {str(e)}")
