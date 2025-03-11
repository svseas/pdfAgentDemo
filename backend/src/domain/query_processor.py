from typing import List, Dict, Any
import numpy as np
import httpx
import logging
import re
from src.domain.embedding_generator import EmbeddingGenerator
from src.core.config import settings

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.llm_url = f"{settings.LMSTUDIO_BASE_URL}/chat/completions"
        
    def _calculate_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray, chunk_index: int, chunk_content: str) -> float:
        """
        Calculate similarity score between query and document embeddings.
        Applies position bias and content relevance boost.
        
        Args:
            query_embedding: Query embedding vector
            doc_embedding: Document chunk embedding vector
            chunk_index: Position of the chunk in the document
            chunk_content: Text content of the chunk
            
        Returns:
            Final similarity score
        """
        # Convert to numpy arrays if they aren't already
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(doc_embedding, np.ndarray):
            doc_embedding = np.array(doc_embedding)
        
        # Ensure arrays are 1-dimensional
        query_embedding = query_embedding.flatten()
        doc_embedding = doc_embedding.flatten()
            
        # Calculate cosine similarity
        cosine_sim = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        # Apply position bias (higher weight for earlier chunks)
        position_weight = 1.0 / (1.0 + 0.1 * chunk_index)  # Decay factor of 0.1
        
        # Apply position bias only
        final_score = cosine_sim * position_weight
        
        return float(final_score)  # Convert to Python float

    def get_relevant_chunks(
        self, 
        query: str, 
        doc_chunks: List[Dict[str, Any]], 
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a given query.
        
        Args:
            query: The user's question
            doc_chunks: List of document chunks with their embeddings
            top_k: Number of most relevant chunks to return (defaults to settings.TOP_K_MATCHES)
        
        Returns:
            List of the most relevant document chunks
        """
        if top_k is None:
            top_k = settings.TOP_K_MATCHES

        logger.info(f"Processing query: {query}")
        logger.info(f"Total chunks to process: {len(doc_chunks)}")
            
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Calculate similarities with content boost
        chunk_similarities = []
        for i, chunk in enumerate(doc_chunks):
            try:
                if chunk["embedding"] is None:
                    logger.warning(f"Chunk {i} has no embedding")
                    continue
                    
                similarity = self._calculate_similarity(
                    query_embedding,
                    chunk["embedding"],
                    i,  # Pass chunk index
                    chunk["content"]  # Pass content for boosting
                )
                chunk_similarities.append((chunk, similarity, i))
                logger.info(f"Chunk {i} similarity: {similarity}")
            except Exception as e:
                logger.error(f"Error calculating similarity for chunk {i}: {str(e)}")
                continue
        
        # Sort by similarity score
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k chunks and sort by position
        top_chunks = chunk_similarities[:top_k]
        top_chunks.sort(key=lambda x: x[2])  # Sort by original position
        
        result_chunks = [chunk for chunk, _, _ in top_chunks]
        logger.info(f"Selected {len(result_chunks)} chunks")
        
        return result_chunks

    async def generate_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response using LMStudio based on the query and relevant document chunks.
        
        Args:
            query: The user's question
            relevant_chunks: List of relevant document chunks
            temperature: Temperature parameter for response generation
            
        Returns:
            Generated response from the LLM
        """
        # Construct context from relevant chunks
        context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        logger.info(f"Generated context length: {len(context)}")
        logger.info(f"Number of chunks used: {len(relevant_chunks)}")
        
        # Construct system message
        system_message = (
            "Bạn là một trợ lý AI chuyên nghiệp, giúp người dùng hiểu nội dung văn bản. "
            "Khi trả lời câu hỏi, hãy:\n"
            "1. Tập trung vào thông tin được hỏi\n"
            "2. Trích dẫn các con số cụ thể nếu có\n"
            "3. Liệt kê đầy đủ các đối tượng được đề cập\n"
            "4. Sắp xếp thông tin một cách logic\n"
            "5. Sử dụng ngôn ngữ rõ ràng, chính xác\n\n"
            "Nếu văn bản không có thông tin cần thiết, hãy nêu rõ điều này."
        )
        
        # Construct user message with context
        user_message = f"""Nội dung văn bản:

{context}

Yêu cầu: {query}

Hãy trả lời dựa trên nội dung văn bản được cung cấp. Nếu nội dung không đủ thông tin để trả lời chính xác, hãy nêu rõ điều này."""

        # Prepare the request payload
        payload = {
            "model": settings.LMSTUDIO_MODEL,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": -1,
            "stream": False
        }

        # Make request to LMStudio
        async with httpx.AsyncClient(timeout=settings.LMSTUDIO_TIMEOUT) as client:
            try:
                response = await client.post(
                    self.llm_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except httpx.RequestError as e:
                logger.error(f"Error calling LMStudio API: {str(e)}")
                raise Exception(f"Error generating response from LLM: {str(e)}")
