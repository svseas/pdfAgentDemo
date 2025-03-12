from typing import List, Dict, Any
import numpy as np
import httpx
import logging
import re
import os
from pathlib import Path
from src.domain.embedding_generator import EmbeddingGenerator
from src.core.config import settings
from src.domain.grag import GRAGService
from src.domain.stepback_agent import StepbackAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose logging

class QueryProcessor:
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.llm_url = f"{settings.LMSTUDIO_BASE_URL}/chat/completions"
        self.grag_service = None  # Lazy initialization
        self.stepback_agent = StepbackAgent()
        logger.info("QueryProcessor initialized")
        
    def _initialize_grag(self):
        """Initialize GRAG service for reranking"""
        try:
            from src.domain.grag.service import GRAGService
            
            # Use local model path
            cwd = os.getcwd()
            model_path = Path(cwd) / "models" / "amrbart"
            logger.info(f"Current working directory: {cwd}")
            logger.info(f"Looking for model at: {model_path}")
            logger.info(f"Model path exists: {model_path.exists()}")
            if not model_path.exists():
                logger.error(f"AMR model not found at {model_path}. Please run python backend/src/scripts/download_amr_model.py first")
                return None
                
            self.grag_service = GRAGService(
                embedding_model_name="BAAI/bge-small-en",  # Same model as our embedding generator
                amr_model_name=str(model_path)  # Convert Path to string
            )
            logger.info("GRAG service initialized successfully")
            return self.grag_service
        except Exception as e:
            logger.error(f"Failed to initialize GRAG service: {e}")
            self.grag_service = None
            return None
        
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
        try:
            # Convert to numpy arrays if they aren't already
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            if not isinstance(doc_embedding, np.ndarray):
                doc_embedding = np.array(doc_embedding, dtype=np.float32)
            
            # Ensure arrays are 1-dimensional and have the same shape
            query_embedding = query_embedding.flatten()
            doc_embedding = doc_embedding.flatten()
            
            # Log shapes for debugging
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            logger.debug(f"Doc embedding shape: {doc_embedding.shape}")
            logger.debug(f"Doc embedding type: {type(doc_embedding)}")
            
            # Check if shapes match
            if query_embedding.shape[0] != doc_embedding.shape[0]:
                logger.error(f"Shape mismatch: query {query_embedding.shape} vs doc {doc_embedding.shape}")
                return 0.0
                
            # Calculate cosine similarity
            query_norm = float(np.linalg.norm(query_embedding))
            doc_norm = float(np.linalg.norm(doc_embedding))
            
            if query_norm < 1e-10 or doc_norm < 1e-10:  # Check for near-zero norms
                return 0.0
                
            cosine_sim = float(np.dot(query_embedding, doc_embedding)) / (query_norm * doc_norm)
            
            # Apply position bias (higher weight for earlier chunks)
            position_weight = 1.0 / (1.0 + 0.1 * chunk_index)  # Decay factor of 0.1
            
            # Apply position bias only
            final_score = cosine_sim * position_weight
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            logger.debug(f"Doc embedding shape: {doc_embedding.shape}")
            logger.debug(f"Doc embedding type: {type(doc_embedding)}")
            return 0.0  # Return 0 similarity on error

    def get_relevant_chunks(
        self, 
        query: str, 
        doc_chunks: List[Dict[str, Any]], 
        top_k: int = None,
        use_grag: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a given query.
        
        Args:
            query: The user's question
            doc_chunks: List of document chunks with their embeddings
            top_k: Number of most relevant chunks to return (defaults to settings.TOP_K_MATCHES)
            use_grag: Whether to use GRAG for reranking (defaults to False). Set to True to enable graph-based reranking.
        
        Returns:
            List of the most relevant document chunks
        """
        if top_k is None:
            top_k = settings.TOP_K_MATCHES

        logger.info(f"Processing query: {query}")
        logger.info(f"Total chunks to process: {len(doc_chunks)}")
        logger.info(f"GRAG reranking enabled: {use_grag}")
            
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Calculate similarities with content boost
        chunk_similarities = []
        for i, chunk in enumerate(doc_chunks):
            try:
                if not chunk.get("embedding"):
                    logger.warning(f"Chunk {i} has no embedding")
                    continue
                
                # Convert embedding to numpy array if it's a list
                doc_embedding = chunk["embedding"]
                if isinstance(doc_embedding, list):
                    doc_embedding = np.array(doc_embedding, dtype=np.float32)
                elif isinstance(doc_embedding, np.ndarray):
                    doc_embedding = doc_embedding.astype(np.float32)
                    
                similarity = self._calculate_similarity(
                    query_embedding,
                    doc_embedding,
                    i,  # Pass chunk index
                    chunk["content"]  # Pass content for boosting
                )
                
                if similarity > 0:  # Only include non-zero similarities
                    chunk_similarities.append((chunk, similarity, i))
                    logger.debug(f"Initial similarity for chunk {i}: {similarity}")
            except Exception as e:
                logger.error(f"Error calculating similarity for chunk {i}: {str(e)}")
                continue
        
        # Sort by similarity score
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k chunks and sort by position
        top_chunks = chunk_similarities[:top_k] if chunk_similarities else []
        top_chunks.sort(key=lambda x: x[2])  # Sort by original position
        
        result_chunks = [chunk for chunk, sim, _ in top_chunks]
        logger.info(f"Selected {len(result_chunks)} chunks after initial ranking")
        logger.info(f"Initial top similarities: {[sim for _, sim, _ in top_chunks[:3]]}")

        # Apply GRAG reranking if enabled
        if use_grag and len(result_chunks) > 1:
            try:
                if self.grag_service is None:
                    logger.info("GRAG service not initialized, initializing now...")
                    self._initialize_grag()
                
                if self.grag_service is not None:
                    logger.info("Starting GRAG reranking...")
                    
                    # Store original order for logging
                    original_order = [chunk["id"] for chunk in result_chunks]
                    
                    result_chunks = self.grag_service.rerank(
                        question=query,
                        documents=result_chunks,
                        top_k=top_k
                    )
                    
                    # Log reranking results
                    new_order = [chunk["id"] for chunk in result_chunks]
                    logger.info(f"GRAG reranking complete. Order changed: {original_order} -> {new_order}")
                else:
                    logger.warning("GRAG service initialization failed, using original ranking")
            except Exception as e:
                logger.error(f"Error during GRAG reranking: {e}")
                logger.info("Falling back to similarity-based ranking")
                # Fall back to similarity-based ranking
                pass
        
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
            "1. Tổ chức câu trả lời theo cấu trúc rõ ràng với các mục và tiêu đề phù hợp\n"
            "2. Sắp xếp thông tin theo trình tự thời gian hoặc logic\n"
            "3. Trích dẫn đầy đủ các sự kiện, ngày tháng và con số quan trọng\n"
            "4. Nêu rõ các mốc lịch sử và sự kiện quan trọng\n"
            "5. Phân tích ý nghĩa và tầm quan trọng của các sự kiện\n"
            "6. Kết luận bằng việc tổng hợp các điểm chính và ý nghĩa lịch sử\n\n"
            "Nếu văn bản không có thông tin cần thiết, hãy nêu rõ điều này."
        )
        
        # Construct user message with context
        user_message = f"""Nội dung văn bản:

{context}

Yêu cầu: {query}

Hãy trả lời dựa trên nội dung văn bản được cung cấp. Nếu nội dung không đủ thông tin để trả lời chính xác, hãy nêu rõ điều này."""

        # Prepare the request payload to match working curl command exactly
        payload = {
            "model": "qwen2.5-7b-instruct-1m",  # Try Qwen model first
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": -1,
            "stream": False
        }
        
        # Log the constructed payload
        logger.debug(f"Constructed payload: {payload}")

        # List of models to try in order
        models = [
            "qwen2.5-7b-instruct-1m",  # Try Qwen first
            "llama3-docchat-1.0-8b-i1",  # Fallback to Llama if Qwen fails
        ]
        
        last_error = None
        for model in models:
            try:
                logger.info(f"Attempting to use model: {model}")
                payload["model"] = model
                
                async with httpx.AsyncClient(timeout=settings.LMSTUDIO_TIMEOUT) as client:
                    # Log request details
                    logger.debug(f"Making request to URL: {self.llm_url}")
                    logger.debug(f"Request payload: {payload}")
                    
                    # Make request
                    response = await client.post(
                        self.llm_url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                    )
                    
                    # Log response details
                    logger.debug(f"Response status: {response.status_code}")
                    logger.debug(f"Response headers: {response.headers}")
                    
                    response.raise_for_status()
                    response_data = response.json()
                    logger.debug(f"Response data: {response_data}")
                    
                    if "choices" not in response_data:
                        raise Exception(f"Unexpected response format: {response_data}")
                    
                    initial_answer = response_data["choices"][0]["message"]["content"]
                    
                    # If we got here, the model worked
                    logger.info(f"Successfully generated response using model: {model}")
                    
                    # Enhance answer using stepback prompting
                    enhanced_answer = await self.stepback_agent.enhance_answer(
                        context=context,
                        query=query,
                        initial_answer=initial_answer
                    )
                    
                    return enhanced_answer
                    
            except Exception as e:
                logger.error(f"Error with model {model}: {str(e)}")
                last_error = e
                continue  # Try next model
        
        # If we get here, all models failed
        logger.error("All models failed to generate response")
        raise Exception(f"Failed to generate response with any model. Last error: {str(last_error)}")
