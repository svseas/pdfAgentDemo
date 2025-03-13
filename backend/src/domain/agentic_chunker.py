"""Agentic chunking implementation."""
import json
import random
import requests
import logging
import time
from typing import List, Dict, Any
from src.domain.interfaces import TextSplitterInterface, LLMInterface
from src.domain.exceptions import ChunkingError, LLMError
from src.domain.semantic_text_splitter import SemanticTextSplitter

logger = logging.getLogger(__name__)

class LLMHealthCheck:
    """LLM health check functionality."""
    
    def __init__(self, endpoint: str, timeout: int = 10):
        self.endpoint = endpoint
        self.timeout = timeout
        
    def check(self) -> bool:
        """Check LLM service health."""
        try:
            response = requests.get(self.endpoint, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"LLM health check failed (status {response.status_code})")
                return False
            return True
        except requests.RequestException as e:
            logger.warning(f"LLM health check failed: {str(e)}")
            return False

class LLMClient:
    """Client for LLM API interactions."""
    
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        request_timeout: int = 60,
        max_retries: int = 3,
        cooldown_time: int = 3
    ):
        self.endpoint = endpoint.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.cooldown_time = cooldown_time
        self.health_check = LLMHealthCheck(endpoint)
        
    def call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call LLM API with retry mechanism."""
        if not self.health_check.check():
            raise LLMError("LLM is not available")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt + random.uniform(0, 1)) * self.cooldown_time
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)

                logger.info(f"Sending request to LLM (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.request_timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info("LLM request successful")
                    return result

                if response.status_code in [502, 503, 504]:
                    logger.warning(f"LLM service temporarily unavailable (status {response.status_code})")
                    last_error = f"Service temporarily unavailable (status {response.status_code})"
                    continue
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded, backing off")
                    last_error = "Rate limit exceeded"
                    continue
                else:
                    raise LLMError(f"LLM API error (status {response.status_code}): {response.text}")

            except requests.Timeout:
                last_error = f"Request timed out after {self.request_timeout}s"
                logger.warning(f"Attempt {attempt + 1} timed out")
                continue
            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                continue

        raise LLMError(f"Failed to get LLM response after {self.max_retries} attempts: {last_error}")

class AgenticChunker(TextSplitterInterface):
    """Intelligent text chunking using LLM."""
    
    def __init__(
        self,
        llm_endpoint: str = "http://localhost:1234/v1",
        model: str = "llama3-docchat-1.0-8b-i1",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        language: str = "vietnamese",
        batch_size: int = 2
    ):
        """Initialize agentic chunker."""
        self.language = language
        self.batch_size = batch_size
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            endpoint=llm_endpoint,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize fallback splitter
        self.fallback_splitter = SemanticTextSplitter(
            max_characters=2500,
            semantic_units=["paragraph", "sentence"],
            break_mode="paragraph",
            flex=0.4
        )
        
    def _improve_chunks_with_llm(self, chunks: List[str]) -> List[str]:
        """Improve chunks using LLM."""
        improved_chunks = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_text = "\n\n".join(batch)
            
            try:
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are a legal document analyzer specializing in {self.language} documents. Your task is to:
1. Maintain the original language - DO NOT translate
2. Combine related sentences into meaningful chunks (minimum 3-4 sentences per chunk)
3. Preserve legal context and relationships between ideas
4. Return ONLY the improved chunks without any metadata or explanatory text
5. Separate chunks with double newlines"""
                    },
                    {
                        "role": "user",
                        "content": f"Improve these text chunks by combining related content while maintaining the original {self.language} language:\n\n{batch_text}"
                    }
                ]
                
                response = self.llm_client.call(messages)
                improved_text = response["choices"][0]["message"]["content"]
                
                # Process and validate chunks
                raw_chunks = [chunk.strip() for chunk in improved_text.split("\n\n") if chunk.strip()]
                
                for chunk in raw_chunks:
                    # Skip metadata-like chunks
                    if any(marker in chunk.lower() for marker in [
                        "here are", "improved chunks", "analysis", "translation",
                        "processed text", "output", "result"
                    ]):
                        continue
                        
                    # Validate chunk size
                    sentence_count = len(chunk.split('.'))
                    if sentence_count >= 2:
                        improved_chunks.append(chunk)
                    else:
                        logger.warning(f"Skipping small chunk with {sentence_count} sentences")
                
                time.sleep(self.llm_client.cooldown_time)
                logger.info(f"Successfully processed batch {i//self.batch_size + 1}")
                
            except Exception as e:
                logger.warning(f"Failed to process batch {i//self.batch_size + 1}: {str(e)}")
                improved_chunks.extend(batch)
                continue
                
        return improved_chunks if improved_chunks else chunks

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks using LLM-based improvement.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
            
        Raises:
            ChunkingError: If chunking fails
        """
        try:
            # Start with semantic chunking
            logger.info("Starting with semantic chunking")
            chunks = self.fallback_splitter.split(text)
            
            # Try to improve with LLM
            try:
                logger.info("Attempting to improve chunks with LLM")
                improved_chunks = self._improve_chunks_with_llm(chunks)
                
                if improved_chunks:
                    logger.info(f"Successfully improved chunks with LLM: {len(improved_chunks)} chunks")
                    return improved_chunks
                else:
                    logger.warning("LLM improvement yielded no chunks, using original semantic chunks")
                    return chunks
                    
            except Exception as e:
                logger.error(f"LLM improvement failed: {str(e)}")
                return chunks
                
        except Exception as e:
            raise ChunkingError(f"Failed to chunk text: {str(e)}")