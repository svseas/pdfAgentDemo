import json
import random
import requests
from typing import List, Dict, Any
import re
import time
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import logging

logger = logging.getLogger(__name__)

class AgenticChunker:
    def __init__(
        self,
        llm_endpoint: str = "http://localhost:1234/v1",  # Base URL
        model: str = "llama3-docchat-1.0-8b-i1",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        language: str = "vietnamese",
        request_timeout: int = 60,  # Increased timeout
        health_check_timeout: int = 10,  # Separate timeout for health check
        max_retries: int = 3,  # Number of retries for failed requests
        batch_size: int = 2,  # Smaller batch size for better reliability
        cooldown_time: int = 3,  # Increased cooldown time between requests
    ):
        """
        Initialize the agentic chunker for legal documents.
        
        Args:
            llm_endpoint: API endpoint for the LLM (base URL)
            model: Model identifier
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            language: Document language (default: vietnamese)
            request_timeout: Timeout for API requests in seconds
        """
        self.llm_endpoint = llm_endpoint.rstrip('/')  # Remove trailing slash if present
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.language = language
        self.request_timeout = request_timeout
        self.health_check_timeout = health_check_timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.cooldown_time = cooldown_time
        
        # Initialize document structure guidance based on language
        self.document_structure = {
            "vietnamese": {
                "section_markers": [
                    "Điều", "Chương", "Mục", "Phần", "Khoản",
                    "ĐIỀU", "CHƯƠNG", "MỤC", "PHẦN", "KHOẢN"
                ],
                "legal_terms": [
                    "luật", "nghị định", "thông tư", "quyết định", "văn bản", 
                    "hợp đồng", "quy định", "điều khoản", "nghĩa vụ", "quyền"
                ]
            },
            "english": {
                "section_markers": [
                    "Article", "Section", "Chapter", "Part", "Paragraph",
                    "ARTICLE", "SECTION", "CHAPTER", "PART", "PARAGRAPH"
                ],
                "legal_terms": [
                    "law", "decree", "circular", "decision", "document", 
                    "contract", "regulation", "term", "obligation", "right"
                ]
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract full text from a PDF file."""
        elements = partition_pdf(filename=pdf_path)
        return " ".join(str(element) for element in elements if str(element).strip())
        
    def call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the LLM API with the given messages with retry mechanism."""
        def check_llm_health():
            try:
                response = requests.get(self.llm_endpoint, timeout=self.health_check_timeout)
                if response.status_code != 200:
                    logger.warning(f"LLM health check failed (status {response.status_code})")
                    return False
                return True
            except requests.RequestException as e:
                logger.warning(f"LLM health check failed: {str(e)}")
                return False

        if not check_llm_health():
            logger.error("LLM is not available")
            raise RuntimeError("LLM is not available")

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
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt + random.uniform(0, 1)) * self.cooldown_time
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)

                logger.info(f"Sending request to LLM (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    f"{self.llm_endpoint}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.request_timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info("LLM request successful")
                    return result

                # Specific error handling based on status code
                if response.status_code in [502, 503, 504]:  # Gateway/Service Unavailable
                    logger.warning(f"LLM service temporarily unavailable (status {response.status_code})")
                    last_error = f"Service temporarily unavailable (status {response.status_code})"
                    continue
                elif response.status_code == 429:  # Too Many Requests
                    logger.warning("Rate limit exceeded, backing off")
                    last_error = "Rate limit exceeded"
                    continue
                else:
                    logger.error(f"LLM API error (status {response.status_code}): {response.text}")
                    raise RuntimeError(f"LLM API error (status {response.status_code}): {response.text}")

            except requests.Timeout:
                last_error = f"Request timed out after {self.request_timeout}s"
                logger.warning(f"Attempt {attempt + 1} timed out")
                continue
            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                continue

        # If we get here, all retries failed
        logger.error(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        raise RuntimeError(f"Failed to get LLM response after {self.max_retries} attempts: {last_error}")
    
    def chunk_text_with_semantic_fallback(self, text: str) -> List[str]:
        """Fallback to semantic chunking when LLM is not available."""
        from src.domain.semantic_text_splitter import TextSplitter
        
        logger.info("Using semantic chunking fallback")
        splitter = TextSplitter(
            max_characters=2500,  # Increased for larger chunks
            semantic_units=["paragraph", "sentence"],
            break_mode="paragraph",  # Prefer paragraph breaks
            flex=0.4  # More flexibility (0.4 means chunks can be 60%-140% of max_characters)
        )
        
        chunks = splitter.chunks(text)
        logger.info(f"Created {len(chunks)} chunks using semantic fallback")
        return chunks
    
    def process_text(self, text: str) -> List[str]:
        """
        Process text into chunks using LLM.
        
        Args:
            text: The text to process
            
        Returns:
            List of text chunks
        """
        try:
            # Try semantic chunking first since it's faster and more reliable
            logger.info("Starting with semantic chunking")
            chunks = self.chunk_text_with_semantic_fallback(text)
            
            # Then try to improve chunks with LLM if available
            try:
                logger.info("Attempting to improve chunks with LLM")
                improved_chunks = []
                
                # Process chunks in configured batch size
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
                        
                        response = self.call_llm(messages)
                        improved_text = response["choices"][0]["message"]["content"]
                        
                        # Extract chunks and apply validation
                        raw_chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', improved_text) if chunk.strip()]
                        
                        # Filter out metadata and validate chunks
                        improved_batch = []
                        for chunk in raw_chunks:
                            # Skip chunks that look like metadata or translations
                            if any(marker in chunk.lower() for marker in [
                                "here are", "improved chunks", "analysis", "translation",
                                "processed text", "output", "result"
                            ]):
                                continue
                                
                            # Count sentences in chunk
                            sentence_count = len(re.split(r'[.!?]+', chunk))
                            if sentence_count >= 2:  # Require at least 2 sentences
                                improved_batch.append(chunk)
                            else:
                                logger.warning(f"Skipping small chunk with {sentence_count} sentences")
                        improved_chunks.extend(improved_batch)
                        
                        # Rate limiting with configured cooldown
                        time.sleep(self.cooldown_time)
                        
                        logger.info(f"Successfully processed batch {i//self.batch_size + 1}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process batch {i//self.batch_size + 1}: {str(e)}")
                        # On batch failure, keep original chunks for this batch
                        improved_chunks.extend(batch)
                        continue
                
                if improved_chunks:
                    logger.info(f"Successfully improved chunks with LLM: {len(improved_chunks)} chunks")
                    return improved_chunks
                else:
                    logger.warning("LLM improvement yielded no chunks, using original semantic chunks")
                    return chunks
                    
            except Exception as e:
                logger.error(f"LLM improvement failed: {str(e)}, using original semantic chunks")
                return chunks
                
        except Exception as e:
            logger.error(f"All chunking methods failed: {str(e)}")
            raise RuntimeError(f"Failed to chunk text: {str(e)}")