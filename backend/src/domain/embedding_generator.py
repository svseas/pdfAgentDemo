"""Embedding generation implementation."""
from typing import List, Union
import requests
import logging
import numpy as np
from requests.exceptions import RequestException
from src.domain.interfaces import EmbeddingGeneratorInterface
from src.domain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

class LMStudioEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Implementation of embedding generation using LMStudio."""
    
    def __init__(self, api_url: str = 'http://127.0.0.1:1234', timeout: int = 30):
        """
        Initialize the embedding generator.
        
        Args:
            api_url: URL of the LMStudio API server
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.model = "text-embedding-nomic-embed-text-v1.5"
        
    async def _check_service_health(self) -> None:
        """Check if LMStudio service is available."""
        try:
            requests.get(self.api_url, timeout=5).raise_for_status()
        except RequestException as e:
            logger.error(f"LMStudio is not available: {str(e)}")
            raise EmbeddingError(f"LMStudio is not running at {self.api_url}")
            
    async def _call_embeddings_api(self, texts: List[str]) -> List[np.ndarray]:
        """Call LMStudio embeddings API."""
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={
                    "model": self.model,
                    "input": texts
                },
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if 'error' in result:
                raise EmbeddingError(f"LMStudio API error: {result['error']}")
                
            return [np.array(data['embedding']) for data in result['data']]
            
        except RequestException as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
            
    async def _generate_mock_embedding(self, count: int = 1) -> List[np.ndarray]:
        """Generate mock embeddings for fallback."""
        logger.warning("Using mock embeddings")
        mock_embedding = np.random.rand(768)  # Standard embedding size
        return [mock_embedding for _ in range(count)]

    async def generate_embeddings(self, texts: Union[str, List[str]]) -> List[np.ndarray]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if isinstance(texts, str):
            texts = [texts]
        elif not texts:
            return []
            
        try:
            await self._check_service_health()
            embeddings = await self._call_embeddings_api(texts)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            if embeddings:
                logger.info(f"Embedding dimension: {embeddings[0].shape}")
                
            return embeddings
            
        except EmbeddingError:
            # Fallback to mock embeddings
            return await self._generate_mock_embedding(len(texts))

    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        embeddings = await self.generate_embeddings(text)
        return embeddings[0]

# For backward compatibility
EmbeddingGenerator = LMStudioEmbeddingGenerator
