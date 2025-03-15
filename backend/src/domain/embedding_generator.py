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
            requests.get(f"{self.api_url}/v1/models", timeout=5).raise_for_status()
        except RequestException as e:
            logger.error(f"LMStudio is not available: {str(e)}")
            raise EmbeddingError(f"LMStudio is not running at {self.api_url}")
            
    async def _call_embeddings_api(self, texts: List[str]) -> List[np.ndarray]:
        """Call LMStudio embeddings API."""
        try:
            response = requests.post(
                f"{self.api_url}/v1/embeddings",
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
                
            if not isinstance(result, dict) or 'data' not in result:
                raise EmbeddingError("Invalid response format from LMStudio")
                
            embeddings = []
            for data in result['data']:
                if not isinstance(data, dict) or 'embedding' not in data:
                    raise EmbeddingError("Invalid embedding data from LMStudio")
                embeddings.append(np.array(data['embedding']))
                
            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Expected {len(texts)} embeddings but got {len(embeddings)}"
                )
                
            return embeddings
            
        except RequestException as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
            
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
            
        # Always check if LMStudio is available
        await self._check_service_health()
        embeddings = await self._call_embeddings_api(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        if embeddings:
            logger.info(f"Embedding dimension: {embeddings[0].shape}")
            
        return embeddings

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
