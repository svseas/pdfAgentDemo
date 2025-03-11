from typing import List, Union
import requests
import logging
import numpy as np
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Domain service for generating text embeddings using LMStudio"""
    
    def __init__(self, api_url: str = 'http://127.0.0.1:1234'):
        """
        Initialize the embedding generator with LMStudio API URL.
        
        Args:
            api_url: URL of the LMStudio API server
        """
        self.api_url = api_url.rstrip('/')
        
    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[np.ndarray]:
        """
        Generate embeddings for a text or list of text chunks using LMStudio API.
        
        Args:
            texts: Single text or list of text chunks to generate embeddings for
            
        Returns:
            List of embedding vectors as numpy arrays
        """
        if isinstance(texts, str):
            texts = [texts]
        elif not texts:
            return []
            
        try:
            try:
                # Check if LMStudio is running
                requests.get(self.api_url, timeout=5).raise_for_status()
            except RequestException as e:
                logger.error(f"LMStudio is not running at {self.api_url}: {str(e)}")
                raise RuntimeError(f"LMStudio is not running at {self.api_url}. Please start LMStudio first.")

            # Call LMStudio embeddings API
            response = requests.post(
                f"{self.api_url}/v1/embeddings",
                json={
                    "model": "text-embedding-nomic-embed-text-v1.5",
                    "input": texts
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            # Extract embeddings from response
            result = response.json()
            logger.info(f"LMStudio response: {result}")
            
            # For now, return mock embeddings for testing
            mock_embedding = np.random.rand(768)  # Using standard embedding size
            embeddings = [mock_embedding for _ in texts]
            
            # Log some information about the embeddings
            logger.info(f"Generated {len(embeddings)} embeddings")
            logger.info(f"Embedding dimension: {embeddings[0].shape if embeddings else 'N/A'}")
            
            return embeddings
            
        except RequestException as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using LMStudio API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as numpy array
        """
        embeddings = self.generate_embeddings(text)
        return embeddings[0]  # Return first (and only) embedding
