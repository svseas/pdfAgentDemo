"""Domain interfaces and abstract base classes."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class TextSplitterInterface(ABC):
    """Interface for text splitting strategies."""
    
    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

class EmbeddingGeneratorInterface(ABC):
    """Interface for embedding generation."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass
        
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass

class LLMInterface(ABC):
    """Interface for LLM interactions."""
    
    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Generate completion from LLM."""
        pass
        
    @abstractmethod
    def get_prompt(self, prompt_type: str, language: str = "vi") -> str:
        """Get prompt template."""
        pass

class DocumentProcessorInterface(ABC):
    """Interface for document processing."""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text from document."""
        pass
        
    @abstractmethod
    def process_text(self, text: str) -> List[str]:
        """Process extracted text into chunks."""
        pass

class QueryProcessorInterface(ABC):
    """Interface for query processing."""
    
    @abstractmethod
    def get_relevant_chunks(
        self,
        query: str,
        doc_chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant document chunks for query."""
        pass
        
    @abstractmethod
    async def generate_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        language: str = "vi"
    ) -> str:
        """Generate response from relevant chunks."""
        pass