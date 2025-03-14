"""Domain interfaces."""
from typing import Dict, Any, List, Optional, Protocol
from datetime import datetime
import numpy as np

class BaseRepository(Protocol):
    """Base interface for repositories."""

    async def create(self, data: Dict[str, Any]) -> Any:
        """Create a new record."""
        ...

    async def get_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """Get record by ID."""
        ...

    async def update(self, id: int, data: Dict[str, Any]) -> Any:
        """Update record."""
        ...

    async def delete(self, id: int) -> bool:
        """Delete record."""
        ...

class EmbeddingGeneratorInterface(Protocol):
    """Interface for embedding generation."""

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        ...

    async def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        ...

class DocumentProcessorInterface(Protocol):
    """Interface for document processing."""

    async def process_document(
        self,
        file_path: str,
        document_id: int,
        language: str = "vietnamese"
    ) -> Dict[str, Any]:
        """Process a document file."""
        ...

    async def extract_text(
        self,
        file_path: str,
        language: str = "vietnamese"
    ) -> str:
        """Extract text from a document file."""
        ...

    async def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Split text into chunks."""
        ...

class TextSplitterInterface(Protocol):
    """Interface for text splitting."""

    def split_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """Split text into chunks."""
        ...

    def merge_short_texts(
        self,
        texts: List[str],
        min_length: int = 500
    ) -> List[str]:
        """Merge short text chunks."""
        ...

class LLMInterface(Protocol):
    """Interface for language model operations."""

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using language model."""
        ...

    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat response using language model."""
        ...

class QueryProcessorInterface(Protocol):
    """Interface for query processing."""

    async def process_query(
        self,
        query: str,
        language: str = "vietnamese"
    ) -> Dict[str, Any]:
        """Process a query."""
        ...

    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> str:
        """Generate response using context."""
        ...

    def get_relevant_chunks(
        self,
        query: str,
        doc_chunks: List[Dict[str, Any]],
        top_k: int = 5,
        use_grag: bool = False
    ) -> List[Dict[str, Any]]:
        """Get relevant chunks for query."""
        ...

class DocumentRepository(BaseRepository):
    """Interface for document repository."""

    async def get_document_chunks(
        self,
        document_id: int,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Get document chunks."""
        ...

    async def get_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get similar chunks using vector similarity."""
        ...

    async def get_surrounding_chunks(
        self,
        doc_metadata_id: int,
        chunk_index: int,
        window_size: int = 1
    ) -> List[Dict[str, Any]]:
        """Get surrounding chunks for context."""
        ...

class QueryRepository(BaseRepository):
    """Interface for query repository."""

    async def create_sub_query(
        self,
        original_query_id: int,
        sub_query_text: str,
        sub_query_embedding: Optional[List[float]] = None
    ) -> int:
        """Create a sub-query."""
        ...

    async def get_sub_queries(
        self,
        original_query_id: int
    ) -> List[Dict[str, Any]]:
        """Get all sub-queries for an original query."""
        ...

class CitationRepository(BaseRepository):
    """Interface for citation repository."""

    async def create_citation(
        self,
        document_id: int,
        chunk_id: Optional[int],
        citation_text: str,
        citation_type: str,
        normalized_format: str,
        authority_level: int,
        metadata: Dict[str, Any] = {}
    ) -> int:
        """Create a citation."""
        ...

    async def create_response_citation(
        self,
        citation_id: int,
        context_used_id: int,
        relevance_score: float
    ) -> int:
        """Create a citation usage in response."""
        ...

class ContextRepository(BaseRepository):
    """Interface for context repository."""

    async def create_context_set(
        self,
        original_query_id: int,
        context_data: Dict[str, Any],
        context_metadata: Dict[str, Any]
    ) -> int:
        """Create a context set."""
        ...

    async def create_context_result(
        self,
        document_id: int,
        chunk_id: Optional[int],
        summary_id: Optional[int],
        relevance_score: float,
        used_in_response: bool = False
    ) -> int:
        """Create a context result."""
        ...