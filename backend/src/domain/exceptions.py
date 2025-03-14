"""Domain exceptions."""

class DomainError(Exception):
    """Base class for domain exceptions."""
    pass

class RepositoryError(DomainError):
    """Exception raised when repository operations fail."""
    pass

class AgentError(DomainError):
    """Exception raised when agent operations fail."""
    pass

class ChunkingError(DomainError):
    """Exception raised when document chunking fails."""
    pass

class EmbeddingError(DomainError):
    """Exception raised when embedding generation fails."""
    pass

class ContextBuilderError(DomainError):
    """Exception raised when context building fails."""
    pass

class ChunkRetrievalError(DomainError):
    """Exception raised when chunk retrieval fails."""
    pass

class ContextStorageError(DomainError):
    """Exception raised when context storage fails."""
    pass

class QueryProcessingError(DomainError):
    """Exception raised when query processing fails."""
    pass

class CitationError(DomainError):
    """Exception raised when citation operations fail."""
    pass

class DocumentProcessingError(DomainError):
    """Exception raised when document processing fails."""
    pass

class TextSplittingError(DomainError):
    """Exception raised when text splitting fails."""
    pass

class LLMError(DomainError):
    """Exception raised when LLM operations fail."""
    pass