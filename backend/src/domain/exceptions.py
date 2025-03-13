"""Custom domain exceptions."""

class DomainError(Exception):
    """Base class for domain exceptions."""
    pass

class TextProcessingError(DomainError):
    """Raised when text processing fails."""
    pass

class DocumentProcessingError(DomainError):
    """Raised when document processing fails."""
    pass

class EmbeddingError(DomainError):
    """Raised when embedding generation fails."""
    pass

class LLMError(DomainError):
    """Raised when LLM interaction fails."""
    pass

class QueryProcessingError(DomainError):
    """Raised when query processing fails."""
    pass

class ChunkingError(DomainError):
    """Raised when text chunking fails."""
    pass

class ValidationError(DomainError):
    """Raised when validation fails."""
    pass