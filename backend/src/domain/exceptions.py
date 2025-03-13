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

class AgentError(DomainError):
    """Raised when agent processing fails.
    
    This exception is used by all agents in the system to indicate failures
    during their processing steps. It can wrap other exceptions to provide
    more context about where and why the failure occurred.
    
    Examples:
        >>> try:
        ...     result = await agent.process(data)
        ... except AgentError as e:
        ...     logger.error(f"Agent processing failed: {e}")
    """
    pass