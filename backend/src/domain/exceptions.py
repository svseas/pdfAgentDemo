"""Custom exceptions for the application."""

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class ContextBuilderError(Exception):
    """Exception raised when context building fails."""
    pass

class ChunkRetrievalError(Exception):
    """Exception raised when chunk retrieval fails."""
    pass

class ChunkingError(Exception):
    """Exception raised when text chunking fails."""
    pass

class LLMError(Exception):
    """Exception raised when LLM operations fail."""
    pass

class ContextStorageError(Exception):
    """Exception raised when context storage operations fail."""
    pass

class RepositoryError(Exception):
    """Base exception for repository-related errors."""
    pass

class DocumentProcessingError(Exception):
    """Exception raised when document processing fails."""
    pass

class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    pass

class QueryProcessingError(Exception):
    """Exception raised when query processing fails."""
    pass

class WorkflowError(Exception):
    """Exception raised when workflow operations fail."""
    pass

class ValidationError(Exception):
    """Exception raised when data validation fails."""
    pass

class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""
    pass

class ServiceError(Exception):
    """Base exception for service-related errors."""
    pass

class IntegrationError(Exception):
    """Exception raised when external service integration fails."""
    pass

class ResourceNotFoundError(Exception):
    """Exception raised when a requested resource is not found."""
    pass

class AuthorizationError(Exception):
    """Exception raised when authorization fails."""
    pass

class RateLimitError(Exception):
    """Exception raised when rate limits are exceeded."""
    pass

class TextProcessingError(Exception):
    """Exception raised when text processing operations fail."""
    pass

class PDFProcessingError(Exception):
    """Exception raised when PDF processing operations fail."""
    pass

# LLM-specific errors
class LLMTimeoutError(LLMError):
    """Exception raised when LLM request times out."""
    pass

class LLMRateLimitError(LLMError):
    """Exception raised when LLM rate limit is exceeded."""
    pass

class LLMContextLengthError(LLMError):
    """Exception raised when input context is too long."""
    pass

class LLMAPIError(LLMError):
    """Exception raised when LLM API returns an error."""
    pass