"""Document API route modules.

This package contains the following route modules:

- document_upload: File upload and processing routes
- document_query: Document querying and search routes
- document_context: Context building and management routes
- document_summary: Document summarization routes
- document_citation: Citation extraction and synthesis routes
- query_synthesizer: Answer synthesis routes

Each module is focused on a specific aspect of document handling and follows
consistent patterns for error handling, workflow tracking, and response formatting.
"""

from .document_upload import router as upload_router
from .document_query import router as query_router
from .document_context import router as context_router
from .document_summary import router as summary_router
from .document_citation import router as citation_router
from .query_synthesizer import router as synthesizer_router

__all__ = [
    "upload_router",
    "query_router",
    "context_router",
    "summary_router",
    "citation_router",
    "synthesizer_router"
]