"""Document API routes."""
from fastapi import APIRouter

from src.api.v1.routes.document_upload import router as upload_router
from src.api.v1.routes.document_query import router as query_router
from src.api.v1.routes.document_context import router as context_router
from src.api.v1.routes.document_summary import router as summary_router
from src.api.v1.routes.document_citation import router as citation_router
from src.api.v1.routes.query_analyzer import router as analyzer_router

# Create main router
router = APIRouter()

# Include all document-related routers
router.include_router(upload_router)
router.include_router(query_router)
router.include_router(context_router)
router.include_router(summary_router)
router.include_router(citation_router)
router.include_router(analyzer_router)
