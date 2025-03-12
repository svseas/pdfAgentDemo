from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.embedding_generator import EmbeddingGenerator
from src.domain.query_processor import QueryProcessor
from src.domain.pdf_processor import PDFProcessor
from src.repositories.document_repository import DocumentRepository
from src.services.document_service import DocumentService
from src.core.config import settings
from src.core.database import get_db

def get_embedding_generator() -> EmbeddingGenerator:
    """Dependency for getting an EmbeddingGenerator instance"""
    return EmbeddingGenerator(settings.LMSTUDIO_BASE_URL)

def get_query_processor(
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
) -> QueryProcessor:
    """Dependency for getting a QueryProcessor instance"""
    return QueryProcessor(embedding_generator)

async def get_document_repository(
    session: AsyncSession = Depends(get_db)
) -> DocumentRepository:
    """Dependency for getting a DocumentRepository instance"""
    return DocumentRepository(session)

def get_document_service(
    repository: DocumentRepository = Depends(get_document_repository),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    query_processor: QueryProcessor = Depends(get_query_processor)
) -> DocumentService:
    """Dependency for getting a DocumentService instance"""
    pdf_processor = PDFProcessor()
    return DocumentService(
        document_repository=repository,
        pdf_processor=pdf_processor,
        embedding_generator=embedding_generator,
        query_processor=query_processor
    )
