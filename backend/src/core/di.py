"""FastAPI dependency injection configuration."""
from typing import AsyncGenerator, Any
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import AsyncSessionLocal
from src.core.llm.interfaces import LLMInterface
from src.repositories.document_repository import DocumentRepository
from src.repositories.query_repository import QueryRepository
from src.repositories.citation_repository import CitationRepository
from src.repositories.context_repository import ContextRepository
from src.services.document_service import DocumentService
from src.core.llm.providers import OpenAILLM

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_llm() -> AsyncGenerator[LLMInterface, None]:
    """Get LLM provider instance."""
    try:
        yield OpenAILLM()
    finally:
        pass

# Repository dependencies
async def get_document_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[DocumentRepository, None]:
    """Get document repository instance."""
    try:
        yield DocumentRepository(session)
    finally:
        pass

async def get_query_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[QueryRepository, None]:
    """Get query repository instance."""
    try:
        yield QueryRepository(session)
    finally:
        pass

async def get_citation_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[CitationRepository, None]:
    """Get citation repository instance."""
    try:
        yield CitationRepository(session)
    finally:
        pass

async def get_context_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[ContextRepository, None]:
    """Get context repository instance."""
    try:
        yield ContextRepository(session)
    finally:
        pass

async def get_document_service(
    session: AsyncSession = Depends(get_session),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    query_repo: QueryRepository = Depends(get_query_repository),
    citation_repo: CitationRepository = Depends(get_citation_repository),
    context_repo: ContextRepository = Depends(get_context_repository),
    llm: LLMInterface = Depends(get_llm)
) -> AsyncGenerator[DocumentService, None]:
    """Get document service instance."""
    try:
        yield DocumentService(
            session=session,
            doc_repo=doc_repo,
            query_repo=query_repo,
            citation_repo=citation_repo,
            context_repo=context_repo,
            llm=llm
        )
    finally:
        pass