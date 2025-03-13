"""FastAPI dependencies."""
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.config import settings
from src.core.database import AsyncSessionLocal
from src.core.llm_service import LLMService
from src.core.llm.interfaces import PromptTemplateInterface
from src.domain.embedding_generator import EmbeddingGenerator
from src.domain.pdf_processor import PDFProcessor
from src.domain.query_processor import QueryProcessor
from src.repositories.document_repository import DocumentRepository
from src.services.document_service import DocumentService
from src.domain.agents.recursive_summarization_agent import RecursiveSummarizationAgent
from src.domain.agents.query_analyzer_agent import QueryAnalyzerAgent
from src.domain.agents.citation_agent import CitationAgent
from src.domain.agents.query_synthesizer_agent import QuerySynthesizerAgent

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

def get_embedding_generator() -> EmbeddingGenerator:
    """Get embedding generator instance."""
    return EmbeddingGenerator()

def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    return LLMService()

def get_pdf_processor() -> PDFProcessor:
    """Get PDF processor instance."""
    return PDFProcessor(
        chunking_method=settings.CHUNKING_METHOD,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        language=settings.AGENTIC_CHUNKING_LANGUAGE
    )

def get_query_processor(
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    llm_service: LLMService = Depends(get_llm_service)
) -> QueryProcessor:
    """Get query processor instance."""
    return QueryProcessor(
        embedding_generator=embedding_generator,
        llm_service=llm_service
    )

async def get_document_repository(
    db: AsyncSession = Depends(get_db)
) -> DocumentRepository:
    """Get document repository instance."""
    return DocumentRepository(db)

async def get_document_service(
    document_repository: DocumentRepository = Depends(get_document_repository),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    query_processor: QueryProcessor = Depends(get_query_processor)
) -> DocumentService:
    """Get document service instance."""
    return DocumentService(
        document_repository=document_repository,
        pdf_processor=pdf_processor,
        embedding_generator=embedding_generator,
        query_processor=query_processor
    )

def get_prompt_manager() -> PromptTemplateInterface:
    """Get prompt manager instance."""
    from src.core.llm.prompts import PromptManager
    return PromptManager()

async def get_summarization_agent(
    session: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    prompt_manager: PromptTemplateInterface = Depends(get_prompt_manager),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
) -> RecursiveSummarizationAgent:
    """Get summarization agent instance."""
    return RecursiveSummarizationAgent(
        session=session,
        pdf_processor=pdf_processor,
        embedding_generator=embedding_generator,
        llm=llm_service,
        prompt_manager=prompt_manager
    )

async def get_query_analyzer_agent(
    session: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
) -> QueryAnalyzerAgent:
    """Get query analyzer agent instance."""
    return QueryAnalyzerAgent(
        session=session,
        llm=llm_service
    )

async def get_citation_agent(
    session: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
) -> CitationAgent:
    """Get citation agent instance."""
    return CitationAgent(
        session=session,
        llm=llm_service
    )

async def get_query_synthesizer_agent(
    session: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
) -> QuerySynthesizerAgent:
    """Get query synthesizer agent instance."""
    return QuerySynthesizerAgent(
        session=session,
        llm=llm_service
    )
