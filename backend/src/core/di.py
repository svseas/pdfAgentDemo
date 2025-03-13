"""Dependency injection container configuration."""
from typing import AsyncGenerator, Callable, Any
from functools import lru_cache
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.core.database import get_session
from backend.src.core.config import settings
from backend.src.core.llm.interfaces import LLMInterface, PromptTemplateInterface
from backend.src.core.llm.providers import LLMFactory
from backend.src.core.llm.prompts import PromptManager
from backend.src.domain.interfaces import (
    WorkflowRepository,
    QueryRepository,
    AgentStepRepository,
    ContextRepository,
    CitationRepository,
    AgentInterface
)
from backend.src.repositories.workflow_repository import (
    SQLWorkflowRepository,
    SQLQueryRepository,
    SQLAgentStepRepository,
    SQLContextRepository,
    SQLCitationRepository
)
from backend.src.domain.workflow import WorkflowOrchestrator

# Repository dependencies
async def get_workflow_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[WorkflowRepository, None]:
    """Get workflow repository instance."""
    try:
        yield SQLWorkflowRepository(session)
    finally:
        await session.close()

async def get_query_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[QueryRepository, None]:
    """Get query repository instance."""
    try:
        yield SQLQueryRepository(session)
    finally:
        await session.close()

async def get_agent_step_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[AgentStepRepository, None]:
    """Get agent step repository instance."""
    try:
        yield SQLAgentStepRepository(session)
    finally:
        await session.close()

async def get_context_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[ContextRepository, None]:
    """Get context repository instance."""
    try:
        yield SQLContextRepository(session)
    finally:
        await session.close()

async def get_citation_repository(
    session: AsyncSession = Depends(get_session)
) -> AsyncGenerator[CitationRepository, None]:
    """Get citation repository instance."""
    try:
        yield SQLCitationRepository(session)
    finally:
        await session.close()

# Agent factory functions
@lru_cache()
def get_agent_factory() -> Callable[[str, AsyncSession], AgentInterface]:
    """Get factory function for creating agent instances."""
    from backend.src.domain.agents import (
        RecursiveSummarizationAgent,
        QueryAnalyzerAgent,
        ContextBuilderAgent,
        CitationAgent,
        QuerySynthesizerAgent
    )

    agent_types = {
        "summarizer": RecursiveSummarizationAgent,
        "query_analyzer": QueryAnalyzerAgent,
        "context_builder": ContextBuilderAgent,
        "citation": CitationAgent,
        "synthesizer": QuerySynthesizerAgent
    }

    def create_agent(agent_type: str, session: AsyncSession) -> AgentInterface:
        """Create an agent instance of the specified type."""
        if agent_type not in agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = agent_types[agent_type]
        llm = get_llm()
        prompt_manager = get_prompt_manager()
        
        if agent_type == "summarizer":
            from backend.src.domain.pdf_processor import PDFProcessor
            pdf_processor = PDFProcessor(session)
            return agent_class(session, pdf_processor, llm=llm, prompt_manager=prompt_manager)
        
        return agent_class(session, llm=llm, prompt_manager=prompt_manager)

    return create_agent

# Workflow orchestrator dependency
async def get_workflow_orchestrator(
    session: AsyncSession = Depends(get_session),
    agent_factory: Callable = Depends(get_agent_factory)
) -> AsyncGenerator[Any, None]:
    """Get workflow orchestrator instance."""
    try:
        yield WorkflowOrchestrator(session, agent_factory)
    finally:
        await session.close()

# LLM dependencies
@lru_cache()
def get_llm() -> LLMInterface:
    """Get LLM instance."""
    return LLMFactory.create_llm(settings)

@lru_cache()
def get_prompt_manager() -> PromptTemplateInterface:
    """Get prompt manager instance."""
    return PromptManager()

# Service dependencies
async def get_document_service(
    workflow_repo: WorkflowRepository = Depends(get_workflow_repository),
    query_repo: QueryRepository = Depends(get_query_repository),
    context_repo: ContextRepository = Depends(get_context_repository),
    citation_repo: CitationRepository = Depends(get_citation_repository),
    orchestrator: Any = Depends(get_workflow_orchestrator),
    llm: LLMInterface = Depends(get_llm),
    prompt_manager: PromptTemplateInterface = Depends(get_prompt_manager)
) -> AsyncGenerator[Any, None]:
    """Get document service instance."""
    from backend.src.services.document_service import DocumentService
    try:
        yield DocumentService(
            workflow_repo=workflow_repo,
            query_repo=query_repo,
            context_repo=context_repo,
            citation_repo=citation_repo,
            orchestrator=orchestrator,
            llm=llm,
            prompt_manager=prompt_manager
        )
    finally:
        pass  # Cleanup handled by individual repository dependencies