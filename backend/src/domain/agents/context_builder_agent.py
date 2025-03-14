"""Context building and retrieval agent."""
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.workflow_repository import (
    AgentStepRepository,
    ContextRepository
)
from src.domain.query_processor import QueryProcessor
from src.domain.exceptions import AgentError
from src.domain.agents.base_agent import BaseAgent

class ContextBuilderAgent(BaseAgent):
    """Agent that builds context for queries using vector search.
    
    This agent is responsible for:
    - Processing queries to find relevant context
    - Searching document chunks and summaries
    - Storing and tracking context results
    
    Attributes:
        query_processor: Processor for finding relevant chunks
        context_repo: Repository for context operations
    """
    
    def __init__(
        self,
        session: AsyncSession,
        agent_step_repo: AgentStepRepository,
        context_repo: ContextRepository,
        query_processor: QueryProcessor,
        *args,
        **kwargs
    ):
        """Initialize context builder agent.
        
        Args:
            session: Database session
            agent_step_repo: Repository for agent step logging
            context_repo: Repository for context operations
            query_processor: Processor for finding relevant chunks
            *args, **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(session, agent_step_repo, *args, **kwargs)
        self.query_processor = query_processor
        self.context_repo = context_repo

    async def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of context building logic.
        
        Args:
            input_data: Must contain:
                - workflow_run_id: ID of current workflow run
                - sub_query_id: Optional ID of sub-query
                - query_text: Text of query to process
                - top_k: Optional number of results to return (default 5)
                
        Returns:
            Dict containing:
                - query: Original query text
                - context: List of relevant context chunks
                - context_ids: List of stored context result IDs
                
        Raises:
            AgentError: If context building fails
        """
        try:
            query_text = input_data.get("query_text")
            if not query_text:
                raise AgentError("No query text provided")
                
            agent_step_id = input_data.get("agent_step_id")
            if not agent_step_id:
                raise AgentError("No agent step ID provided")

            # Get relevant chunks using query processor
            context_results = await self._get_relevant_chunks(
                query_text,
                input_data.get("top_k", 5)
            )
            
            # Store context results
            context_ids = await self._store_context_results(
                agent_step_id,
                context_results
            )
            
            return {
                "query": query_text,
                "context": context_results,
                "context_ids": context_ids
            }
            
        except Exception as e:
            raise AgentError(f"Context building failed: {str(e)}") from e

    async def _get_relevant_chunks(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Get relevant chunks for query.
        
        Args:
            query: Query text to process
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
            
        Raises:
            AgentError: If chunk retrieval fails
        """
        try:
            chunks = await self.context_repo.get_all_chunks()
            return await self.query_processor.get_relevant_chunks(
                query=query,
                doc_chunks=chunks,
                top_k=top_k
            )
        except Exception as e:
            raise AgentError(f"Failed to get relevant chunks: {str(e)}") from e

    async def _store_context_results(
        self,
        agent_step_id: int,
        context_results: List[Dict[str, Any]]
    ) -> List[int]:
        """Store context results in database.
        
        Args:
            agent_step_id: ID of current agent step
            context_results: List of context chunks to store
            
        Returns:
            List of created context result IDs
            
        Raises:
            AgentError: If storing results fails
        """
        try:
            context_ids = []
            for result in context_results:
                context_id = await self.context_repo.create_context_result(
                    agent_step_id=agent_step_id,
                    document_id=result["document_id"],
                    chunk_id=result.get("chunk_id"),
                    summary_id=result.get("summary_id"),
                    relevance_score=result.get("relevance", 0.8)
                )
                context_ids.append(context_id)
            return context_ids
            
        except Exception as e:
            raise AgentError(f"Failed to store context results: {str(e)}") from e