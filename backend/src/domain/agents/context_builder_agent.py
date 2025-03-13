"""Context building and retrieval agent."""
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.workflow_repository import SQLContextRepository
from src.domain.query_processor import QueryProcessor
from .base_agent import BaseAgent

class ContextBuilderAgent(BaseAgent):
    """Agent that builds context for queries using vector search."""
    
    def __init__(
        self,
        session: AsyncSession,
        query_processor: QueryProcessor,
        *args,
        **kwargs
    ):
        super().__init__(session, *args, **kwargs)
        self.query_processor = query_processor
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return output."""
        workflow_run_id = input_data.get("workflow_run_id")
        sub_query_id = input_data.get("sub_query_id")
        query_text = input_data.get("query_text")
        
        # Log step start
        agent_step_id = await self.log_step(
            workflow_run_id,
            sub_query_id,
            input_data,
            {},
            "running"
        )
        
        try:
            context_repo = SQLContextRepository(self.session)
            
            # Search summaries and chunks using query processor
            context_results = await self.query_processor.get_relevant_chunks(
                query=query_text,
                doc_chunks=await context_repo.get_all_chunks(),
                top_k=input_data.get("top_k", 5)
            )
            
            # Store context results
            context_ids = []
            for result in context_results:
                context_id = await context_repo.create_context_result(
                    agent_step_id=agent_step_id,
                    document_id=result["document_id"],
                    chunk_id=result.get("chunk_id"),
                    summary_id=result.get("summary_id"),
                    relevance_score=result.get("relevance", 0.8)
                )
                context_ids.append(context_id)
            
            output_data = {
                "query": query_text,
                "context": context_results,
                "context_ids": context_ids
            }
            
            # Update step status
            await self._update_step_status(
                agent_step_id,
                "success",
                output_data
            )
            
            return output_data
            
        except Exception as e:
            await self._update_step_status(
                agent_step_id,
                "failed",
                {"error": str(e)}
            )
            raise