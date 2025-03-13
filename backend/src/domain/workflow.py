"""Workflow orchestrator implementation."""
from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.domain.interfaces import AgentInterface
from backend.src.domain.agents import (
    RecursiveSummarizationAgent,
    QueryAnalyzerAgent,
    ContextBuilderAgent,
    CitationAgent,
    QuerySynthesizerAgent
)
from backend.src.repositories.workflow_repository import (
    SQLWorkflowRepository,
    SQLQueryRepository,
    SQLAgentStepRepository
)

class WorkflowOrchestrator:
    """Orchestrates the flow of information between agents."""
    
    def __init__(self, session: AsyncSession, agent_factory):
        self.session = session
        self.agent_factory = agent_factory
        
        # Initialize repositories
        self.workflow_repo = SQLWorkflowRepository(session)
        self.query_repo = SQLQueryRepository(session)
        self.agent_step_repo = SQLAgentStepRepository(session)
        
        # Initialize agents
        self.summarization_agent = self.agent_factory("summarizer", session)
        self.query_analyzer = self.agent_factory("query_analyzer", session)
        self.context_builder = self.agent_factory("context_builder", session)
        self.citation_agent = self.agent_factory("citation", session)
        self.query_synthesizer = self.agent_factory("synthesizer", session)
    
    async def start_workflow(self, query_text: str) -> Dict[str, Any]:
        """Start a new workflow with the given query."""
        # Create user query
        query_id = await self.query_repo.create_user_query(query_text)
        
        # Create workflow run
        workflow_run_id = await self.workflow_repo.create_workflow_run(
            query_id,
            status="running"
        )
        
        # Initialize workflow state
        workflow_state = {
            "workflow_run_id": workflow_run_id,
            "query_id": query_id,
            "query_text": query_text,
            "sub_queries": [],
            "contexts": [],
            "start_time": datetime.utcnow().isoformat()
        }
        
        try:
            # Process with Query Analyzer
            query_analysis = await self.query_analyzer.process(workflow_state)
            
            # Process each sub-query with Context Builder
            sub_query_results = []
            all_context_results = []
            
            for sub_query in query_analysis["sub_queries"]:
                sub_query_context = await self.context_builder.process({
                    "workflow_run_id": workflow_run_id,
                    "sub_query_id": sub_query["id"],
                    "query_text": sub_query["text"]
                })
                
                sub_query_results.append({
                    "sub_query": sub_query,
                    "context": sub_query_context["context"]
                })
                
                if "context_ids" in sub_query_context:
                    all_context_results.extend(sub_query_context["context_ids"])
            
            # Process with Citation Agent
            citation_results = await self.citation_agent.process({
                "workflow_run_id": workflow_run_id,
                "context_results": all_context_results
            })
            
            # Synthesize final response
            synthesis_result = await self.query_synthesizer.process({
                "workflow_run_id": workflow_run_id,
                "original_query": query_text,
                "sub_query_results": sub_query_results,
                "citations": citation_results["citations"]
            })
            
            # Update workflow completion
            await self.workflow_repo.update_workflow_status(
                workflow_run_id,
                "completed",
                synthesis_result["answer"]
            )
            
            return {
                "workflow_run_id": workflow_run_id,
                "answer": synthesis_result["answer"],
                "sub_queries": query_analysis["sub_queries"],
                "contexts_used": synthesis_result["contexts_used"],
                "citations": citation_results["citations"]
            }
            
        except Exception as e:
            # Handle errors and update workflow status
            await self.workflow_repo.update_workflow_status(
                workflow_run_id,
                "failed"
            )
            raise e