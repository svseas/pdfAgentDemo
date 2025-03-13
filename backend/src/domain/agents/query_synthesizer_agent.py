"""Query synthesis and answer generation agent."""
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.workflow_repository import (
    AgentStepRepository,
    ContextRepository
)
from src.domain.stepback_agent import StepbackAgent
from src.domain.exceptions import AgentError
from .base_agent import BaseAgent

class QuerySynthesizerAgent(BaseAgent):
    """Agent that synthesizes answers from sub-query results.
    
    This agent is responsible for:
    - Combining results from multiple sub-queries
    - Using stepback prompting for enhanced answers
    - Incorporating citations and context
    - Generating final coherent responses
    
    Attributes:
        context_repo: Repository for context operations
    """
    
    def __init__(
        self,
        session: AsyncSession,
        agent_step_repo: AgentStepRepository,
        context_repo: ContextRepository,
        *args,
        **kwargs
    ):
        """Initialize query synthesizer agent.
        
        Args:
            session: Database session
            agent_step_repo: Repository for agent step logging
            context_repo: Repository for context operations
            *args, **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(session, agent_step_repo, *args, **kwargs)
        self.context_repo = context_repo

    async def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of answer synthesis logic.
        
        Args:
            input_data: Must contain:
                - workflow_run_id: ID of current workflow run
                - original_query: Original query text
                - sub_query_results: List of sub-query results
                - citations: Optional list of citations
                - language: Language code (default: "vi")
                
        Returns:
            Dict containing:
                - original_query: Original query text
                - answer: Generated answer
                - contexts_used: List of used context IDs
                - citations_used: List of used citation IDs
                
        Raises:
            AgentError: If answer synthesis fails
        """
        try:
            workflow_run_id = input_data.get("workflow_run_id")
            original_query = input_data.get("original_query")
            sub_query_results = input_data.get("sub_query_results", [])
            citations = input_data.get("citations", [])
            language = input_data.get("language", "vi")
            
            if not original_query:
                raise AgentError("No original query provided")
            
            # Mark used contexts
            contexts_used = await self._mark_used_contexts(sub_query_results)
            
            # Generate answer
            if self.llm and self.prompt_manager:
                answer = await self._generate_llm_answer(
                    original_query,
                    sub_query_results,
                    citations,
                    language
                )
            else:
                answer = self._generate_template_answer(
                    original_query,
                    sub_query_results,
                    citations,
                    language
                )
            
            return {
                "original_query": original_query,
                "answer": answer,
                "contexts_used": contexts_used,
                "citations_used": [c["id"] for c in citations]
            }
            
        except Exception as e:
            raise AgentError(f"Answer synthesis failed: {str(e)}") from e

    async def _mark_used_contexts(
        self,
        sub_query_results: List[Dict[str, Any]]
    ) -> List[int]:
        """Mark contexts as used and return their IDs.
        
        Args:
            sub_query_results: List of sub-query results
            
        Returns:
            List of used context IDs
            
        Raises:
            AgentError: If context marking fails
        """
        try:
            contexts_used = []
            for result in sub_query_results:
                for context in result["context"]:
                    if "id" in context:
                        contexts_used.append(context["id"])
                        await self.context_repo.mark_context_used(context["id"])
            return contexts_used
            
        except Exception as e:
            raise AgentError(f"Failed to mark used contexts: {str(e)}") from e

    async def _generate_llm_answer(
        self,
        query: str,
        sub_query_results: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        language: str
    ) -> str:
        """Generate answer using LLM with stepback prompting.
        
        Args:
            query: Original query text
            sub_query_results: List of sub-query results
            citations: List of citations
            language: Language code
            
        Returns:
            Generated answer text
            
        Raises:
            AgentError: If answer generation fails
        """
        try:
            # Build context from sub-query results
            context = "\n\n".join([
                f"Sub-query: {result['sub_query']['text']}\n"
                f"Context: {result.get('context', [])}"
                for result in sub_query_results
            ])
            
            # Add citations to context
            if citations:
                context += "\n\nCitations:\n" + "\n".join([
                    f"- {citation['normalized']}"
                    for citation in sorted(
                        citations,
                        key=lambda c: c.get("authority", 0),
                        reverse=True
                    )[:5]
                ])
            
            # Use stepback prompting for enhanced answer
            stepback_agent = StepbackAgent(self.llm)
            
            # Generate initial answer
            initial_answer = await self.llm.generate_completion([
                {
                    "role": "system",
                    "content": "You are an expert at synthesizing information."
                },
                {
                    "role": "user",
                    "content": f"Based on the following information, answer the question: {query}\n\n{context}"
                }
            ])
            
            # Enhance answer using stepback prompting
            return await stepback_agent.enhance_answer(
                context=context,
                query=query,
                initial_answer=initial_answer,
                language=language
            )
            
        except Exception as e:
            raise AgentError(f"Failed to generate LLM answer: {str(e)}") from e

    def _generate_template_answer(
        self,
        query: str,
        sub_query_results: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        language: str = "vi"
    ) -> str:
        """Generate template-based answer when LLM is not available.
        
        Args:
            query: Original query text
            sub_query_results: List of sub-query results
            citations: List of citations
            language: Language code
            
        Returns:
            Generated answer text
        """
        if language == "vi":
            answer = f"Câu trả lời cho: {query}\n\n"
            answer += "Dựa trên các câu hỏi phụ sau:\n"
            for result in sub_query_results:
                answer += f"- {result['sub_query']['text']}\n"
            
            if citations:
                answer += "\nĐược hỗ trợ bởi các trích dẫn sau:\n"
                sorted_citations = sorted(
                    citations,
                    key=lambda c: c.get("authority", 0),
                    reverse=True
                )
                for citation in sorted_citations[:3]:
                    answer += f"- {citation['normalized']}\n"
        else:
            answer = f"Answer for: {query}\n\n"
            answer += "Based on the following sub-queries:\n"
            for result in sub_query_results:
                answer += f"- {result['sub_query']['text']}\n"
            
            if citations:
                answer += "\nSupported by the following citations:\n"
                sorted_citations = sorted(
                    citations,
                    key=lambda c: c.get("authority", 0),
                    reverse=True
                )
                for citation in sorted_citations[:3]:
                    answer += f"- {citation['normalized']}\n"
        
        return answer