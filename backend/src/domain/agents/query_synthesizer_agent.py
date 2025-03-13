"""Query synthesis and answer generation agent."""
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.workflow_repository import SQLContextRepository
from src.domain.stepback_agent import StepbackAgent
from .base_agent import BaseAgent

class QuerySynthesizerAgent(BaseAgent):
    """Agent that synthesizes answers from sub-query results."""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return output."""
        workflow_run_id = input_data.get("workflow_run_id")
        original_query = input_data.get("original_query")
        sub_query_results = input_data.get("sub_query_results", [])
        citations = input_data.get("citations", [])
        language = input_data.get("language", "vi")
        
        # Log step start
        agent_step_id = await self.log_step(
            workflow_run_id,
            None,
            input_data,
            {},
            "running"
        )
        
        try:
            # Combine contexts and citations
            context_repo = SQLContextRepository(self.session)
            contexts_used = []
            
            # Mark used contexts
            for result in sub_query_results:
                for context in result["context"]:
                    if "id" in context:
                        contexts_used.append(context["id"])
                        await context_repo.mark_context_used(context["id"])
            
            # Generate answer using stepback prompting
            if self.llm and self.prompt_manager:
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
                    {"role": "system", "content": "You are an expert at synthesizing information."},
                    {"role": "user", "content": f"Based on the following information, answer the question: {original_query}\n\n{context}"}
                ])
                
                # Enhance answer using stepback prompting
                answer = await stepback_agent.enhance_answer(
                    context=context,
                    query=original_query,
                    initial_answer=initial_answer,
                    language=language
                )
            else:
                # Fallback to template-based answer
                answer = self._generate_template_answer(
                    original_query,
                    sub_query_results,
                    citations,
                    language
                )
            
            output_data = {
                "original_query": original_query,
                "answer": answer,
                "contexts_used": contexts_used,
                "citations_used": [c["id"] for c in citations]
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
    
    def _generate_template_answer(
        self,
        query: str,
        sub_query_results: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        language: str = "vi"
    ) -> str:
        """Generate template-based answer when LLM is not available."""
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