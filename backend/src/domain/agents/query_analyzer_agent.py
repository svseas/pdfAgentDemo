"""Query analysis and decomposition agent."""
import numpy as np
from typing import Dict, Any, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.workflow_repository import (
    SQLQueryRepository,
    SQLContextRepository
)
from src.domain.stepback_agent import StepbackAgent
from src.domain.embedding_generator import EmbeddingGenerator
from .base_agent import BaseAgent

class QueryAnalyzerAgent(BaseAgent):
    """Agent that analyzes queries and breaks them down into sub-queries."""
    
    def __init__(self, session: AsyncSession, llm=None, prompt_manager=None):
        """Initialize agent with embedding generator."""
        super().__init__(session, llm, prompt_manager)
        self.embedding_generator = EmbeddingGenerator()
        self.prompt_builder = prompt_manager if prompt_manager else None
    
    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        return np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array))

    async def get_all_document_ids(self) -> List[int]:
        """Get all document IDs from the repository."""
        from src.repositories.document_repository import DocumentRepository
        doc_repo = DocumentRepository(self.session)
        return await doc_repo.get_all_document_ids()

    async def get_relevant_summaries(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Get most relevant summaries across all documents for a query."""
        context_repo = SQLContextRepository(self.session)
        
        # Get all document IDs
        document_ids = await self.get_all_document_ids()
        
        # Combine all summaries with embeddings from all documents
        all_summaries = []
        for doc_id in document_ids:
            summaries = await context_repo.get_document_summaries(doc_id)
            if summaries["final_summary"] and summaries["final_summary"]["embedding"]:
                summary = summaries["final_summary"]
                summary["document_id"] = doc_id
                all_summaries.append(summary)
            for summary in summaries["intermediate_summaries"]:
                summary["document_id"] = doc_id
                all_summaries.append(summary)
            for summary in summaries["chunk_summaries"]:
                summary["document_id"] = doc_id
                all_summaries.append(summary)
        
        # Calculate similarities and sort
        scored_summaries = []
        for summary in all_summaries:
            if summary["embedding"]:
                similarity = self.cosine_similarity(query_embedding, summary["embedding"])
                scored_summaries.append((summary, similarity))
                
        scored_summaries.sort(key=lambda x: x[1], reverse=True)
        return scored_summaries[:top_k]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return output."""
        workflow_run_id = input_data.get("workflow_run_id")
        query_text = input_data.get("query_text")
        # Normalize language code
        language = "vi" if input_data.get("language", "vi").lower() in ["vi", "vietnamese"] else "en"
        
        # Log step start
        agent_step_id = await self.log_step(
            workflow_run_id,
            None,
            input_data,
            {},
            "running"
        )
        
        try:
            query_repo = SQLQueryRepository(self.session)
            context_repo = SQLContextRepository(self.session)
            sub_queries = []
            
            # Initialize variables
            relevant_summaries = []
            stepback_result = None
            decomposition_strategy = "template-based fallback"

            if self.llm and self.prompt_manager:
                # Generate query embedding
                query_embedding = await self.embedding_generator.generate_embedding(query_text)
                
                # Get relevant summaries from all documents
                relevant_summaries = await self.get_relevant_summaries(
                    query_embedding,
                    top_k=10
                )
                decomposition_strategy = "LLM-based with summary context"
                
                # Build context from top summaries
                context = "\n\n".join([
                    f"Summary (score {score:.3f}):\n{summary['text']}"
                    for summary, score in relevant_summaries
                ])
                
                # Use stepback prompting for query analysis
                stepback_agent = StepbackAgent(self.llm)
                
                # Get broader perspective using summaries as context
                stepback_result = await stepback_agent.generate_stepback(
                    context=context,
                    query=query_text,
                    language=language
                )
                
                # Use top 5 summaries for sub-query generation
                top_5_summaries = relevant_summaries[:5]
                top_5_context = "\n\n".join([
                    f"Summary (score {score:.3f}):\n{summary['text']}"
                    for summary, score in top_5_summaries
                ])
                
                # Use LLM with stepback insight for query decomposition
                prompt = self.prompt_builder.format_prompt(
                    "query_decomposition",
                    language=language,
                    query=query_text,
                    context=top_5_context,
                    stepback_analysis=stepback_result
                )
                
                result = await self.llm.generate_completion([
                    {"role": "system", "content": "You are a query analysis expert."},
                    {"role": "user", "content": prompt}
                ])
                
                # Parse sub-queries from LLM response
                for line in result.split("\n"):
                    if line.strip() and not line.startswith(("Based on", "Consider", "Note")):
                        sub_query_id = await query_repo.create_sub_query(
                            workflow_run_id,
                            input_data.get("query_id"),
                            line.strip()
                        )
                        
                        sub_queries.append({
                            "id": sub_query_id,
                            "text": line.strip()
                        })
                
                # Log used summaries as context results
                for summary, score in relevant_summaries:
                    await context_repo.create_context_result(
                        agent_step_id=agent_step_id,
                        document_id=summary["document_id"],
                        chunk_id=None,
                        summary_id=summary["id"],
                        relevance_score=score,
                        used_in_response=True
                    )
            else:
                # Fallback to template-based decomposition
                templates_vi = [
                    "Những định nghĩa và khái niệm chính liên quan đến: {query}",
                    "Các thông tin quan trọng về: {query}",
                    "Các luận điểm và dẫn chứng liên quan đến: {query}",
                    "Các ví dụ và trường hợp áp dụng của: {query}",
                    "Các quy định và hướng dẫn về: {query}"
                ]
                
                templates_en = [
                    "Key definitions and concepts related to: {query}",
                    "Important information about: {query}",
                    "Arguments and evidence regarding: {query}",
                    "Examples and applications of: {query}",
                    "Rules and guidelines about: {query}"
                ]
                
                templates = templates_vi if language == "vi" else templates_en
                
                for template in templates:
                    sub_query_text = template.format(query=query_text)
                    sub_query_id = await query_repo.create_sub_query(
                        workflow_run_id,
                        input_data.get("query_id"),
                        sub_query_text
                    )
                    
                    sub_queries.append({
                        "id": sub_query_id,
                        "text": sub_query_text
                    })
            
            output_data = {
                "original_query": query_text,
                "sub_queries": sub_queries,
                "reasoning": {
                    "relevant_summaries": [
                        {
                            "text": summary["text"],
                            "score": score,
                            "level": summary["metadata"].get("level", 0),
                            "document_id": summary["document_id"]
                        }
                        for summary, score in relevant_summaries
                    ] if relevant_summaries else [],
                    "stepback_analysis": stepback_result if stepback_result else "Using template-based decomposition due to LLM unavailability",
                    "decomposition_strategy": decomposition_strategy
                }
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