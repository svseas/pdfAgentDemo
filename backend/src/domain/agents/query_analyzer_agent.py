"""Query analysis and decomposition agent."""
import numpy as np
import logging
import json
from typing import Dict, Any, List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from src.repositories.workflow_repository import (
    AgentStepRepository,
    QueryRepository,
    ContextRepository
)
from src.repositories.document_repository import DocumentRepository
from src.domain.stepback_agent import StepbackAgent
from src.domain.embedding_generator import EmbeddingGenerator
from src.domain.exceptions import AgentError
from .base_agent import BaseAgent

class QueryAnalyzerAgent(BaseAgent):
    """Agent that analyzes queries and breaks them down into sub-queries.
    
    This agent is responsible for:
    - Analyzing query intent and structure
    - Breaking down complex queries into sub-queries
    - Finding relevant document summaries
    - Using stepback prompting for enhanced analysis
    
    Attributes:
        embedding_generator: Generator for text embeddings
        query_repo: Repository for query operations
        context_repo: Repository for context operations
        doc_repo: Repository for document operations
    """
    
    def __init__(
        self,
        session: AsyncSession,
        agent_step_repo: AgentStepRepository,
        query_repo: QueryRepository,
        context_repo: ContextRepository,
        doc_repo: DocumentRepository,
        embedding_generator: EmbeddingGenerator,
        *args,
        **kwargs
    ):
        """Initialize quer  y analyzer agent.
        
        Args:
            session: Database session
            agent_step_repo: Repository for agent step logging
            query_repo: Repository for query operations
            context_repo: Repository for context operations
            doc_repo: Repository for document operations
            embedding_generator: Generator for text embeddings
            *args, **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(session, agent_step_repo, *args, **kwargs)
        self.embedding_generator = embedding_generator
        self.query_repo = query_repo
        self.context_repo = context_repo
        self.doc_repo = doc_repo

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        return np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array))

    async def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of query analysis logic.
        
        Args:
            input_data: Must contain:
                - workflow_run_id: ID of current workflow run
                - query_text: Text of query to analyze
                - language: Language code (default: "vi")
                
        Returns:
            Dict containing:
                - original_query: Original query text
                - sub_queries: List of generated sub-queries
                - reasoning: Dict with analysis details
                
        Raises:
            AgentError: If query analysis fails
        """
        try:
            workflow_run_id = input_data.get("workflow_run_id")
            query_text = input_data.get("query_text")
            language = "vi" if input_data.get("language", "vi").lower() in ["vi", "vietnamese"] else "en"
            
            if not query_text:
                raise AgentError("No query text provided")
            
            # Get relevant summaries if LLM available
            relevant_summaries = []
            stepback_result = None
            decomposition_strategy = "template-based fallback"
            
            if self.llm and self.prompt_manager:
                # Generate query embedding and get relevant summaries
                query_embedding = await self.embedding_generator.generate_embedding(query_text)
                relevant_summaries = await self._get_relevant_summaries(query_embedding)
                decomposition_strategy = "LLM-based with summary context"
                
                # Generate sub-queries using LLM
                sub_queries = await self._generate_llm_sub_queries(
                    query_text,
                    relevant_summaries,
                    language,
                    workflow_run_id,
                    input_data.get("query_id")
                )
            else:
                # Fallback to template-based decomposition
                sub_queries = await self._generate_template_sub_queries(
                    query_text,
                    language,
                    workflow_run_id,
                    input_data.get("query_id")
                )
            
            # Store context results if summaries were used
            if relevant_summaries:
                await self._store_summary_contexts(
                    input_data.get("agent_step_id"),  # Pass agent_step_id from input
                    relevant_summaries
                )
            
            return {
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
                    ],
                    "stepback_analysis": stepback_result if stepback_result else "Using template-based decomposition",
                    "decomposition_strategy": decomposition_strategy
                }
            }
            
        except Exception as e:
            raise AgentError(f"Query analysis failed: {str(e)}") from e

    async def _get_relevant_summaries(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Get most relevant summaries for query.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of summaries to return
            
        Returns:
            List of (summary, score) tuples
            
        Raises:
            AgentError: If summary retrieval fails
        """
        try:
            # Get all document IDs
            document_ids = await self.doc_repo.get_all_document_ids()
            
            # Get summaries from all documents
            all_summaries = []
            for doc_id in document_ids:
                summaries = await self.context_repo.get_document_summaries(doc_id)
                
                # Add final summary if it exists
                if summaries["final_summary"] and summaries["final_summary"]["embedding"]:
                    summary = summaries["final_summary"]
                    summary["document_id"] = doc_id
                    all_summaries.append(summary)
                
                # Add intermediate and chunk summaries
                for summary in summaries["intermediate_summaries"] + summaries["chunk_summaries"]:
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
            
        except Exception as e:
            raise AgentError(f"Failed to get relevant summaries: {str(e)}") from e

    async def _generate_llm_sub_queries(
        self,
        query_text: str,
        relevant_summaries: List[Tuple[Dict[str, Any], float]],
        language: str,
        workflow_run_id: int,
        original_query_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Generate sub-queries using LLM.
        
        Args:
            query_text: Original query text
            relevant_summaries: List of relevant summaries
            language: Language code
            workflow_run_id: Current workflow run ID
            original_query_id: Optional original query ID
            
        Returns:
            List of generated sub-queries
            
        Raises:
            AgentError: If sub-query generation fails
        """
        try:
            # Use stepback prompting for query analysis
            stepback_agent = StepbackAgent(self.llm)
            
            # Build context from top summaries
            top_5_summaries = relevant_summaries[:5]
            context = "\n\n".join([
                f"Summary (score {score:.3f}):\n{summary['text']}"
                for summary, score in top_5_summaries
            ])
            
            # Get broader perspective using summaries
            stepback_result = await stepback_agent.generate_stepback(
                context=context,
                query=query_text,
                language=language
            )
            
            # Generate sub-queries using stepback insight
            prompt = self.prompt_manager.format_prompt(
                "query_decomposition",
                language=language,
                query=query_text,
                context=context,
                stepback_analysis=stepback_result
            )
            
            result = await self.llm.generate_completion([
                {"role": "system", "content": "You are a query analysis expert. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ])
            
            logger.info(f"LLM Response: {result}")
            
            try:
                # Parse JSON response
                result = result.strip()
                if result.startswith("```json"):
                    result = result[7:]  # Skip ```json
                if result.endswith("```"):
                    result = result[:-3]  # Skip ```
                    
                response_data = json.loads(result.strip())
                sub_queries = response_data.get("sub_queries", [])
                
                # Create sub-queries with embeddings
                created_queries = []
                for query in sub_queries:
                    # Generate embedding
                    try:
                        query_embedding = await self.embedding_generator.generate_embedding(query["text"])
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for sub-query: {str(e)}")
                        query_embedding = None
                    
                    # Create sub-query
                    sub_query_id = await self.query_repo.create_sub_query(
                        workflow_run_id,
                        original_query_id,
                        query["text"],
                        query_embedding
                    )
                    created_queries.append({
                        "id": sub_query_id,
                        "text": query["text"]
                    })
                
                return created_queries
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                raise AgentError("Invalid response format from LLM")
            
        except Exception as e:
            raise AgentError(f"Failed to generate LLM sub-queries: {str(e)}") from e

    async def _generate_template_sub_queries(
        self,
        query_text: str,
        language: str,
        workflow_run_id: int,
        original_query_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Generate sub-queries using templates.
        
        Args:
            query_text: Original query text
            language: Language code
            workflow_run_id: Current workflow run ID
            original_query_id: Optional original query ID
            
        Returns:
            List of generated sub-queries
            
        Raises:
            AgentError: If sub-query generation fails
        """
        try:
            templates = {
                "vi": [
                    "Những định nghĩa và khái niệm chính liên quan đến: {query}",
                    "Các thông tin quan trọng về: {query}",
                    "Các luận điểm và dẫn chứng liên quan đến: {query}",
                    "Các ví dụ và trường hợp áp dụng của: {query}",
                    "Các quy định và hướng dẫn về: {query}"
                ],
                "en": [
                    "Key definitions and concepts related to: {query}",
                    "Important information about: {query}",
                    "Arguments and evidence regarding: {query}",
                    "Examples and applications of: {query}",
                    "Rules and guidelines about: {query}"
                ]
            }
            
            sub_query_texts = [
                template.format(query=query_text)
                for template in templates["vi" if language == "vi" else "en"]
            ]
            
            return await self._create_sub_queries(
                sub_query_texts,
                workflow_run_id,
                original_query_id
            )
            
        except Exception as e:
            raise AgentError(f"Failed to generate template sub-queries: {str(e)}") from e

    async def _create_sub_queries(
        self,
        sub_query_texts: List[str],
        workflow_run_id: int,
        original_query_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Create sub-query records in database.
        
        Args:
            sub_query_texts: List of sub-query texts
            workflow_run_id: Current workflow run ID
            original_query_id: Optional original query ID
            
        Returns:
            List of created sub-queries
            
        Raises:
            AgentError: If sub-query creation fails
        """
        try:
            sub_queries = []
            for text in sub_query_texts:
                text = text.strip()
                # Keep only the actual questions
                if (text and
                    "?" in text and  # Must be a question
                    not text.startswith(("Dưới đây", "Các câu")) and  # Skip introductory text
                    "**" in text):  # Questions are usually marked with **
                    
                    # Clean up the question text
                    question = text.replace("**", "").strip()
                    if question.startswith(("#### ", "### ")):
                        question = question[5:] if question.startswith("#### ") else question[4:]
                    if question.startswith(("1. ", "2. ", "3. ", "4. ", "5. ")):
                        question = question[3:]
                    
                    # Generate embedding for sub-query
                    try:
                        sub_query_embedding = await self.embedding_generator.generate_embedding(question)
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for sub-query: {str(e)}")
                        sub_query_embedding = None

                    # Create sub-query with embedding
                    sub_query_id = await self.query_repo.create_sub_query(
                        workflow_run_id,
                        original_query_id,
                        question,
                        sub_query_embedding
                    )
                    sub_queries.append({
                        "id": sub_query_id,
                        "text": question
                    })
            return sub_queries
            
        except Exception as e:
            raise AgentError(f"Failed to create sub-queries: {str(e)}") from e

    async def _store_summary_contexts(
        self,
        agent_step_id: int,
        relevant_summaries: List[Tuple[Dict[str, Any], float]]
    ) -> None:
        """Store used summaries as context results.
        
        Args:
            agent_step_id: Current agent step ID
            relevant_summaries: List of relevant summaries
            
        Raises:
            AgentError: If context storage fails
        """
        try:
            if not agent_step_id:
                raise AgentError("No agent step ID provided for storing contexts")
                
            for summary, score in relevant_summaries:
                await self.context_repo.create_context_result(
                    agent_step_id=agent_step_id,
                    document_id=summary["document_id"],
                    chunk_id=None,
                    summary_id=summary["id"],
                    relevance_score=score,
                    used_in_response=True
                )
        except Exception as e:
            raise AgentError(f"Failed to store summary contexts: {str(e)}") from e