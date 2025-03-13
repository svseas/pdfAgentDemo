"""Recursive document summarization agent."""
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.workflow_repository import (
    AgentStepRepository,
    ContextRepository
)
from src.domain.pdf_processor import PDFProcessor
from src.domain.embedding_generator import EmbeddingGenerator
from src.domain.exceptions import AgentError
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class RecursiveSummarizationAgent(BaseAgent):
    """Agent that recursively summarizes document chunks.
    
    This agent is responsible for:
    - Creating multi-level document summaries
    - Generating embeddings for summaries
    - Managing summary hierarchies
    - Handling batch processing
    - Implementing caching strategies
    
    The summarization process has three levels:
    1. Chunk summaries (level 1)
    2. Intermediate summaries (level 2)
    3. Final document summary (level 3)
    
    Attributes:
        pdf_processor: Processor for PDF documents
        embedding_generator: Generator for text embeddings
        context_repo: Repository for context operations
    """
    
    def __init__(
        self,
        session: AsyncSession,
        agent_step_repo: AgentStepRepository,
        context_repo: ContextRepository,
        pdf_processor: PDFProcessor,
        embedding_generator: EmbeddingGenerator,
        *args,
        **kwargs
    ):
        """Initialize recursive summarization agent.
        
        Args:
            session: Database session
            agent_step_repo: Repository for agent step logging
            context_repo: Repository for context operations
            pdf_processor: Processor for PDF documents
            embedding_generator: Generator for text embeddings
            *args, **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(session, agent_step_repo, *args, **kwargs)
        self.pdf_processor = pdf_processor
        self.embedding_generator = embedding_generator
        self.context_repo = context_repo

    async def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of recursive summarization logic.
        
        Args:
            input_data: Must contain:
                - document_id: ID of document to summarize
                - workflow_run_id: ID of current workflow run
                - language: Language code (default: "vi")
                - max_length: Maximum summary length (default: 500)
                
        Returns:
            Dict containing:
                - document_id: ID of processed document
                - chunk_summaries: List of chunk-level summaries
                - intermediate_summaries: List of intermediate summaries
                - final_summary: Final document summary
                - metadata: Processing metadata
                
        Raises:
            AgentError: If summarization fails
        """
        try:
            document_id = input_data.get("document_id")
            if not document_id:
                raise AgentError("No document ID provided")
                
            language = input_data.get("language", "vi")
            max_length = input_data.get("max_length", 500)
            
            # Check for existing summaries
            existing = await self._check_existing_summaries(document_id)
            if existing:
                return existing
            
            # Get document chunks
            chunks = await self._get_document_chunks(document_id)
            if not chunks:
                raise AgentError("No document chunks found")
            
            # Process in batches
            batch_size = 10
            overlap_size = 3
            
            # Create chunk summaries
            chunk_summaries = await self._create_chunk_summaries(
                document_id,
                chunks,
                batch_size,
                overlap_size,
                language,
                max_length
            )
            
            # Create intermediate summaries
            intermediate_summaries = await self._create_intermediate_summaries(
                document_id,
                chunk_summaries,
                language,
                max_length
            )
            
            # Create final summary
            final_summary = await self._create_final_summary(
                document_id,
                intermediate_summaries,
                language,
                max_length
            )
            
            return {
                "document_id": document_id,
                "chunk_summaries": chunk_summaries,
                "intermediate_summaries": intermediate_summaries,
                "final_summary": final_summary["text"],
                "metadata": {
                    "total_chunks": len(chunks),
                    "total_batches": len(chunk_summaries),
                    "total_intermediates": len(intermediate_summaries)
                }
            }
            
        except Exception as e:
            raise AgentError(f"Document summarization failed: {str(e)}") from e

    async def _check_existing_summaries(
        self,
        document_id: int
    ) -> Optional[Dict[str, Any]]:
        """Check for and update existing summaries.
        
        Args:
            document_id: Document ID to check
            
        Returns:
            Dict with existing summaries if found and valid
            
        Raises:
            AgentError: If summary checking fails
        """
        try:
            existing = await self.context_repo.get_document_summaries(document_id)
            if not existing["final_summary"]:
                return None
                
            # Check if embeddings need updating
            need_embeddings = False
            
            if not existing["final_summary"].get("embedding"):
                need_embeddings = True
                logger.info("Final summary missing embedding")
                
            for summary in existing["chunk_summaries"]:
                if not summary.get("embedding"):
                    need_embeddings = True
                    logger.info("Some chunk summaries missing embeddings")
                    break
                    
            for summary in existing["intermediate_summaries"]:
                if not summary.get("embedding"):
                    need_embeddings = True
                    logger.info("Some intermediate summaries missing embeddings")
                    break
                    
            if need_embeddings:
                await self._update_missing_embeddings(existing)
                
            return {
                "document_id": document_id,
                "chunk_summaries": existing["chunk_summaries"],
                "intermediate_summaries": existing["intermediate_summaries"],
                "final_summary": existing["final_summary"]["text"],
                "metadata": existing["metadata"]
            }
            
        except Exception as e:
            raise AgentError(f"Failed to check existing summaries: {str(e)}") from e

    async def _update_missing_embeddings(
        self,
        summaries: Dict[str, Any]
    ) -> None:
        """Update missing embeddings for existing summaries.
        
        Args:
            summaries: Dict containing existing summaries
            
        Raises:
            AgentError: If embedding update fails
        """
        try:
            # Update chunk summary embeddings
            for summary in summaries["chunk_summaries"]:
                if not summary.get("embedding"):
                    embedding = await self.embedding_generator.generate_embedding(
                        summary["text"]
                    )
                    await self.context_repo.update_summary_embedding(
                        summary["id"],
                        embedding
                    )
            
            # Update intermediate summary embeddings
            for summary in summaries["intermediate_summaries"]:
                if not summary.get("embedding"):
                    embedding = await self.embedding_generator.generate_embedding(
                        summary["text"]
                    )
                    await self.context_repo.update_summary_embedding(
                        summary["id"],
                        embedding
                    )
            
            # Update final summary embedding
            if not summaries["final_summary"].get("embedding"):
                embedding = await self.embedding_generator.generate_embedding(
                    summaries["final_summary"]["text"]
                )
                await self.context_repo.update_summary_embedding(
                    summaries["final_summary"]["id"],
                    embedding
                )
                
        except Exception as e:
            raise AgentError(f"Failed to update missing embeddings: {str(e)}") from e

    async def _get_document_chunks(
        self,
        document_id: int
    ) -> List[Dict[str, Any]]:
        """Get document chunks from repository.
        
        Args:
            document_id: Document ID to get chunks for
            
        Returns:
            List of document chunks
            
        Raises:
            AgentError: If chunk retrieval fails
        """
        try:
            chunks = await self.context_repo.get_document_chunks(document_id)
            return [
                chunk for chunk in chunks
                if chunk.get("chunk_text", "").strip()
            ]
        except Exception as e:
            raise AgentError(f"Failed to get document chunks: {str(e)}") from e

    async def _create_chunk_summaries(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]],
        batch_size: int,
        overlap_size: int,
        language: str,
        max_length: int
    ) -> List[Dict[str, Any]]:
        """Create summaries for chunk batches.
        
        Args:
            document_id: Document ID being processed
            chunks: List of document chunks
            batch_size: Number of chunks per batch
            overlap_size: Number of overlapping chunks
            language: Language code
            max_length: Maximum summary length
            
        Returns:
            List of created chunk summaries
            
        Raises:
            AgentError: If chunk summarization fails
        """
        try:
            summaries = []
            for i in range(0, len(chunks), batch_size - overlap_size):
                # Get current batch with overlap
                batch = chunks[i:i + batch_size]
                batch_text = "\n\n".join(
                    chunk["chunk_text"] for chunk in batch
                )
                
                # Create batch summary
                summary_text = await self._generate_summary(
                    batch_text,
                    language,
                    max_length,
                    summary_type="chunk"
                )
                
                # Generate embedding
                embedding = await self.embedding_generator.generate_embedding(
                    summary_text
                )
                
                # Store summary
                summary_id = await self.context_repo.create_summary(
                    document_id=document_id,
                    summary_level=1,
                    summary_text=summary_text,
                    summary_embedding=embedding,
                    summary_metadata={
                        "chunk_start": i,
                        "chunk_end": i + len(batch) - 1,
                        "total_chunks": len(chunks)
                    }
                )
                
                summaries.append({
                    "id": summary_id,
                    "text": summary_text,
                    "chunk_range": f"{i}-{i + len(batch) - 1}"
                })
                
                logger.info(f"Created summary for chunks {i}-{i + len(batch) - 1}")
                
            return summaries
            
        except Exception as e:
            raise AgentError(f"Failed to create chunk summaries: {str(e)}") from e

    async def _create_intermediate_summaries(
        self,
        document_id: int,
        chunk_summaries: List[Dict[str, Any]],
        language: str,
        max_length: int,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Create intermediate level summaries.
        
        Args:
            document_id: Document ID being processed
            chunk_summaries: List of chunk summaries
            language: Language code
            max_length: Maximum summary length
            batch_size: Number of summaries per batch
            
        Returns:
            List of created intermediate summaries
            
        Raises:
            AgentError: If intermediate summarization fails
        """
        try:
            summaries = []
            for i in range(0, len(chunk_summaries), batch_size):
                current_batch = chunk_summaries[i:i + batch_size]
                combined_text = "\n\n".join(
                    f"Chunks {s['chunk_range']}:\n{s['text']}"
                    for s in current_batch
                )
                
                # Create intermediate summary
                summary_text = await self._generate_summary(
                    combined_text,
                    language,
                    max_length,
                    summary_type="intermediate"
                )
                
                # Generate embedding
                embedding = await self.embedding_generator.generate_embedding(
                    summary_text
                )
                
                # Store summary
                summary_id = await self.context_repo.create_summary(
                    document_id=document_id,
                    summary_level=2,
                    summary_text=summary_text,
                    summary_embedding=embedding,
                    summary_metadata={
                        "batch_start": i,
                        "batch_end": i + len(current_batch) - 1,
                        "total_batches": len(chunk_summaries)
                    }
                )
                
                # Create hierarchy relationships
                for chunk_summary in current_batch:
                    await self.context_repo.create_summary_hierarchy(
                        parent_summary_id=summary_id,
                        child_summary_id=chunk_summary["id"],
                        relationship_type="contains"
                    )
                
                summaries.append({
                    "id": summary_id,
                    "text": summary_text,
                    "batch_range": f"{i}-{i + len(current_batch) - 1}"
                })
                
                logger.info(
                    f"Created intermediate summary for batches "
                    f"{i}-{i + len(current_batch) - 1}"
                )
                
            return summaries
            
        except Exception as e:
            raise AgentError(
                f"Failed to create intermediate summaries: {str(e)}"
            ) from e

    async def _create_final_summary(
        self,
        document_id: int,
        intermediate_summaries: List[Dict[str, Any]],
        language: str,
        max_length: int
    ) -> Dict[str, Any]:
        """Create final document summary.
        
        Args:
            document_id: Document ID being processed
            intermediate_summaries: List of intermediate summaries
            language: Language code
            max_length: Maximum summary length
            
        Returns:
            Dict containing final summary
            
        Raises:
            AgentError: If final summarization fails
        """
        try:
            # Combine intermediate summaries
            combined_text = "\n\n".join(
                f"Batch {s['batch_range']}:\n{s['text']}"
                for s in intermediate_summaries
            )
            
            # Create final summary
            summary_text = await self._generate_summary(
                combined_text,
                language,
                max_length,
                summary_type="final"
            )
            
            # Generate embedding
            embedding = await self.embedding_generator.generate_embedding(
                summary_text
            )
            
            # Store final summary
            summary_id = await self.context_repo.create_summary(
                document_id=document_id,
                summary_level=3,
                summary_text=summary_text,
                summary_embedding=embedding,
                summary_metadata={
                    "total_intermediates": len(intermediate_summaries)
                }
            )
            
            # Create hierarchy relationships
            for intermediate_summary in intermediate_summaries:
                await self.context_repo.create_summary_hierarchy(
                    parent_summary_id=summary_id,
                    child_summary_id=intermediate_summary["id"],
                    relationship_type="contains"
                )
            
            return {
                "id": summary_id,
                "text": summary_text
            }
            
        except Exception as e:
            raise AgentError(f"Failed to create final summary: {str(e)}") from e

    async def _generate_summary(
        self,
        text: str,
        language: str,
        max_length: int,
        summary_type: str
    ) -> str:
        """Generate summary using LLM or fallback method.
        
        Args:
            text: Text to summarize
            language: Language code
            max_length: Maximum summary length
            summary_type: Type of summary being generated
            
        Returns:
            Generated summary text
            
        Raises:
            AgentError: If summary generation fails
        """
        try:
            if self.llm and self.prompt_manager:
                # Get appropriate prompt template
                template_map = {
                    "chunk": "summarization",
                    "intermediate": "intermediate_summarization",
                    "final": "document_summarization"
                }
                
                prompt = self.prompt_manager.format_prompt(
                    template_map[summary_type],
                    language=language,
                    text=text,
                    max_length=max_length
                )
                
                return await self.llm.generate_completion([
                    {
                        "role": "system",
                        "content": f"""You are a document summarization expert.
                        Create a {summary_type} summary that:
                        1. Captures key information and maintains logical flow
                        2. Preserves important details and relationships
                        3. Focuses on facts and main points
                        4. Uses clear language suitable for both human readers and AI analysis
                        5. Stays within {max_length} characters
                        
                        The summary should be in {language} language."""
                    },
                    {"role": "user", "content": prompt}
                ])
            else:
                # Fallback to simple truncation
                return text[:max_length]
                
        except Exception as e:
            raise AgentError(f"Failed to generate summary: {str(e)}") from e