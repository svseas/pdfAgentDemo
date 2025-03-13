"""Recursive document summarization agent."""
import logging
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from src.repositories.workflow_repository import SQLContextRepository
from src.domain.pdf_processor import PDFProcessor
from src.domain.embedding_generator import EmbeddingGenerator
from src.domain.agents.base_agent import BaseAgent
from src.core.llm.interfaces import LLMInterface, PromptTemplateInterface

logger = logging.getLogger(__name__)



LLMType = LLMInterface | None
PromptManagerType = PromptTemplateInterface | None
EmbeddingGeneratorType = EmbeddingGenerator

class RecursiveSummarizationAgent(BaseAgent):
    """Agent that recursively summarizes document chunks."""
    
    def __init__(
        self,
        session: AsyncSession,
        pdf_processor: PDFProcessor,
        embedding_generator: EmbeddingGeneratorType,
        llm: LLMType = None,
        prompt_manager: PromptManagerType = None,
        *args,
        **kwargs
    ):
        super().__init__(session, llm=llm, prompt_manager=prompt_manager, *args, **kwargs)
        self.pdf_processor = pdf_processor
        self.embedding_generator = embedding_generator
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return output."""
        document_id = input_data.get("document_id")
        workflow_run_id = input_data.get("workflow_run_id")
        language = input_data.get("language", "vi")
        max_length = input_data.get("max_length", 500)
        batch_size = 10  # Process 10 chunks at a time
        overlap_size = 3  # 3 chunks overlap between batches
        
        # Log step start
        agent_step_id = await self.log_step(
            workflow_run_id,
            None,
            input_data,
            {},
            "running"
        )
        
        try:
            context_repo = SQLContextRepository(self.session)
            
            # Check for existing summaries
            existing_summaries = await context_repo.get_document_summaries(document_id)
            if existing_summaries["final_summary"]:
                logger.info(f"Found existing summaries for document {document_id}")
                
                # Check if summaries have embeddings
                need_embeddings = False
                if not existing_summaries["final_summary"].get("embedding"):
                    need_embeddings = True
                    logger.info("Final summary missing embedding, will generate")
                
                if not need_embeddings:
                    for summary in existing_summaries["chunk_summaries"]:
                        if not summary.get("embedding"):
                            need_embeddings = True
                            logger.info("Some chunk summaries missing embeddings, will generate")
                            break
                            
                if not need_embeddings:
                    for summary in existing_summaries["intermediate_summaries"]:
                        if not summary.get("embedding"):
                            need_embeddings = True
                            logger.info("Some intermediate summaries missing embeddings, will generate")
                            break
                
                if not need_embeddings:
                    return {
                        "document_id": document_id,
                        "chunk_summaries": existing_summaries["chunk_summaries"],
                        "intermediate_summaries": existing_summaries["intermediate_summaries"],
                        "final_summary": existing_summaries["final_summary"]["text"],
                        "metadata": existing_summaries["metadata"]
                    }
                    
                # Generate missing embeddings
                logger.info("Generating missing embeddings for existing summaries")
                
                # Generate embeddings for chunk summaries if needed
                for summary in existing_summaries["chunk_summaries"]:
                    if not summary.get("embedding"):
                        embedding = self.embedding_generator.generate_embedding(summary["text"])
                        await context_repo.update_summary_embedding(summary["id"], embedding)
                        
                # Generate embeddings for intermediate summaries if needed
                for summary in existing_summaries["intermediate_summaries"]:
                    if not summary.get("embedding"):
                        embedding = self.embedding_generator.generate_embedding(summary["text"])
                        await context_repo.update_summary_embedding(summary["id"], embedding)
                        
                # Generate embedding for final summary if needed
                if not existing_summaries["final_summary"].get("embedding"):
                    embedding = self.embedding_generator.generate_embedding(existing_summaries["final_summary"]["text"])
                    await context_repo.update_summary_embedding(existing_summaries["final_summary"]["id"], embedding)
                
                return {
                    "document_id": document_id,
                    "chunk_summaries": existing_summaries["chunk_summaries"],
                    "intermediate_summaries": existing_summaries["intermediate_summaries"],
                    "final_summary": existing_summaries["final_summary"]["text"],
                    "metadata": existing_summaries["metadata"]
                }

            # No existing summaries, create new ones
            logger.info(f"No existing summaries found for document {document_id}, creating new ones")
            
            # Get document chunks from repository
            chunks = await context_repo.get_document_chunks(document_id)
            logger.info(f"Retrieved {len(chunks)} existing chunks from repository")
            
            # Filter out empty chunks
            chunks = [chunk for chunk in chunks if chunk.get("chunk_text", "").strip()]
            logger.info(f"Processing {len(chunks)} non-empty chunks")
            
            # Create chunk batch summaries
            batch_summaries = []
            for i in range(0, len(chunks), batch_size - overlap_size):
                # Get current batch with overlap
                batch = chunks[i:i + batch_size]
                batch_text = "\n\n".join(chunk["chunk_text"] for chunk in batch)
                
                # Create batch summary
                if self.llm and self.prompt_manager:
                    prompt = self.prompt_manager.format_prompt(
                        "summarization",
                        language=language,
                        text=batch_text,
                        max_length=max_length
                    )
                    
                    summary_text = await self.llm.generate_completion([
                        {
                            "role": "system",
                            "content": f"""You are a document summarization expert.
                            Create a concise summary that:
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
                    summary_text = batch_text[:max_length]
                
                # Generate embedding for batch summary
                embedding = await self.embedding_generator.generate_embedding(summary_text)
                
                # Store batch summary with chunk range metadata and embedding
                summary_id = await context_repo.create_summary(
                    document_id=document_id,
                    summary_level=1,  # Chunk batch level
                    summary_text=summary_text,
                    summary_embedding=embedding,
                    summary_metadata={
                        "chunk_start": i,
                        "chunk_end": i + len(batch) - 1,
                        "total_chunks": len(chunks)
                    }
                )
                
                batch_summaries.append({
                    "id": summary_id,
                    "text": summary_text,
                    "chunk_range": f"{i}-{i + len(batch) - 1}"
                })
                
                logger.info(f"Created summary for chunks {i}-{i + len(batch) - 1}")
            
            # Create intermediate summaries (10 batch summaries per group)
            intermediate_summaries = []
            intermediate_batch_size = 10
            
            for i in range(0, len(batch_summaries), intermediate_batch_size):
                current_batch = batch_summaries[i:i + intermediate_batch_size]
                combined_text = "\n\n".join(f"Chunks {s['chunk_range']}:\n{s['text']}" for s in current_batch)
                
                if self.llm and self.prompt_manager:
                    prompt = self.prompt_manager.format_prompt(
                        "intermediate_summarization",
                        language=language,
                        text=combined_text,
                        max_length=max_length
                    )
                    
                    summary_text = await self.llm.generate_completion([
                        {
                            "role": "system",
                            "content": f"""You are a document summarization expert.
                            Create a comprehensive summary that:
                            1. Synthesizes information from multiple chunk summaries
                            2. Identifies common themes and key points
                            3. Maintains relationships between concepts
                            4. Creates a coherent narrative
                            5. Uses clear language suitable for both human readers and AI analysis
                            6. Stays within {max_length} characters
                            
                            The summary should be in {language} language."""
                        },
                        {"role": "user", "content": prompt}
                    ])
                else:
                    summary_text = combined_text[:max_length]
                
                # Generate embedding for intermediate summary
                summary_embedding = await self.embedding_generator.generate_embedding(summary_text)
                
                # Store intermediate summary with embedding
                summary_id = await context_repo.create_summary(
                    document_id=document_id,
                    summary_level=2,  # Intermediate level
                    summary_text=summary_text,
                    summary_embedding=summary_embedding,
                    summary_metadata={
                        "batch_start": i,
                        "batch_end": i + len(current_batch) - 1,
                        "total_batches": len(batch_summaries)
                    }
                )
                
                # Create hierarchy relationships
                for batch_summary in current_batch:
                    await context_repo.create_summary_hierarchy(
                        parent_summary_id=summary_id,
                        child_summary_id=batch_summary["id"],
                        relationship_type="contains"
                    )
                
                intermediate_summaries.append({
                    "id": summary_id,
                    "text": summary_text,
                    "batch_range": f"{i}-{i + len(current_batch) - 1}"
                })
                
                logger.info(f"Created intermediate summary for batches {i}-{i + len(current_batch) - 1}")
            
            # Create final document summary
            combined_text = "\n\n".join(f"Batch {s['batch_range']}:\n{s['text']}" for s in intermediate_summaries)
            
            if self.llm and self.prompt_manager:
                prompt = self.prompt_manager.format_prompt(
                    "document_summarization",
                    language=language,
                    text=combined_text,
                    max_length=max_length
                )
                
                summary_text = await self.llm.generate_completion([
                    {
                        "role": "system",
                        "content": f"""You are a document summarization expert.
                        Create a final summary that:
                        1. Provides a comprehensive overview of the entire document
                        2. Highlights the most important themes and key points
                        3. Preserves critical relationships and context
                        4. Uses clear language optimized for:
                           - Query analysis (identifying key concepts and relationships)
                           - Answer synthesis (generating accurate and relevant responses)
                        5. Stays within {max_length} characters
                        
                        The summary should be in {language} language."""
                    },
                    {"role": "user", "content": prompt}
                ])
            else:
                summary_text = combined_text[:max_length]
            
            # Generate embedding for final summary
            summary_embedding = await self.embedding_generator.generate_embedding(summary_text)
            
            # Store final summary with embedding
            final_summary_id = await context_repo.create_summary(
                document_id=document_id,
                summary_level=3,  # Document level
                summary_text=summary_text,
                summary_embedding=summary_embedding,
                summary_metadata={
                    "total_chunks": len(chunks),
                    "total_batches": len(batch_summaries),
                    "total_intermediates": len(intermediate_summaries)
                }
            )
            
            # Create hierarchy relationships
            for intermediate_summary in intermediate_summaries:
                await context_repo.create_summary_hierarchy(
                    parent_summary_id=final_summary_id,
                    child_summary_id=intermediate_summary["id"],
                    relationship_type="contains"
                )
            
            # Prepare output data
            output_data = {
                "document_id": document_id,
                "chunk_summaries": batch_summaries,
                "intermediate_summaries": intermediate_summaries,
                "final_summary_id": final_summary_id,
                "final_summary": summary_text,
                "metadata": {
                    "total_chunks": len(chunks),
                    "total_batches": len(batch_summaries),
                    "total_intermediates": len(intermediate_summaries)
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