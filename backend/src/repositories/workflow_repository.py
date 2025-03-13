"""Repository implementations for workflow tracking."""
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from ..domain.interfaces import (
    WorkflowRepository,
    QueryRepository,
    AgentStepRepository,
    ContextRepository,
    CitationRepository
)
from ..models.workflow import (
    WorkflowRun,
    UserQuery,
    SubQuery,
    AgentStep,
    ContextResult,
    Citation,
    ResponseCitation,
    DocumentSummary
)
from .document_repository import DocumentRepository

class SQLWorkflowRepository(WorkflowRepository):
    """SQLAlchemy implementation of WorkflowRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: Dict[str, Any]) -> Any:
        """Create a new workflow run."""
        try:
            result = await self.create_workflow_run(
                user_query_id=data["user_query_id"],
                status=data.get("status", "running")
            )
            await self.session.commit()
            return result
        except Exception as e:
            await self.session.rollback()
            raise

    async def get_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """Get workflow run by ID."""
        try:
            result = await self.session.execute(
                select(WorkflowRun).where(WorkflowRun.id == id)
            )
            workflow = result.scalar_one_or_none()
            if workflow:
                return {
                    "id": workflow.id,
                    "user_query_id": workflow.user_query_id,
                    "status": workflow.status,
                    "started_at": workflow.started_at,
                    "completed_at": workflow.completed_at,
                    "final_answer": workflow.final_answer
                }
            return None
        except Exception as e:
            await self.session.rollback()
            raise

    async def update(self, id: int, data: Dict[str, Any]) -> Any:
        """Update workflow run."""
        try:
            workflow = await self.session.get(WorkflowRun, id)
            if workflow:
                for key, value in data.items():
                    setattr(workflow, key, value)
                await self.session.commit()
                return await self.get_by_id(id)
            return None
        except Exception as e:
            await self.session.rollback()
            raise

    async def delete(self, id: int) -> bool:
        """Delete workflow run."""
        try:
            workflow = await self.session.get(WorkflowRun, id)
            if workflow:
                await self.session.delete(workflow)
                await self.session.commit()
                return True
            return False
        except Exception as e:
            await self.session.rollback()
            raise

    async def create_workflow_run(
        self,
        user_query_id: int,
        status: str = "running"
    ) -> int:
        """Create a new workflow run."""
        try:
            workflow = WorkflowRun(
                user_query_id=user_query_id,
                status=status
            )
            self.session.add(workflow)
            await self.session.flush()  # Get ID without committing
            return workflow.id
        except Exception as e:
            await self.session.rollback()
            raise

    async def update_workflow_status(
        self,
        workflow_id: int,
        status: str,
        final_answer: Optional[str] = None
    ) -> None:
        """Update workflow status and final answer."""
        try:
            workflow = await self.session.get(WorkflowRun, workflow_id)
            if workflow:
                workflow.status = status
                if final_answer is not None:
                    workflow.final_answer = final_answer
                if status in ["completed", "failed"]:
                    workflow.completed_at = datetime.utcnow()
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise

class SQLQueryRepository(QueryRepository):
    """SQLAlchemy implementation of QueryRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_user_query(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None
    ) -> int:
        """Create a new user query."""
        try:
            query = UserQuery(
                query_text=query_text,
                query_embedding=query_embedding
            )
            self.session.add(query)
            await self.session.flush()  # Get ID without committing
            return query.id
        except Exception as e:
            await self.session.rollback()
            raise

class SQLAgentStepRepository(AgentStepRepository):
    """SQLAlchemy implementation of AgentStepRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: Dict[str, Any]) -> Any:
        """Create a new agent step."""
        return await self.log_agent_step(
            workflow_run_id=data["workflow_run_id"],
            agent_type=data["agent_type"],
            sub_query_id=data.get("sub_query_id"),
            input_data=data["input_data"],
            output_data=data["output_data"],
            status=data["status"],
            start_time=data["start_time"]
        )

    async def get_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """Get agent step by ID."""
        try:
            result = await self.session.execute(
                select(AgentStep).where(AgentStep.id == id)
            )
            step = result.scalar_one_or_none()
            if step:
                return {
                    "id": step.id,
                    "workflow_run_id": step.workflow_run_id,
                    "agent_type": step.agent_type,
                    "sub_query_id": step.sub_query_id,
                    "input_data": step.input_data,
                    "output_data": step.output_data,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "status": step.status
                }
            return None
        except Exception as e:
            await self.session.rollback()
            raise

    async def update(self, id: int, data: Dict[str, Any]) -> Any:
        """Update agent step."""
        try:
            step = await self.session.get(AgentStep, id)
            if step:
                for key, value in data.items():
                    setattr(step, key, value)
                await self.session.commit()
                return await self.get_by_id(id)
            return None
        except Exception as e:
            await self.session.rollback()
            raise

    async def delete(self, id: int) -> bool:
        """Delete agent step."""
        try:
            step = await self.session.get(AgentStep, id)
            if step:
                await self.session.delete(step)
                await self.session.commit()
                return True
            return False
        except Exception as e:
            await self.session.rollback()
            raise

    async def log_agent_step(
        self,
        workflow_run_id: int,
        agent_type: str,
        sub_query_id: Optional[int],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str,
        start_time: datetime
    ) -> int:
        """Log an agent processing step."""
        try:
            step = AgentStep(
                workflow_run_id=workflow_run_id,
                agent_type=agent_type,
                sub_query_id=sub_query_id,
                input_data=input_data,
                output_data=output_data,
                start_time=start_time,
                status=status
            )
            self.session.add(step)
            await self.session.flush()  # Get ID without committing
            return step.id
        except Exception as e:
            await self.session.rollback()
            raise

    async def update_step_status(
        self,
        step_id: int,
        status: str,
        output_data: Dict[str, Any],
        end_time: datetime
    ) -> None:
        """Update agent step status and output."""
        try:
            step = await self.session.get(AgentStep, step_id)
            if step:
                step.status = status
                step.output_data = output_data
                step.end_time = end_time
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise

class SQLContextRepository(ContextRepository):
    """SQLAlchemy implementation of ContextRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document with their metadata and embeddings.
        
        Args:
            document_id: ID of the document to get chunks for
            
        Returns:
            List of dictionaries containing chunk information:
            - id: Chunk ID
            - chunk_text: Content of the chunk
            - chunk_index: Position in document
            - document_id: Parent document ID
            - metadata: Additional chunk metadata including:
                - position: Chunk index in document
                - total_chunks: Total number of chunks
                - embedding: Vector embedding if available
                
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            doc_repo = DocumentRepository(self.session)
            chunks = await doc_repo.get_chunks_by_doc_id(document_id)
            
            if not chunks:
                logger.warning(f"No chunks found for document {document_id}")
                return []
                
            total_chunks = len(chunks)
            logger.info(f"Found {total_chunks} chunks for document {document_id}")
            
            return [
                {
                    "id": chunk.id,
                    "chunk_text": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.doc_metadata_id,
                    "metadata": {
                        "position": chunk.chunk_index,
                        "total_chunks": total_chunks,
                        "embedding": chunk.embedding.tolist() if chunk.embedding is not None else None
                    }
                }
                for chunk in chunks
            ]
            
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
            await self.session.rollback()
            raise

    async def get_document_summaries(self, document_id: int) -> Dict[str, Any]:
        """Get all summaries for a document with their metadata and embeddings.
        
        Args:
            document_id: ID of the document to get summaries for
            
        Returns:
            Dictionary containing:
            - chunk_summaries: List of batch-level summaries
            - intermediate_summaries: List of intermediate summaries
            - final_summary: Document-level summary if available
            - metadata: Summary statistics and status
            
        Each summary contains:
            - id: Summary ID
            - text: Summary content
            - metadata: Additional metadata
            - embedding: Vector embedding if available
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        try:
            # Get all summaries for the document
            result = await self.session.execute(
                select(DocumentSummary)
                .where(DocumentSummary.document_id == document_id)
                .order_by(DocumentSummary.created_at)
            )
            summaries = result.scalars().all()
            
            if not summaries:
                logger.warning(f"No summaries found for document {document_id}")
                return {
                    "chunk_summaries": [],
                    "intermediate_summaries": [],
                    "final_summary": None,
                    "metadata": {
                        "total_chunks": 0,
                        "total_intermediates": 0,
                        "has_final": False
                    }
                }
                
            logger.info(f"Found {len(summaries)} summaries for document {document_id}")
            
            # Organize summaries by level
            chunk_summaries = []
            intermediate_summaries = []
            final_summary = None
            
            for summary in summaries:
                metadata = summary.summary_metadata or {}
                level = metadata.get("level", 0)
                
                summary_data = {
                    "id": summary.id,
                    "text": summary.summary_text,
                    "metadata": metadata,
                    "embedding": summary.embedding.tolist() if summary.embedding is not None else None
                }
                
                if level == 1:  # Chunk batch level
                    chunk_summaries.append(summary_data)
                elif level == 2:  # Intermediate level
                    intermediate_summaries.append(summary_data)
                elif level == 3:  # Document level
                    final_summary = summary_data
            
            return {
                "chunk_summaries": chunk_summaries,
                "intermediate_summaries": intermediate_summaries,
                "final_summary": final_summary,
                "metadata": {
                    "total_chunks": len(chunk_summaries),
                    "total_intermediates": len(intermediate_summaries),
                    "has_final": final_summary is not None
                }
            }
            
        except Exception as e:
            await self.session.rollback()
            raise

    async def create_context_result(
        self,
        agent_step_id: int,
        document_id: int,
        chunk_id: Optional[int],
        summary_id: Optional[int],
        relevance_score: float,
        used_in_response: bool = False
    ) -> int:
        """Create a new context result."""
        try:
            context = ContextResult(
                agent_step_id=agent_step_id,
                document_id=document_id,
                chunk_id=chunk_id,
                summary_id=summary_id,
                relevance_score=relevance_score,
                used_in_response=used_in_response
            )
            self.session.add(context)
            await self.session.flush()  # Get ID without committing
            return context.id
        except Exception as e:
            await self.session.rollback()
            raise

    async def create_summary(
        self,
        document_id: int,
        summary_level: int,
        summary_text: str,
        summary_embedding: Optional[List[float]] = None,
        section_identifier: Optional[str] = None,
        parent_summary_id: Optional[int] = None,
        summary_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Create a document summary."""
        try:
            # Merge default metadata with provided metadata
            metadata = {
                "level": summary_level,
                "section_id": section_identifier
            }
            if summary_metadata:
                metadata.update(summary_metadata)
                
            summary = DocumentSummary(
                document_id=document_id,
                summary_text=summary_text,
                summary_type="section" if section_identifier else "document",
                parent_summary_id=parent_summary_id,
                summary_metadata=metadata,
                embedding=summary_embedding
            )
            self.session.add(summary)
            await self.session.flush()  # Get ID without committing
            return summary.id
        except Exception as e:
            await self.session.rollback()
            raise

    async def update_summary_embedding(
        self,
        summary_id: int,
        embedding: List[float]
    ) -> None:
        """Update the embedding for a document summary."""
        try:
            summary = await self.session.get(DocumentSummary, summary_id)
            if summary:
                summary.embedding = embedding
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise

    async def create_summary_hierarchy(
        self,
        parent_summary_id: int,
        child_summary_id: int,
        relationship_type: str
    ) -> None:
        """Create a relationship between summaries."""
        try:
            child_summary = await self.session.get(DocumentSummary, child_summary_id)
            if child_summary:
                child_summary.parent_summary_id = parent_summary_id
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise

class SQLCitationRepository(CitationRepository):
    """SQLAlchemy implementation of CitationRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_citation(
        self,
        document_id: int,
        chunk_id: Optional[int],
        citation_text: str,
        citation_type: str,
        normalized_format: str,
        authority_level: int,
        metadata: Dict[str, Any] = {}
    ) -> int:
        """Create a new citation."""
        try:
            citation = Citation(
                document_id=document_id,
                chunk_id=chunk_id,
                citation_text=citation_text,
                citation_type=citation_type,
                normalized_format=normalized_format,
                authority_level=authority_level,
                metadata=metadata
            )
            self.session.add(citation)
            await self.session.flush()  # Get ID without committing
            return citation.id
        except Exception as e:
            await self.session.rollback()
            raise

    async def create_response_citation(
        self,
        workflow_run_id: int,
        citation_id: int,
        context_used_id: int,
        relevance_score: float
    ) -> int:
        """Create a citation usage in response."""
        try:
            response_citation = ResponseCitation(
                workflow_run_id=workflow_run_id,
                citation_id=citation_id,
                context_used_id=context_used_id,
                relevance_score=relevance_score
            )
            self.session.add(response_citation)
            await self.session.flush()  # Get ID without committing
            return response_citation.id
        except Exception as e:
            await self.session.rollback()
            raise