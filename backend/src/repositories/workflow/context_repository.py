"""Context repository implementation."""
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.workflow import ContextSet, DocumentSummary, ContextResult
from src.repositories.base import BaseRepository
from src.repositories.enums import SummaryLevel, SummaryType
from src.domain.exceptions import ContextStorageError, RepositoryError
from src.repositories.document_repository import DocumentRepository

class ContextRepository(BaseRepository[ContextSet]):
    """Repository for managing context sets and related data."""

    async def create_context_set(
        self,
        workflow_run_id: int,
        original_query_id: int,
        context_data: Dict[str, Any],
        context_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Create or update a context set."""
        try:
            # Check if context set exists for this original query
            result = await self.session.execute(
                select(ContextSet).where(
                    ContextSet.original_query_id == original_query_id
                )
            )
            existing_context_set = result.scalar_one_or_none()

            if existing_context_set:
                # Update existing context set
                existing_context_set.workflow_run_id = workflow_run_id
                existing_context_set.context_data = context_data
                existing_context_set.context_metadata = context_metadata or {}
                existing_context_set.updated_at = datetime.now()
                await self.session.flush()
                return existing_context_set.id
            else:
                # Create new context set
                context_set = ContextSet(
                    workflow_run_id=workflow_run_id,
                    original_query_id=original_query_id,
                    context_data=context_data,
                    context_metadata=context_metadata or {}
                )
                self.session.add(context_set)
                await self.session.flush()
                return context_set.id

        except Exception as e:
            raise ContextStorageError(f"Failed to create/update context set: {str(e)}")

    async def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document with their metadata and embeddings."""
        try:
            doc_repo = DocumentRepository(self.session)
            chunks = await doc_repo.get_chunks_by_doc_id(document_id)
            
            if not chunks:
                return []
                
            total_chunks = len(chunks)
            
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
            raise RepositoryError(f"Failed to get document chunks: {str(e)}")

    async def create_summary(
        self,
        document_id: int,
        summary_level: SummaryLevel,
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
                summary_type=SummaryType.SECTION if section_identifier else SummaryType.DOCUMENT,
                parent_summary_id=parent_summary_id,
                summary_metadata=metadata,
                embedding=summary_embedding
            )
            self.session.add(summary)
            await self.session.flush()
            return summary.id
        except Exception as e:
            raise RepositoryError(f"Failed to create summary: {str(e)}")

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
            raise RepositoryError(f"Failed to update summary embedding: {str(e)}")

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
            await self.session.flush()
            return context.id
        except Exception as e:
            raise RepositoryError(f"Failed to create context result: {str(e)}")