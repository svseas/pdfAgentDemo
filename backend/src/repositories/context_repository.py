"""Context repository implementation."""
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document_processing import ContextSet, DocumentSummary, ContextResult
from src.repositories.base import BaseRepository
from src.repositories.enums import SummaryLevel, SummaryType
from src.domain.exceptions import ContextStorageError, RepositoryError
from src.repositories.document_repository import DocumentRepository

class ContextRepository(BaseRepository[ContextSet]):
    """Repository for managing context sets and results."""

    async def create_context_set(
        self,
        original_query_id: int,
        context_data: Dict[str, Any],
        context_metadata: Dict[str, Any]
    ) -> int:
        """Create a new context set."""
        try:
            context_set = ContextSet(
                original_query_id=original_query_id,
                context_data=context_data,
                context_metadata=context_metadata,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.session.add(context_set)
            await self.session.flush()
            return context_set.id
        except Exception as e:
            raise ContextStorageError(f"Failed to create context set: {str(e)}")

    async def create_context_result(
        self,
        document_id: int,
        chunk_id: Optional[int],
        summary_id: Optional[int],
        relevance_score: float,
        used_in_response: bool = False
    ) -> int:
        """Create a new context result."""
        try:
            context_result = ContextResult(
                document_id=document_id,
                chunk_id=chunk_id,
                summary_id=summary_id,
                relevance_score=relevance_score,
                used_in_response=used_in_response
            )
            self.session.add(context_result)
            await self.session.flush()
            return context_result.id
        except Exception as e:
            raise ContextStorageError(f"Failed to create context result: {str(e)}")

    async def get_context_set(
        self,
        context_set_id: int
    ) -> Optional[ContextSet]:
        """Get a context set by ID."""
        try:
            result = await self.session.execute(
                select(ContextSet).where(ContextSet.id == context_set_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Failed to get context set: {str(e)}")

    async def get_query_context_sets(
        self,
        original_query_id: int
    ) -> List[ContextSet]:
        """Get all context sets for a query."""
        try:
            result = await self.session.execute(
                select(ContextSet).where(
                    ContextSet.original_query_id == original_query_id
                ).order_by(ContextSet.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Failed to get query context sets: {str(e)}")

    async def get_context_results(
        self,
        context_set_id: int
    ) -> List[ContextResult]:
        """Get all context results for a context set."""
        try:
            result = await self.session.execute(
                select(ContextResult).where(
                    ContextResult.context_set_id == context_set_id
                ).order_by(ContextResult.relevance_score.desc())
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Failed to get context results: {str(e)}")

    async def update_context_result(
        self,
        context_result_id: int,
        used_in_response: bool
    ) -> None:
        """Update whether a context result was used in response."""
        try:
            context_result = await self.session.get(ContextResult, context_result_id)
            if context_result:
                context_result.used_in_response = used_in_response
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to update context result: {str(e)}")

    async def get_document_summaries(
        self,
        document_id: int,
        summary_type: Optional[SummaryType] = None
    ) -> List[DocumentSummary]:
        """Get all summaries for a document."""
        try:
            query = select(DocumentSummary).where(
                DocumentSummary.document_id == document_id
            )
            if summary_type:
                query = query.where(DocumentSummary.summary_type == summary_type)
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Failed to get document summaries: {str(e)}")

    async def create_document_summary(
        self,
        document_id: int,
        summary_text: str,
        summary_type: SummaryType,
        parent_summary_id: Optional[int] = None,
        summary_metadata: Dict[str, Any] = {},
        embedding: Optional[List[float]] = None
    ) -> int:
        """Create a new document summary."""
        try:
            summary = DocumentSummary(
                document_id=document_id,
                summary_text=summary_text,
                summary_type=summary_type,
                parent_summary_id=parent_summary_id,
                summary_metadata=summary_metadata,
                embedding=embedding,
                created_at=datetime.now()
            )
            self.session.add(summary)
            await self.session.flush()
            return summary.id
        except Exception as e:
            raise RepositoryError(f"Failed to create document summary: {str(e)}")
            
    async def update_summary_embedding(
        self,
        summary_id: int,
        embedding: List[float]
    ) -> None:
        """Update embedding for a document summary.
        
        Args:
            summary_id: ID of summary to update
            embedding: New embedding vector
            
        Raises:
            RepositoryError: If update fails
        """
        try:
            summary = await self.session.get(DocumentSummary, summary_id)
            if summary:
                summary.embedding = embedding.tolist() if embedding is not None else None
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to update summary embedding: {str(e)}")