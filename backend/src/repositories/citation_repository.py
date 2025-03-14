"""Citation repository implementation."""
from typing import Dict, Any, Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document_processing import Citation, ResponseCitation
from src.repositories.base import BaseRepository
from src.repositories.enums import CitationType
from src.domain.exceptions import RepositoryError

class CitationRepository(BaseRepository[Citation]):
    """Repository for managing citations and their usage in responses."""

    async def create_citation(
        self,
        document_id: int,
        chunk_id: Optional[int],
        citation_text: str,
        citation_type: CitationType,
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
            await self.session.flush()
            return citation.id
        except Exception as e:
            raise RepositoryError(f"Failed to create citation: {str(e)}")

    async def create_response_citation(
        self,
        citation_id: int,
        context_used_id: int,
        relevance_score: float
    ) -> int:
        """Create a citation usage in response."""
        try:
            response_citation = ResponseCitation(
                citation_id=citation_id,
                context_used_id=context_used_id,
                relevance_score=relevance_score
            )
            self.session.add(response_citation)
            await self.session.flush()
            return response_citation.id
        except Exception as e:
            raise RepositoryError(f"Failed to create response citation: {str(e)}")

    async def get_document_citations(
        self,
        document_id: int,
        citation_type: Optional[CitationType] = None
    ) -> List[Citation]:
        """Get all citations for a document, optionally filtered by type."""
        try:
            query = select(Citation).where(Citation.document_id == document_id)
            if citation_type:
                query = query.where(Citation.citation_type == citation_type)
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Failed to get document citations: {str(e)}")

    async def get_citations_by_context(
        self,
        context_used_id: int
    ) -> List[ResponseCitation]:
        """Get all citations used for a specific context."""
        try:
            query = select(ResponseCitation).where(
                ResponseCitation.context_used_id == context_used_id
            )
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Failed to get context citations: {str(e)}")