"""Query repository implementation."""
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document_processing import OriginalUserQuery, SubQuery
from src.repositories.base import BaseRepository
from src.domain.exceptions import RepositoryError

class QueryRepository(BaseRepository[OriginalUserQuery]):
    """Repository for managing original queries and sub-queries."""

    async def _create_impl(self, data: Dict[str, Any]) -> OriginalUserQuery:
        """Implement query creation."""
        try:
            query = OriginalUserQuery(**data)
            self.session.add(query)
            await self.session.flush()
            return query
        except Exception as e:
            raise RepositoryError(f"Failed to create query: {str(e)}")

    async def _get_by_id_impl(self, id: int) -> Optional[OriginalUserQuery]:
        """Implement query retrieval."""
        try:
            result = await self.session.execute(
                select(OriginalUserQuery).where(OriginalUserQuery.id == id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Failed to get query {id}: {str(e)}")

    async def _update_impl(self, id: int, data: Dict[str, Any]) -> Optional[OriginalUserQuery]:
        """Implement query update."""
        try:
            query = await self.session.get(OriginalUserQuery, id)
            if query:
                for key, value in data.items():
                    setattr(query, key, value)
                return query
            return None
        except Exception as e:
            raise RepositoryError(f"Failed to update query {id}: {str(e)}")

    async def _delete_impl(self, id: int) -> bool:
        """Implement query deletion."""
        try:
            query = await self.session.get(OriginalUserQuery, id)
            if query:
                await self.session.delete(query)
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to delete query {id}: {str(e)}")

    async def update_query_embedding(
        self,
        query_id: int,
        embedding: np.ndarray
    ) -> None:
        """Update embedding for a query."""
        try:
            query = await self._get_by_id_impl(query_id)
            if query:
                query.query_embedding = embedding
                query.updated_at = datetime.now()
                await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to update query embedding: {str(e)}")

    async def create_sub_query(
        self,
        original_query_id: int,
        sub_query_text: str,
        sub_query_embedding: Optional[np.ndarray] = None
    ) -> int:
        """Create a new sub query with optional embedding."""
        try:
            sub_query = SubQuery(
                original_query_id=original_query_id,
                sub_query_text=sub_query_text,
                sub_query_embedding=sub_query_embedding,
                created_at=datetime.now()
            )
            self.session.add(sub_query)
            await self.session.commit()
            await self.session.refresh(sub_query)
            return sub_query.id
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to create sub query: {str(e)}")

    async def get_sub_queries(
        self,
        original_query_id: int
    ) -> List[SubQuery]:
        """Get all sub-queries for an original query."""
        try:
            result = await self.session.execute(
                select(SubQuery).where(
                    SubQuery.original_query_id == original_query_id
                )
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Failed to get sub queries: {str(e)}")

    async def get_query_by_text(
        self,
        query_text: str
    ) -> Optional[OriginalUserQuery]:
        """Get original query by exact text match."""
        try:
            result = await self.session.execute(
                select(OriginalUserQuery).where(
                    OriginalUserQuery.query_text == query_text
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Failed to get query by text: {str(e)}")