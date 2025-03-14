"""Query repository implementation."""
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.workflow import UserQuery, OriginalUserQuery, SubQuery
from src.repositories.base import BaseRepository
from src.repositories.enums import QueryType
from src.domain.exceptions import RepositoryError

class QueryRepository(BaseRepository[UserQuery]):
    """Repository for managing user queries and sub-queries."""

    async def _create_impl(self, data: Dict[str, Any]) -> UserQuery:
        """Implement query creation."""
        try:
            query = UserQuery(**data)
            self.session.add(query)
            await self.session.flush()
            return query
        except Exception as e:
            raise RepositoryError(f"Failed to create query: {str(e)}")

    async def _get_by_id_impl(self, id: int) -> Optional[UserQuery]:
        """Implement query retrieval."""
        try:
            result = await self.session.execute(
                select(UserQuery)
                .options(joinedload(UserQuery.original_query))
                .where(UserQuery.id == id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Failed to get query {id}: {str(e)}")

    async def _update_impl(self, id: int, data: Dict[str, Any]) -> Optional[UserQuery]:
        """Implement query update."""
        try:
            query = await self.session.get(UserQuery, id)
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
            query = await self.session.get(UserQuery, id)
            if query:
                await self.session.delete(query)
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to delete query {id}: {str(e)}")

    async def create_user_query(
        self,
        query_text: str,
        query_type: QueryType = QueryType.USER
    ) -> int:
        """Create a new user query and original query if needed."""
        try:
            if query_type == QueryType.SYSTEM:
                # For system queries, just create a user query without original reference
                query = await self.create({
                    "query_text": query_text,
                    "created_at": datetime.now()
                })
                return query.id

            # For user queries, check if original exists
            result = await self.session.execute(
                select(OriginalUserQuery).where(
                    OriginalUserQuery.query_text == query_text
                )
            )
            original_query = result.scalar_one_or_none()

            # Create original query if it doesn't exist
            if not original_query:
                original_query = OriginalUserQuery(
                    query_text=query_text,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                self.session.add(original_query)
                await self.session.commit()
                await self.session.refresh(original_query)

            # Create user query with reference to original
            query = await self.create({
                "query_text": query_text,
                "original_query_id": original_query.id,
                "created_at": datetime.now()
            })
            return query.id

        except Exception as e:
            raise RepositoryError(f"Failed to create user query: {str(e)}")

    async def update_query_embedding(
        self,
        query_id: int,
        embedding: np.ndarray
    ) -> None:
        """Update embedding for a query and its original query."""
        try:
            # Get the query with original query loaded
            query = await self._get_by_id_impl(query_id)
            if not query:
                return

            # Update query embedding
            query.query_embedding = embedding
            
            # Update original query embedding if not already set
            if query.original_query and query.original_query.query_embedding is None:
                query.original_query.query_embedding = embedding
                query.original_query.updated_at = datetime.now()

            await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to update query embedding: {str(e)}")

    async def create_sub_query(
        self,
        workflow_run_id: int,
        original_query_id: Optional[int],
        sub_query_text: str,
        sub_query_embedding: Optional[np.ndarray] = None
    ) -> int:
        """Create a new sub query with optional embedding."""
        try:
            sub_query = SubQuery(
                workflow_run_id=workflow_run_id,
                original_query_id=original_query_id,
                sub_query_text=sub_query_text,
                sub_query_embedding=sub_query_embedding
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