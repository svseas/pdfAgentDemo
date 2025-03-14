"""Base repository implementation."""
from typing import Any, Dict, Optional, TypeVar, Generic
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def _handle_transaction(self, operation: str) -> None:
        """Handle transaction based on operation result."""
        if operation == "commit":
            await self.session.commit()
        elif operation == "rollback":
            await self.session.rollback()

    async def create(self, data: Dict[str, Any]) -> T:
        """Create a new record."""
        try:
            result = await self._create_impl(data)
            await self._handle_transaction("commit")
            return result
        except Exception as e:
            await self._handle_transaction("rollback")
            raise

    async def get_by_id(self, id: int) -> Optional[T]:
        """Get record by ID."""
        try:
            return await self._get_by_id_impl(id)
        except Exception as e:
            await self._handle_transaction("rollback")
            raise

    async def update(self, id: int, data: Dict[str, Any]) -> Optional[T]:
        """Update a record."""
        try:
            result = await self._update_impl(id, data)
            if result:
                await self._handle_transaction("commit")
            return result
        except Exception as e:
            await self._handle_transaction("rollback")
            raise

    async def delete(self, id: int) -> bool:
        """Delete a record."""
        try:
            result = await self._delete_impl(id)
            if result:
                await self._handle_transaction("commit")
            return result
        except Exception as e:
            await self._handle_transaction("rollback")
            raise

    # Implementation methods to be overridden by concrete repositories
    async def _create_impl(self, data: Dict[str, Any]) -> T:
        raise NotImplementedError

    async def _get_by_id_impl(self, id: int) -> Optional[T]:
        raise NotImplementedError

    async def _update_impl(self, id: int, data: Dict[str, Any]) -> Optional[T]:
        raise NotImplementedError

    async def _delete_impl(self, id: int) -> bool:
        raise NotImplementedError