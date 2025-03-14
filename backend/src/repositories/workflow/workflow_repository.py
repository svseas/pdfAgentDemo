"""Workflow repository implementation."""
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.workflow import WorkflowRun
from src.repositories.base import BaseRepository
from src.repositories.enums import WorkflowStatus
from src.domain.exceptions import RepositoryError

class WorkflowRepository(BaseRepository[WorkflowRun]):
    """Repository for managing workflow runs."""

    async def _create_impl(self, data: Dict[str, Any]) -> WorkflowRun:
        """Implement workflow creation."""
        try:
            workflow = WorkflowRun(
                user_query_id=data["user_query_id"],
                status=data.get("status", WorkflowStatus.RUNNING)
            )
            self.session.add(workflow)
            await self.session.flush()
            return workflow
        except Exception as e:
            raise RepositoryError(f"Failed to create workflow: {str(e)}")

    async def _get_by_id_impl(self, id: int) -> Optional[WorkflowRun]:
        """Implement workflow retrieval."""
        try:
            result = await self.session.execute(
                select(WorkflowRun).where(WorkflowRun.id == id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Failed to get workflow {id}: {str(e)}")

    async def _update_impl(self, id: int, data: Dict[str, Any]) -> Optional[WorkflowRun]:
        """Implement workflow update."""
        try:
            workflow = await self.session.get(WorkflowRun, id)
            if workflow:
                for key, value in data.items():
                    setattr(workflow, key, value)
                return workflow
            return None
        except Exception as e:
            raise RepositoryError(f"Failed to update workflow {id}: {str(e)}")

    async def _delete_impl(self, id: int) -> bool:
        """Implement workflow deletion."""
        try:
            workflow = await self.session.get(WorkflowRun, id)
            if workflow:
                await self.session.delete(workflow)
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to delete workflow {id}: {str(e)}")

    async def create_workflow_run(
        self,
        user_query_id: int,
        status: WorkflowStatus = WorkflowStatus.RUNNING
    ) -> int:
        """Create a new workflow run."""
        try:
            workflow = await self.create({
                "user_query_id": user_query_id,
                "status": status
            })
            return workflow.id
        except Exception as e:
            raise RepositoryError(f"Failed to create workflow run: {str(e)}")

    async def update_status(
        self,
        workflow_id: int,
        status: WorkflowStatus,
        final_answer: Optional[str] = None
    ) -> None:
        """Update workflow status and final answer."""
        try:
            data = {"status": status}
            if final_answer is not None:
                data["final_answer"] = final_answer
            if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                data["completed_at"] = datetime.now()
            
            await self.update(workflow_id, data)
            await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to update workflow status: {str(e)}")