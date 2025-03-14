"""Agent step repository implementation."""
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.workflow import AgentStep
from src.repositories.base import BaseRepository
from src.repositories.enums import AgentType
from src.domain.exceptions import RepositoryError

class AgentStepRepository(BaseRepository[AgentStep]):
    """Repository for managing agent processing steps."""

    async def _create_impl(self, data: Dict[str, Any]) -> AgentStep:
        """Implement agent step creation."""
        try:
            step = AgentStep(
                workflow_run_id=data["workflow_run_id"],
                agent_type=data["agent_type"],
                sub_query_id=data.get("sub_query_id"),
                input_data=data["input_data"],
                output_data=data["output_data"],
                start_time=data["start_time"],
                status=data["status"]
            )
            self.session.add(step)
            await self.session.flush()
            return step
        except Exception as e:
            raise RepositoryError(f"Failed to create agent step: {str(e)}")

    async def _get_by_id_impl(self, id: int) -> Optional[AgentStep]:
        """Implement agent step retrieval."""
        try:
            result = await self.session.execute(
                select(AgentStep).where(AgentStep.id == id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Failed to get agent step {id}: {str(e)}")

    async def _update_impl(self, id: int, data: Dict[str, Any]) -> Optional[AgentStep]:
        """Implement agent step update."""
        try:
            step = await self.session.get(AgentStep, id)
            if step:
                for key, value in data.items():
                    setattr(step, key, value)
                return step
            return None
        except Exception as e:
            raise RepositoryError(f"Failed to update agent step {id}: {str(e)}")

    async def _delete_impl(self, id: int) -> bool:
        """Implement agent step deletion."""
        try:
            step = await self.session.get(AgentStep, id)
            if step:
                await self.session.delete(step)
                return True
            return False
        except Exception as e:
            raise RepositoryError(f"Failed to delete agent step {id}: {str(e)}")

    async def log_step(
        self,
        workflow_run_id: int,
        agent_type: AgentType,
        sub_query_id: Optional[int],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str,
        start_time: datetime
    ) -> int:
        """Log an agent processing step."""
        try:
            step = await self.create({
                "workflow_run_id": workflow_run_id,
                "agent_type": agent_type,
                "sub_query_id": sub_query_id,
                "input_data": input_data,
                "output_data": output_data,
                "start_time": start_time,
                "status": status
            })
            return step.id
        except Exception as e:
            raise RepositoryError(f"Failed to log agent step: {str(e)}")

    async def update_step_status(
        self,
        step_id: int,
        status: str,
        output_data: Dict[str, Any],
        end_time: datetime
    ) -> None:
        """Update agent step status and output."""
        try:
            await self.update(step_id, {
                "status": status,
                "output_data": output_data,
                "end_time": end_time
            })
        except Exception as e:
            raise RepositoryError(f"Failed to update agent step status: {str(e)}")