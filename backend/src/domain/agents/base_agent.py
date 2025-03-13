"""Base agent implementation."""
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.interfaces import AgentInterface
from src.core.llm.interfaces import LLMInterface, PromptTemplateInterface
from src.repositories.workflow_repository import SQLAgentStepRepository

class BaseAgent(AgentInterface):
    """Base implementation for all agents."""
    
    def __init__(
        self,
        session: AsyncSession,
        llm: Optional[LLMInterface] = None,
        prompt_manager: Optional[PromptTemplateInterface] = None
    ):
        self.session = session
        self.llm = llm
        self.prompt_manager = prompt_manager
    
    async def log_step(
        self,
        workflow_run_id: int,
        sub_query_id: Optional[int],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str
    ) -> int:
        """Log an agent step to the database."""
        repo = SQLAgentStepRepository(self.session)
        return await repo.log_agent_step(
            workflow_run_id=workflow_run_id,
            agent_type=self.__class__.__name__,
            sub_query_id=sub_query_id,
            input_data=input_data,
            output_data=output_data,
            status=status,
            start_time=datetime.utcnow()
        )

    async def _update_step_status(
        self,
        step_id: int,
        status: str,
        output_data: Dict[str, Any]
    ) -> None:
        """Update agent step status."""
        repo = SQLAgentStepRepository(self.session)
        await repo.update_step_status(
            step_id,
            status,
            output_data,
            datetime.utcnow()
        )