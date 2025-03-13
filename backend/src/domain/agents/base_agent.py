"""Base agent implementation and common agent functionality."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from src.domain.interfaces import AgentInterface
from src.core.llm.interfaces import LLMInterface, PromptTemplateInterface
from src.repositories.workflow_repository import AgentStepRepository
from src.domain.exceptions import AgentError

class AgentLogger(Protocol):
    """Protocol for agent step logging functionality."""
    
    async def log_step(
        self,
        workflow_run_id: int,
        sub_query_id: Optional[int],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str
    ) -> int:
        """Log an agent step."""
        ...

    async def update_step_status(
        self,
        step_id: int,
        status: str,
        output_data: Dict[str, Any]
    ) -> None:
        """Update agent step status."""
        ...

class BaseAgentLogger:
    """Default implementation of agent logging functionality."""
    
    def __init__(self, agent_step_repo: AgentStepRepository):
        self.agent_step_repo = agent_step_repo

    async def log_step(
        self,
        workflow_run_id: int,
        sub_query_id: Optional[int],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str,
        agent_type: str
    ) -> int:
        """Log an agent processing step."""
        try:
            return await self.agent_step_repo.log_agent_step(
                workflow_run_id=workflow_run_id,
                agent_type=agent_type,
                sub_query_id=sub_query_id,
                input_data=input_data,
                output_data=output_data,
                status=status,
                start_time=datetime.utcnow()
            )
        except Exception as e:
            raise AgentError(f"Failed to log agent step: {str(e)}") from e

    async def update_step_status(
        self,
        step_id: int,
        status: str,
        output_data: Dict[str, Any]
    ) -> None:
        """Update agent step status."""
        try:
            await self.agent_step_repo.update_step_status(
                step_id,
                status,
                output_data,
                datetime.utcnow()
            )
        except Exception as e:
            raise AgentError(f"Failed to update agent step status: {str(e)}") from e

class BaseAgent(AgentInterface, ABC):
    """Base implementation for all agents.
    
    This class provides common functionality for all agents including:
    - Dependency injection for common services
    - Step logging and status tracking
    - Error handling
    - Base process flow
    
    Attributes:
        session: Database session
        llm: Language model interface
        prompt_manager: Prompt template manager
        logger: Agent step logger
    """
    
    def __init__(
        self,
        session: AsyncSession,
        agent_step_repo: AgentStepRepository,
        llm: Optional[LLMInterface] = None,
        prompt_manager: Optional[PromptTemplateInterface] = None
    ):
        """Initialize agent with required dependencies.
        
        Args:
            session: Database session for transactions
            agent_step_repo: Repository for logging agent steps
            llm: Optional language model interface
            prompt_manager: Optional prompt template manager
        """
        self.session = session
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.logger = BaseAgentLogger(agent_step_repo)

    @abstractmethod
    async def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of agent-specific processing logic.
        
        This method should be implemented by concrete agent classes.
        
        Args:
            input_data: Input data for processing, including:
                - workflow_run_id: ID of current workflow run
                - agent_step_id: ID of current agent step
                - Other agent-specific parameters
            
        Returns:
            Dict containing processing results
            
        Raises:
            AgentError: If processing fails
        """
        pass

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data with logging and error handling.
        
        This method implements the template method pattern, handling:
        - Step logging
        - Error handling
        - Status updates
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dict containing processing results
            
        Raises:
            AgentError: If processing fails
        """
        workflow_run_id = input_data.get("workflow_run_id")
        sub_query_id = input_data.get("sub_query_id")
        
        # Log step start
        try:
            step_id = await self.logger.log_step(
                workflow_run_id=workflow_run_id,
                sub_query_id=sub_query_id,
                input_data=input_data,
                output_data={},
                status="running",
                agent_type=self.__class__.__name__
            )
            
            # Add step_id to input data for _process_impl
            input_data["agent_step_id"] = step_id
            
        except AgentError as e:
            raise AgentError(f"Failed to start agent processing: {str(e)}")
        
        try:
            # Execute agent-specific processing
            output_data = await self._process_impl(input_data)
            
            # Update step status on success
            await self.logger.update_step_status(
                step_id=step_id,
                status="success",
                output_data=output_data
            )
            
            return output_data
            
        except Exception as e:
            # Update step status on failure
            error_data = {"error": str(e)}
            await self.logger.update_step_status(
                step_id=step_id,
                status="failed",
                output_data=error_data
            )
            
            raise AgentError(f"Agent processing failed: {str(e)}") from e

    async def log_step(
        self,
        workflow_run_id: int,
        sub_query_id: Optional[int],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str
    ) -> int:
        """Log an agent step (implementation of AgentInterface)."""
        return await self.logger.log_step(
            workflow_run_id=workflow_run_id,
            sub_query_id=sub_query_id,
            input_data=input_data,
            output_data=output_data,
            status=status,
            agent_type=self.__class__.__name__
        )