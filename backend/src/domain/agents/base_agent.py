"""Base agent implementation."""
from typing import Dict, Any, Optional, Protocol
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from src.domain.exceptions import AgentError
from src.core.llm.interfaces import LLMInterface
from src.core.llm.prompts import PromptManager

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents.
    
    This class provides common functionality for:
    - Database session management
    - LLM interaction
    - Prompt management
    - Error handling
    
    All agents should inherit from this class and implement
    the _process_impl method.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        llm: Optional[LLMInterface] = None,
        prompt_manager: Optional[PromptManager] = None,
        *args,
        **kwargs
    ):
        """Initialize base agent.
        
        Args:
            session: Database session
            llm: Optional language model interface
            prompt_manager: Optional prompt manager
            *args, **kwargs: Additional arguments for specific agents
        """
        self.session = session
        self.llm = llm
        self.prompt_manager = prompt_manager

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data using agent logic.
        
        This method:
        1. Validates input data
        2. Calls agent-specific implementation
        3. Handles transaction management
        4. Provides error handling
        
        Args:
            input_data: Data to process
            
        Returns:
            Dict containing processing results
            
        Raises:
            AgentError: If processing fails
        """
        try:
            # Validate input
            if not isinstance(input_data, dict):
                raise AgentError("Input data must be a dictionary")

            # Process using agent implementation
            result = await self._process_impl(input_data)
            
            # Commit any pending changes
            await self.session.commit()
            
            return result

        except AgentError:
            await self.session.rollback()
            raise
        except Exception as e:
            await self.session.rollback()
            raise AgentError(f"Agent processing failed: {str(e)}") from e

    async def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-specific implementation.
        
        This method must be implemented by each agent class.
        
        Args:
            input_data: Data to process
            
        Returns:
            Dict containing processing results
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Agents must implement _process_impl")