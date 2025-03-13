"""LLM service interfaces."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMInterface(ABC):
    """Interface for LLM providers."""
    
    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion from messages."""
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

class PromptTemplateInterface(ABC):
    """Interface for prompt templates."""
    
    @abstractmethod
    def get_prompt(self, prompt_type: str, language: str = "vi") -> str:
        """Get prompt template for given type and language."""
        pass

    @abstractmethod
    def format_prompt(
        self,
        prompt_type: str,
        language: str = "vi",
        **kwargs: Any
    ) -> str:
        """Format prompt template with variables."""
        pass