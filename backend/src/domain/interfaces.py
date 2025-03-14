"""Domain interfaces and abstract base classes."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime
import numpy as np

class TextSplitterInterface(ABC):
    """Interface for text splitting strategies."""
    
    @abstractmethod
    async def split(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

class EmbeddingGeneratorInterface(ABC):
    """Interface for embedding generation."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass
        
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass

class LLMInterface(ABC):
    """Interface for LLM interactions."""
    
    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Generate completion from LLM."""
        pass
        
    @abstractmethod
    def get_prompt(self, prompt_type: str, language: str = "vi") -> str:
        """Get prompt template."""
        pass

class DocumentProcessorInterface(ABC):
    """Interface for document processing."""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text from document."""
        pass
        
    @abstractmethod
    async def process_text(self, text: str) -> List[str]:
        """Process extracted text into chunks."""
        pass

class AgentInterface(Protocol):
    """Base interface for all agents in the system."""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return output."""
        ...

    async def log_step(
        self,
        workflow_run_id: int,
        sub_query_id: Optional[int],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str
    ) -> int:
        """Log an agent step to the database."""
        ...

class BaseRepository(Protocol):
    """Base interface for repositories."""
    
    async def create(self, data: Dict[str, Any]) -> Any:
        """Create a new record."""
        ...
        
    async def get_by_id(self, id: int) -> Optional[Any]:
        """Get a record by ID."""
        ...
        
    async def update(self, id: int, data: Dict[str, Any]) -> Any:
        """Update a record."""
        ...
        
    async def delete(self, id: int) -> bool:
        """Delete a record."""
        ...

class WorkflowRepository(BaseRepository):
    """Interface for workflow tracking repository."""
    
    async def create_workflow_run(
        self,
        user_query_id: int,
        status: str = "running"
    ) -> int:
        """Create a new workflow run."""
        ...
        
    async def update_workflow_status(
        self,
        workflow_id: int,
        status: str,
        final_answer: Optional[str] = None
    ) -> None:
        """Update workflow status and final answer."""
        ...
        
    async def get_workflow_details(
        self,
        workflow_id: int
    ) -> Dict[str, Any]:
        """Get complete workflow details including steps and sub-queries."""
        ...

class QueryRepository(BaseRepository):
    """Interface for query tracking repository."""
    
    async def create_user_query(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None
    ) -> int:
        """Create a new user query."""
        ...
        
    async def create_sub_query(
        self,
        workflow_run_id: int,
        original_query_id: int,
        sub_query_text: str,
        sub_query_embedding: Optional[List[float]] = None
    ) -> int:
        """Create a new sub-query."""
        ...

class AgentStepRepository(BaseRepository):
    """Interface for agent step tracking repository."""
    
    async def log_agent_step(
        self,
        workflow_run_id: int,
        agent_type: str,
        sub_query_id: Optional[int],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        status: str,
        start_time: datetime
    ) -> int:
        """Log an agent processing step."""
        ...
        
    async def update_step_status(
        self,
        step_id: int,
        status: str,
        output_data: Dict[str, Any],
        end_time: datetime
    ) -> None:
        """Update agent step status and output."""
        ...

class ContextRepository(BaseRepository):
    """Interface for context result tracking repository."""
    
    async def create_context_result(
        self,
        agent_step_id: int,
        document_id: int,
        chunk_id: Optional[int],
        summary_id: Optional[int],
        relevance_score: float,
        used_in_response: bool = False
    ) -> int:
        """Create a new context result."""
        ...
        
    async def mark_context_used(
        self,
        context_id: int,
        used: bool = True
    ) -> None:
        """Mark a context as used in response."""
        ...
        
    async def create_summary(
        self,
        document_id: int,
        summary_level: int,
        summary_text: str,
        section_identifier: Optional[str] = None,
        parent_summary_id: Optional[int] = None,
        summary_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Create a document summary."""
        ...
        
    async def create_summary_hierarchy(
        self,
        parent_summary_id: int,
        child_summary_id: int,
        relationship_type: str
    ) -> None:
        """Create a relationship between summaries."""
        ...

class CitationRepository(BaseRepository):
    """Interface for citation tracking repository."""
    
    async def create_citation(
        self,
        document_id: int,
        chunk_id: Optional[int],
        citation_text: str,
        citation_type: str,
        normalized_format: str,
        authority_level: int,
        metadata: Dict[str, Any] = {}
    ) -> int:
        """Create a new citation."""
        ...
        
    async def create_response_citation(
        self,
        workflow_run_id: int,
        citation_id: int,
        context_used_id: int,
        relevance_score: float
    ) -> int:
        """Create a citation usage in response."""
        ...

class QueryProcessorInterface(ABC):
    """Interface for query processing."""
    
    @abstractmethod
    def get_relevant_chunks(
        self,
        query: str,
        doc_chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant document chunks for query."""
        pass
        
    @abstractmethod
    async def generate_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        language: str = "vi"
    ) -> str:
        """Generate response from relevant chunks."""
        pass