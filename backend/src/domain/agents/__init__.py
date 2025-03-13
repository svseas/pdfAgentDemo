"""Agent implementations for the workflow system.

This package provides a collection of specialized agents that work together
to process documents, analyze queries, and generate responses. Each agent
is responsible for a specific aspect of the workflow:

- BaseAgent: Common functionality for all agents
- CitationAgent: Citation extraction and processing
- ContextBuilderAgent: Context building and retrieval
- QueryAnalyzerAgent: Query analysis and decomposition
- QuerySynthesizerAgent: Answer synthesis and generation
- RecursiveSummarizationAgent: Document summarization

The agents follow SOLID principles and implement clean code practices:
- Single Responsibility Principle: Each agent has one primary responsibility
- Open/Closed Principle: Agents can be extended through inheritance
- Liskov Substitution: All agents properly implement the AgentInterface
- Interface Segregation: Clean separation of agent responsibilities
- Dependency Inversion: Dependencies are injected and abstracted
"""

from .base_agent import BaseAgent
from .citation_agent import CitationAgent
from .context_builder_agent import ContextBuilderAgent
from .query_analyzer_agent import QueryAnalyzerAgent
from .query_synthesizer_agent import QuerySynthesizerAgent
from .recursive_summarization_agent import RecursiveSummarizationAgent

__all__ = [
    "BaseAgent",
    "CitationAgent", 
    "ContextBuilderAgent",
    "QueryAnalyzerAgent",
    "QuerySynthesizerAgent",
    "RecursiveSummarizationAgent"
]