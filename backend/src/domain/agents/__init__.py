"""Agent implementations for document processing and query handling.

This package contains specialized agents that work together to process documents,
analyze queries, and generate responses. Each agent is responsible for a specific
aspect of the processing pipeline:

- RecursiveSummarizationAgent: Generates hierarchical document summaries
- QueryAnalyzerAgent: Analyzes and breaks down complex queries
- ContextBuilderAgent: Builds relevant context for query processing
- CitationAgent: Extracts and manages document citations
- QuerySynthesizerAgent: Synthesizes final responses using context and citations

Each agent follows the BaseAgent interface and can be composed together to form
more complex processing pipelines.
"""

from .base_agent import BaseAgent
from .recursive_summarization_agent import RecursiveSummarizationAgent
from .query_analyzer_agent import QueryAnalyzerAgent
from .context_builder_agent import ContextBuilderAgent
from .citation_agent import CitationAgent
from .query_synthesizer_agent import QuerySynthesizerAgent

__all__ = [
    "BaseAgent",
    "RecursiveSummarizationAgent",
    "QueryAnalyzerAgent",
    "ContextBuilderAgent",
    "CitationAgent",
    "QuerySynthesizerAgent"
]