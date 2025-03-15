"""Enums for repository layer."""
from enum import Enum, auto

class QueryType(str, Enum):
    """Types of queries."""
    USER = "user"
    SYSTEM = "system"
    SUB_QUERY = "sub_query"

class SummaryLevel(int, Enum):
    """Summary hierarchy levels."""
    CHUNK = 1
    INTERMEDIATE = 2
    DOCUMENT = 3

class SummaryType(str, Enum):
    """Types of summaries."""
    CHUNK = "chunk"
    INTERMEDIATE = "intermediate"
    FINAL = "final"

class CitationType(str, Enum):
    """Types of citations."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    REFERENCE = "reference"

class AgentType(str, Enum):
    """Types of agents."""
    SUMMARIZATION = "summarization"
    QUERY_ANALYZER = "query_analyzer"
    CITATION = "citation"
    CONTEXT_BUILDER = "context_builder"
    QUERY_SYNTHESIZER = "query_synthesizer"