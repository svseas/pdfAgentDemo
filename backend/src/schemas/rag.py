from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

class VectorizeRequest(BaseModel):
    text: str

class VectorResponse(BaseModel):
    vector: List[float]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.5
    use_grag: bool = False

class ContextRequest(BaseModel):
    query: str
    relevant_chunks: List[Dict[str, Any]]
    temperature: float = 0.7

class QueryRequest(BaseModel):
    """Request for the complete RAG workflow"""
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.5
    temperature: float = 0.7

class SummarizeRequest(BaseModel):
    """Request for document summarization"""
    document_id: str
    max_length: Optional[int] = None
    language: str = "vietnamese"

class QueryAnalysisRequest(BaseModel):
    """Request for query analysis with stepback prompting"""
    query: str
    language: str = "vietnamese"

class CitationRequest(BaseModel):
    """Request for citation extraction"""
    document_id: str
    query: str
    language: str = "vietnamese"
class ContextBuilderRequest(BaseModel):
    """Request parameters for context building"""
    workflow_run_id: int
    sub_query_id: int
    query_text: str
    query_embedding: List[float]
    is_original: bool = False
    top_k: int = 5

    @validator('query_embedding')
    def validate_embedding_dim(cls, v):
        if len(v) != 768:
            raise ValueError("Query embedding must be 768-dimensional")
        return v

class ContextChunkResponse(BaseModel):
    """Response format for a single context chunk"""
    id: int
    document_id: int
    document_name: str
    chunk_index: int
    is_direct_match: bool
    relevance_score: float
    text: str

class ContextBuilderResponse(BaseModel):
    """Response format for context building results"""
    status: str = "success"
    workflow_run_id: int
    context_set_id: int
    original_query: str
    sub_queries: List[Dict[str, Any]]
    context: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_chunks": 0,
            "total_tokens": 0,
            "chunks": []
        }
    )

class BuildQueryContextRequest(BaseModel):
    """Request for building context from original query and its sub-queries"""
    original_query_id: int
    top_k: int = 5
    similarity_threshold: float = 0.5

class SynthesisRequest(BaseModel):
    """Request for answer synthesis"""
    query: str
    analyzed_query: Dict[str, Any]
    context: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    language: str = "vietnamese"
    temperature: float = 0.7

