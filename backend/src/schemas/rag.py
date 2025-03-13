from typing import List, Dict, Any, Optional
from pydantic import BaseModel

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

class SynthesisRequest(BaseModel):
    """Request for answer synthesis"""
    query: str
    analyzed_query: Dict[str, Any]
    context: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    language: str = "vietnamese"
    temperature: float = 0.7
