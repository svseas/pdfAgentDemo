from typing import List, Dict, Any
from pydantic import BaseModel

class VectorizeRequest(BaseModel):
    text: str

class VectorResponse(BaseModel):
    vector: List[float]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.5

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
