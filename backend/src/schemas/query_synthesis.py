"""Schemas for query synthesis."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Context(BaseModel):
    """Context data structure."""
    total_chunks: int
    total_tokens: int
    chunks: List[Dict[str, Any]]

class SubQuery(BaseModel):
    """Sub-query data structure."""
    id: int
    text: str
    is_original: bool