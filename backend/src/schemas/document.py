from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, FilePath


class DocumentBase(BaseModel):
    """Base schema for document metadata"""
    filename: str = Field(..., min_length=1, max_length=255)
    title: Optional[str] = Field(None, max_length=255)


class DocumentMetadataCreate(DocumentBase):
    """Schema for creating document metadata"""
    file_size: int = Field(..., gt=0, description="File size in bytes")
    mime_type: str = Field(..., max_length=127)


class DocumentMetadataRead(DocumentBase):
    """Schema for reading document metadata"""
    id: int
    total_chunks: int = Field(..., ge=0)
    file_size: int
    mime_type: str
    processing_status: Literal["pending", "processing", "completed", "failed"]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentChunkBase(BaseModel):
    """Base schema for document chunks"""
    filename: str = Field(..., min_length=1, max_length=255)
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)


class DocumentChunkCreate(DocumentChunkBase):
    """Schema for creating document chunks"""
    doc_metadata_id: int = Field(..., gt=0)
    embedding: Optional[list[float]] = Field(None, description="Vector embedding of chunk content")


class DocumentChunkRead(DocumentChunkBase):
    """Schema for reading document chunks"""
    id: int
    doc_metadata_id: int
    embedding: Optional[list[float]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentUpload(BaseModel):
    """Schema for document upload request"""
    file: FilePath = Field(..., description="PDF file to upload")
    title: Optional[str] = Field(None, max_length=255)


class DocumentProcessingStatus(BaseModel):
    """Schema for document processing status response"""
    filename: str
    status: Literal["pending", "processing", "completed", "failed"]
    total_chunks: Optional[int] = None
    error_message: Optional[str] = None
    updated_at: datetime


class DocumentSearchQuery(BaseModel):
    """Schema for document search request"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1)"
    )


class DocumentSearchResult(BaseModel):
    """Schema for document search result"""
    chunk_id: int
    content: str
    similarity_score: float
    doc_metadata: DocumentMetadataRead


class DocumentEmbedRequest(BaseModel):
    """Schema for document embedding request"""
    document_id: int = Field(..., gt=0, description="ID of the document to embed")
    force: bool = Field(default=False, description="Whether to force update existing embeddings")