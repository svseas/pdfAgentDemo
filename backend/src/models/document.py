from datetime import datetime
from typing import Optional
from sqlalchemy import Index, String, Integer, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .base import Base


class Document(Base):
    """Document chunk with vector embedding"""
    __tablename__ = "documents"

    # Override id from base class for explicit typing
    id: Mapped[int] = mapped_column(primary_key=True)
    
    # Document chunk fields
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(768),
        nullable=True,
        comment="Text embedding vector for similarity search"
    )

    # Relationships
    doc_metadata_id: Mapped[int] = mapped_column(
        ForeignKey("document_metadata.id", ondelete="CASCADE"),
        nullable=False
    )
    doc_metadata: Mapped["DocumentMetadata"] = relationship(
        back_populates="chunks",
        lazy="joined"
    )

    __table_args__ = (
        # Cosine similarity search index
        Index(
            "ix_documents_embedding_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        # Unique constraint for filename and chunk combination
        Index(
            "ix_documents_filename_chunk",
            "filename",
            "chunk_index",
            unique=True
        ),
    )


class DocumentMetadata(Base):
    """Metadata for uploaded documents"""
    __tablename__ = "document_metadata"

    # Override id from base class for explicit typing
    id: Mapped[int] = mapped_column(primary_key=True)
    
    # Metadata fields
    filename: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    total_chunks: Mapped[int] = mapped_column(nullable=False)
    file_size: Mapped[int] = mapped_column(
        nullable=False,
        comment="File size in bytes"
    )
    mime_type: Mapped[str] = mapped_column(
        String(127),
        nullable=False,
        comment="MIME type of the document"
    )
    processing_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        comment="Status of document processing: pending, processing, completed, failed"
    )

    # Relationships
    chunks: Mapped[list["Document"]] = relationship(
        back_populates="doc_metadata",
        cascade="all, delete-orphan",
        lazy="selectin"
    )