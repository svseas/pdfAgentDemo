"""SQLAlchemy models for query tracking and document processing."""
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Index
from sqlalchemy.orm import relationship, mapped_column, Mapped
from pgvector.sqlalchemy import Vector

from .base import Base

class OriginalUserQuery(Base):
    """Model for original user queries."""
    __tablename__ = "original_user_queries"

    id = Column(Integer, primary_key=True)
    query_text = Column(String, nullable=False)
    query_embedding: Mapped[list[float]] = mapped_column(
        Vector(768),
        nullable=True,
        comment="Query embedding vector for similarity search"
    )
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationships
    sub_queries = relationship("SubQuery", back_populates="original_query")

    __table_args__ = (
        # Cosine similarity search index
        Index(
            "ix_original_user_queries_embedding_cosine",
            "query_embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"query_embedding": "vector_cosine_ops"}
        ),
    )

class SubQuery(Base):
    """Model for sub-queries generated by query analyzer."""
    __tablename__ = "sub_queries"

    id = Column(Integer, primary_key=True)
    original_query_id = Column(Integer, ForeignKey("original_user_queries.id"), nullable=False)
    sub_query_text = Column(String, nullable=False)
    sub_query_embedding: Mapped[list[float]] = mapped_column(
        Vector(768),
        nullable=True,
        comment="Sub-query embedding vector for similarity search"
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    original_query = relationship("OriginalUserQuery", back_populates="sub_queries")

    __table_args__ = (
        # Cosine similarity search index
        Index(
            "ix_sub_queries_embedding_cosine",
            "sub_query_embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"sub_query_embedding": "vector_cosine_ops"}
        ),
    )

class DocumentSummary(Base):
    """Model for document summaries."""
    __tablename__ = "document_summaries"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("document_metadata.id"), nullable=False)
    summary_text = Column(String, nullable=False)
    summary_type = Column(String, nullable=False)  # "section", "chapter", "full"
    parent_summary_id = Column(Integer, ForeignKey("document_summaries.id"))
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    summary_metadata = Column(JSON, nullable=False, default={})
    embedding: Mapped[list[float]] = mapped_column(
        Vector(768),
        nullable=True,
        comment="Text embedding vector for similarity search"
    )

    # Relationships
    document = relationship("DocumentMetadata")
    parent_summary = relationship("DocumentSummary", remote_side=[id])
    child_summaries = relationship("DocumentSummary")
    context_results = relationship("ContextResult", back_populates="summary")

    __table_args__ = (
        # Cosine similarity search index
        Index(
            "ix_document_summaries_embedding_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )

class ContextResult(Base):
    """Model for context results from queries."""
    __tablename__ = "context_results"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("document_metadata.id"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("documents.id"))
    summary_id = Column(Integer, ForeignKey("document_summaries.id"))
    relevance_score = Column(Float, nullable=False)
    used_in_response = Column(Boolean, default=False)

    # Relationships
    document = relationship("DocumentMetadata")
    chunk = relationship("Document")
    summary = relationship("DocumentSummary", back_populates="context_results")
    response_citations = relationship("ResponseCitation", back_populates="context_used")

class Citation(Base):
    """Model for legal citations."""
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("document_metadata.id"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("documents.id"))
    citation_text = Column(String, nullable=False)
    citation_type = Column(String, nullable=False)  # "case", "statute", etc.
    normalized_format = Column(String, nullable=False)
    authority_level = Column(Integer)
    citation_metadata = Column(JSON, nullable=False, default={})

    # Relationships
    document = relationship("DocumentMetadata")
    chunk = relationship("Document")
    response_citations = relationship("ResponseCitation", back_populates="citation")

class ContextSet(Base):
    """Model for complete context sets."""
    __tablename__ = "context_sets"

    id = Column(Integer, primary_key=True)
    original_query_id = Column(Integer, ForeignKey("original_user_queries.id", ondelete="CASCADE"), nullable=False)
    context_data = Column(JSON, nullable=False, comment="Complete context data including chunks")
    context_metadata = Column(JSON, nullable=False, default={}, comment="Context metadata like total chunks, tokens, etc.")
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationships
    original_query = relationship("OriginalUserQuery")

class ResponseCitation(Base):
    """Model for citations used in responses."""
    __tablename__ = "response_citations"

    id = Column(Integer, primary_key=True)
    citation_id = Column(Integer, ForeignKey("citations.id"), nullable=False)
    context_used_id = Column(Integer, ForeignKey("context_results.id"), nullable=False)
    relevance_score = Column(Float, nullable=False)

    # Relationships
    citation = relationship("Citation", back_populates="response_citations")
    context_used = relationship("ContextResult", back_populates="response_citations")