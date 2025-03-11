from typing import List, Optional
from sqlalchemy import select, text, delete
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import numpy as np

from src.models.document import Document, DocumentMetadata
from src.schemas.document import DocumentMetadataCreate, DocumentChunkCreate

logger = logging.getLogger(__name__)

class DocumentRepository:
    """Repository for document-related database operations"""
    
    def __init__(self, db: AsyncSession):
        self._db = db

    async def create_metadata(self, metadata: DocumentMetadataCreate) -> DocumentMetadata:
        """Create a new document metadata entry"""
        doc_metadata = DocumentMetadata(
            filename=metadata.filename,
            title=metadata.title,
            file_size=metadata.file_size,
            mime_type=metadata.mime_type,
            total_chunks=0,
            processing_status="pending"
        )
        self._db.add(doc_metadata)
        await self._db.commit()
        await self._db.refresh(doc_metadata)
        return doc_metadata

    async def get_metadata_by_id(self, doc_id: int) -> Optional[DocumentMetadata]:
        """Get document metadata by ID"""
        result = await self._db.execute(
            select(DocumentMetadata).where(DocumentMetadata.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def get_metadata_by_filename(self, filename: str) -> Optional[DocumentMetadata]:
        """Get document metadata by filename"""
        result = await self._db.execute(
            select(DocumentMetadata).where(DocumentMetadata.filename == filename)
        )
        return result.scalar_one_or_none()

    async def list_metadata(self) -> List[DocumentMetadata]:
        """List all document metadata"""
        result = await self._db.execute(
            select(DocumentMetadata).order_by(DocumentMetadata.created_at.desc())
        )
        return list(result.scalars().all())

    async def create_chunk(self, chunk: DocumentChunkCreate) -> Document:
        """Create a new document chunk"""
        # Convert numpy array to list if needed
        embedding = (
            chunk.embedding.tolist() 
            if isinstance(chunk.embedding, np.ndarray)
            else chunk.embedding
        )
        
        doc_chunk = Document(
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            embedding=embedding,
            doc_metadata_id=chunk.doc_metadata_id
        )
        self._db.add(doc_chunk)
        await self._db.commit()
        await self._db.refresh(doc_chunk)
        return doc_chunk

    async def update_metadata_status(
        self,
        doc_id: int,
        status: str,
        total_chunks: Optional[int] = None
    ) -> Optional[DocumentMetadata]:
        """Update document metadata status and total chunks"""
        doc_metadata = await self.get_metadata_by_id(doc_id)
        if not doc_metadata:
            return None
            
        doc_metadata.processing_status = status
        if total_chunks is not None:
            doc_metadata.total_chunks = total_chunks
            
        await self._db.commit()
        await self._db.refresh(doc_metadata)
        return doc_metadata

    async def get_chunks_by_doc_id(self, doc_id: int) -> List[Document]:
        """
        Get all chunks with their embeddings for a given document ID.
        
        Args:
            doc_id: ID of the document
            
        Returns:
            List of document chunks with their content and embeddings
        """
        # First, log some information about the chunks
        count_query = select(Document).where(Document.doc_metadata_id == doc_id)
        result = await self._db.execute(count_query)
        chunks = list(result.scalars().all())
        
        logger.info(f"Found {len(chunks)} chunks for document {doc_id}")
        
        # Convert embeddings to numpy arrays
        for chunk in chunks:
            if chunk.embedding is not None:
                chunk.embedding = np.array(chunk.embedding)
        
        # Log some sample chunks with their embedding dimensions
        for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
            logger.info(f"Chunk {i} content (first 100 chars): {chunk.content[:100]}")
            if chunk.embedding is not None:
                logger.info(f"Chunk {i} embedding dimension: {chunk.embedding.shape}")
            else:
                logger.warning(f"Chunk {i} has no embedding")
        
        return chunks

    async def get_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Document]:
        """
        Get chunks similar to the query embedding using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of chunks to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar document chunks
        """
        # First check if we have any documents with embeddings
        check_query = text("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
        count_result = await self._db.execute(check_query)
        count = count_result.scalar()
        logger.info(f"Found {count} documents with embeddings")

        # Convert embedding list to string representation for PostgreSQL
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        # Convert embedding list to string representation for PostgreSQL
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        
        # Use raw SQL with direct value interpolation for vector
        # This is safe because we construct embedding_str ourselves
        query = text(f"""
            WITH vector_query AS (
                SELECT '{embedding_str}'::vector AS query_vector
            )
            SELECT
                d.*,
                (1 - (d.embedding <=> query_vector)) * 100 as similarity,  -- Scale up similarity
                d.content as debug_content
            FROM documents d, vector_query
            WHERE d.embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT :top_k
        """)
        
        result = await self._db.execute(
            query,
            {
                "top_k": top_k
            }
        )
        
        rows = result.fetchall()
        
        # Convert rows to dictionaries with similarity scores
        chunks = []
        for row in rows:
            chunk_dict = {
                "id": row.id,
                "filename": row.filename,
                "chunk_index": row.chunk_index,
                "content": row.content,
                "doc_metadata_id": row.doc_metadata_id,
                "similarity": float(row.similarity)  # Convert Decimal to float for JSON serialization
            }
            chunks.append(chunk_dict)
        
        # Debug logging
        for chunk in chunks[:3]:  # Log first 3 results
            logger.info(f"Chunk similarity: {chunk['similarity']}, content: {chunk['content'][:100]}")
            
        logger.info(f"Found {len(chunks)} chunks with similarity scores")
        
        return chunks
        
