"""Document service implementation."""
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any
import magic

from src.domain.pdf_processor import PDFProcessor
from src.domain.embedding_generator import EmbeddingGenerator
from src.domain.query_processor import QueryProcessor
from src.repositories.document_repository import DocumentRepository
from src.schemas.document import (
    DocumentMetadataCreate,
    DocumentMetadataRead,
    DocumentChunkCreate,
    DocumentProcessingStatus
)

logger = logging.getLogger(__name__)

class DocumentService:
    """Application service for document processing operations"""
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        pdf_processor: PDFProcessor,
        embedding_generator: EmbeddingGenerator,
        query_processor: QueryProcessor
    ):
        self._repository = document_repository
        self._pdf_processor = pdf_processor
        self._embedding_generator = embedding_generator
        self._query_processor = query_processor

    async def create_document(
        self,
        file_path: Path,
        original_filename: str,
        file_size: int
    ) -> DocumentMetadataRead:
        """
        Create a new document entry from a PDF file or return existing one.
        
        Args:
            file_path: Path to the uploaded PDF file
            original_filename: Original name of the uploaded file
            file_size: Size of the file in bytes
            
        Returns:
            Document metadata (either existing or newly created)
            
        Raises:
            ValueError: If file validation fails
        """
        # Validate file type
        mime_type = magic.from_file(str(file_path), mime=True)
        if mime_type != "application/pdf":
            raise ValueError(f"Invalid file type: {mime_type}")
            
        # Check if document already exists
        existing_doc = await self._repository.get_metadata_by_filename(original_filename)
        if existing_doc:
            # If document exists and is completed, return it
            if existing_doc.processing_status == "completed":
                return existing_doc
                
            # If document exists but failed, reset it for reprocessing
            if existing_doc.processing_status == "failed":
                await self._repository.update_metadata_status(
                    existing_doc.id,
                    "pending",
                    total_chunks=0
                )
                return existing_doc
                
            # If document is pending or processing, return it
            return existing_doc
            
        # Create new document metadata
        doc_metadata = await self._repository.create_metadata(
            DocumentMetadataCreate(
                filename=original_filename,
                file_size=file_size,
                mime_type=mime_type
            )
        )
        
        return doc_metadata

    async def process_document(self, doc_id: int, file_path: Path) -> None:
        """
        Process a document by extracting text, generating embeddings,
        and storing chunks in the database.
        
        Args:
            doc_id: ID of the document to process
            file_path: Path to the PDF file
        """
        try:
            # Update status to processing
            await self._repository.update_metadata_status(doc_id, "processing")
            
            # Extract text from PDF
            text = self._pdf_processor.extract_text(str(file_path))
            if not text:
                raise ValueError("No text content extracted from PDF")
                
            # Process text into chunks
            chunks = self._pdf_processor.process_text(text)
            if not chunks:
                raise ValueError("No chunks created from text")
            
            # Generate embeddings
            embeddings = self._embedding_generator.generate_embeddings(chunks)
            
            # Save chunks with embeddings
            filename = file_path.name
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_create = DocumentChunkCreate(
                    filename=filename,
                    chunk_index=i,
                    content=chunk,
                    doc_metadata_id=doc_id,
                    embedding=embedding
                )
                await self._repository.create_chunk(chunk_create)
            
            # Update document status
            await self._repository.update_metadata_status(
                doc_id,
                "completed",
                total_chunks=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            await self._repository.update_metadata_status(doc_id, "failed")
            raise

    async def get_document_status(self, doc_id: int) -> Optional[DocumentProcessingStatus]:
        """Get the processing status of a document"""
        doc = await self._repository.get_metadata_by_id(doc_id)
        if not doc:
            return None
            
        return DocumentProcessingStatus(
            filename=doc.filename,
            status=doc.processing_status,
            total_chunks=doc.total_chunks,
            updated_at=doc.updated_at
        )

    async def list_documents(self) -> List[DocumentMetadataRead]:
        """List all documents"""
        return await self._repository.list_metadata()

    async def query_document(
        self,
        doc_id: int,
        query: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query a document and get a response based on its content.
        
        Args:
            doc_id: ID of the document to query
            query: The user's question
            temperature: Temperature parameter for response generation
            
        Returns:
            Dict containing the response and relevant chunks used
            
        Raises:
            ValueError: If document not found or not processed
        """
        # Get document metadata
        doc = await self._repository.get_metadata_by_id(doc_id)
        if not doc:
            raise ValueError(f"Document with ID {doc_id} not found")
            
        if doc.processing_status != "completed":
            raise ValueError(
                f"Document {doc_id} is not ready for querying "
                f"(status: {doc.processing_status})"
            )
            
        # Get document chunks with embeddings
        chunks = await self._repository.get_chunks_by_doc_id(doc_id)
        if not chunks:
            raise ValueError(f"No chunks found for document {doc_id}")
            
        logger.info(f"Retrieved {len(chunks)} chunks from database")
        
        # Convert chunks to the format expected by QueryProcessor
        chunk_data = [
            {
                "content": chunk.content,
                "embedding": chunk.embedding
            }
            for chunk in chunks
        ]
        
        # Get relevant chunks
        relevant_chunks = self._query_processor.get_relevant_chunks(
            query,
            chunk_data
        )
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Generate response
        response = await self._query_processor.generate_response(
            query,
            relevant_chunks,
            temperature
        )
        
        return {
            "response": response,
            "relevant_chunks": [
                {"content": chunk["content"]} for chunk in relevant_chunks
            ]
        }

    async def delete_document(self, doc_id: int) -> bool:
        """
        Delete a document and all its associated chunks.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        # Get document metadata
        doc = await self._repository.get_metadata_by_id(doc_id)
        if not doc:
            return False
            
        # Delete document (will cascade to chunks due to foreign key)
        await self._repository.delete_document(doc_id)
        
        return True