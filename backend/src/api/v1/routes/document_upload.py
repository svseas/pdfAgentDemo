"""Document upload and processing routes."""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import logging
import shutil
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_document_service, get_embedding_generator, get_db
from src.api.v1.middleware.error_handler import handle_errors
from src.api.v1.common.response_builder import ResponseBuilder
from src.services.document_service import DocumentService
from src.domain.embedding_generator import EmbeddingGenerator
from src.models.document import Document
from src.schemas.document import DocumentEmbedRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload")
@handle_errors
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service)
) -> dict:
    """Upload and process a PDF document.
    
    Args:
        file: The PDF file to upload
        document_service: Service for document operations
        
    Returns:
        Dict containing upload status and document metadata
        
    Raises:
        HTTPException: If file type is invalid or processing fails
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
        
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = upload_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Create document metadata
        doc_metadata = await document_service.create_document(
            file_path=file_path,
            original_filename=file.filename,
            file_size=file.size
        )
        
        # Process document (extract text, generate embeddings, store chunks)
        await document_service.process_document(doc_metadata.id, file_path)
        
        return ResponseBuilder.success(
            data={
                "document_id": doc_metadata.id,
                "filename": file.filename
            },
            message="Document uploaded and processed successfully",
            metadata={
                "file_size": file.size,
                "file_path": str(file_path)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.post("/embed")
@handle_errors
async def embed_document(
    request: DocumentEmbedRequest,
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Generate embeddings for all chunks of a document.
    
    Args:
        request: Document embedding parameters
        embedding_generator: Service for generating embeddings
        db: Database session
        
    Returns:
        Dict containing embedding status
        
    Raises:
        HTTPException: If document not found or embedding fails
    """
    try:
        # Get document chunks
        query = select(Document).where(Document.doc_metadata_id == request.document_id)
        if not request.force:
            query = query.where(Document.embedding.is_(None))
            
        result = await db.execute(query)
        chunks = result.scalars().all()
        
        if not chunks:
            if request.force:
                raise HTTPException(
                    status_code=404,
                    detail=f"No chunks found for document {request.document_id}"
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"No chunks without embeddings found for document {request.document_id}"
                )
        
        # Process chunks in batches
        batch_size = 10
        total_chunks = len(chunks)
        processed_chunks = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            # Generate embeddings
            embeddings = await embedding_generator.generate_embeddings(texts)
            
            # Update chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding.tolist()
                
            await db.commit()
            processed_chunks += len(batch)
            logger.info(f"Processed {processed_chunks}/{total_chunks} chunks")
            
        return ResponseBuilder.success(
            data={
                "document_id": request.document_id,
                "total_chunks": total_chunks,
                "processed_chunks": processed_chunks
            },
            message="Document chunks embedded successfully",
            metadata={
                "force_update": request.force,
                "batch_size": batch_size
            }
        )
        
    except Exception as e:
        logger.error(f"Error embedding document chunks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error embedding document chunks: {str(e)}"
        )