"""Document upload and processing routes."""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import logging
import shutil
from pathlib import Path

from src.api.dependencies import get_document_service
from src.api.v1.middleware.error_handler import handle_errors
from src.api.v1.common.response_builder import ResponseBuilder
from src.services.document_service import DocumentService

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