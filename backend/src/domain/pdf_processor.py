"""PDF processing implementation."""
from pathlib import Path
from typing import List, Optional
import logging
from unstructured.partition.pdf import partition_pdf
from src.core.config import settings
from src.domain.interfaces import DocumentProcessorInterface, TextSplitterInterface
from src.domain.exceptions import DocumentProcessingError
from src.domain.semantic_text_splitter import SemanticTextSplitter
from src.domain.agentic_chunker import AgenticChunker

logger = logging.getLogger(__name__)

class PDFProcessor(DocumentProcessorInterface):
    """Process PDF documents into text chunks."""
    
    def __init__(
        self,
        chunking_method: str = "semantic",
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        language: str = "vietnamese"
    ):
        """
        Initialize PDF processor.
        
        Args:
            chunking_method: Method to use for text chunking ("semantic" or "agentic")
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            language: Document language for agentic chunking
        """
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        
        # Initialize text splitters
        self.semantic_splitter = SemanticTextSplitter(
            max_characters=chunk_size,
            semantic_units=["paragraph", "sentence"],
            break_mode="sentence",
            flex=chunk_overlap / chunk_size
        )
        
        self.agentic_splitter = None  # Lazy initialization
        
    def _get_agentic_splitter(self) -> TextSplitterInterface:
        """Get or create agentic splitter."""
        if self.agentic_splitter is None:
            self.agentic_splitter = AgenticChunker(
                llm_endpoint=settings.LMSTUDIO_BASE_URL,
                model=settings.LMSTUDIO_MODEL,
                language=self.language
            )
        return self.agentic_splitter
        
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        path = Path(file_path)
        
        try:
            if not path.exists():
                raise DocumentProcessingError(f"File not found: {path}")
                
            if path.suffix.lower() != '.pdf':
                raise DocumentProcessingError(f"Not a PDF file: {path}")
                
            logger.info(f"Processing PDF file: {path}")
            
            # Extract text using unstructured
            elements = partition_pdf(filename=str(path))
            full_text = " ".join(str(element) for element in elements if str(element).strip())
            
            logger.info(f"Extracted {len(full_text)} characters of text")
            return full_text
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract text from PDF: {str(e)}")
            
    def process_text(self, text: str) -> List[str]:
        """
        Process text into chunks.
        
        Args:
            text: Text to process
            
        Returns:
            List of text chunks
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            # Choose chunking method
            if self.chunking_method == "agentic":
                logger.info("Using agentic chunking")
                splitter = self._get_agentic_splitter()
            else:
                logger.info("Using semantic chunking")
                splitter = self.semantic_splitter
                
            # Split text into chunks
            chunks = splitter.split(text)
            
            # Log chunk information
            logger.info(f"Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
                logger.info(f"Chunk {i} (length {len(chunk)}): {chunk[:100]}...")
                
            return chunks
            
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            
            # Fallback to semantic chunking if agentic fails
            if self.chunking_method == "agentic":
                logger.info("Falling back to semantic chunking")
                return self.semantic_splitter.split(text)
            else:
                raise DocumentProcessingError(f"Failed to process text: {str(e)}")