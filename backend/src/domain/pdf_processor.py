from pathlib import Path
from typing import List
import logging
from unstructured.partition.pdf import partition_pdf
from src.core.config import settings
from src.domain.semantic_text_splitter import TextSplitter
from src.domain.agentic_chunker import AgenticChunker

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Domain service for PDF processing"""

    @staticmethod
    def _split_text_semantic(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into chunks with overlap using semantic text splitting.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Initialize semantic text splitter with appropriate parameters
        splitter = TextSplitter(
            max_characters=chunk_size,
            semantic_units=["paragraph", "sentence"],
            break_mode="sentence",
            flex=overlap / chunk_size  # Convert overlap to flex ratio
        )
        
        return splitter.chunks(text)

    @staticmethod
    def _split_text_agentic(text: str) -> List[str]:
        """
        Split text into chunks using agentic chunking for legal documents.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        try:
            chunker = AgenticChunker(
                llm_endpoint=settings.LMSTUDIO_BASE_URL,
                model=settings.LMSTUDIO_MODEL,
                language=settings.AGENTIC_CHUNKING_LANGUAGE
            )
            
            logger.info("Using agentic chunking for legal documents")
            chunks = chunker.process_text(text)
            
            if not chunks:
                logger.warning("Agentic chunking returned no chunks, falling back to semantic chunking")
                return PDFProcessor._split_text_semantic(
                    text,
                    settings.CHUNK_SIZE,
                    settings.CHUNK_OVERLAP
                )
                
            return chunks
            
        except Exception as e:
            logger.error(f"Agentic chunking failed: {str(e)}, falling back to semantic chunking")
            return PDFProcessor._split_text_semantic(
                text,
                settings.CHUNK_SIZE,
                settings.CHUNK_OVERLAP
            )

    @staticmethod
    def extract_text_chunks(file_path: Path) -> List[str]:
        """
        Extract text chunks from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of text chunks extracted from the PDF
            
        Raises:
            ValueError: If file doesn't exist or isn't a PDF
        """
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Not a PDF file: {file_path}")
            
        logger.info(f"Processing PDF file: {file_path}")
            
        # Extract text using unstructured
        elements = partition_pdf(filename=str(file_path))
        
        # Join elements into a single text
        full_text = " ".join(str(element) for element in elements if str(element).strip())
        
        logger.info(f"Extracted {len(full_text)} characters of text")
        
        # Choose chunking method based on settings
        if settings.CHUNKING_METHOD == "agentic":
            chunks = PDFProcessor._split_text_agentic(full_text)
        else:
            logger.info("Using semantic text splitting")
            chunks = PDFProcessor._split_text_semantic(
                full_text,
                settings.CHUNK_SIZE,
                settings.CHUNK_OVERLAP
            )
        
        logger.info(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
            logger.info(f"Chunk {i} (length {len(chunk)}): {chunk[:100]}...")
        
        return chunks