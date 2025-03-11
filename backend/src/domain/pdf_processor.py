from pathlib import Path
from typing import List
import logging
from unstructured.partition.pdf import partition_pdf
from src.core.config import settings

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Domain service for PDF processing"""

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into chunks with overlap, trying to break at sentence boundaries.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            
            # If this isn't the last chunk
            if end < len(text):
                # Try to find sentence end (.!?) within last 100 chars of chunk
                for i in range(end, max(end - 100, start), -1):
                    if i < len(text) and text[i] in '.!?':
                        end = i + 1  # Include the punctuation
                        break
                else:
                    # If no sentence end found, try to break at a paragraph
                    for i in range(end, max(end - 100, start), -1):
                        if i < len(text) and text[i] == '\n':
                            end = i + 1
                            break
                    else:
                        # If no paragraph break found, try to break at a space
                        for i in range(end, max(end - 50, start), -1):
                            if i < len(text) and text[i].isspace():
                                end = i
                                break
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = end - overlap
            
        return chunks

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
        
        # Split into chunks
        chunks = PDFProcessor._split_text(
            full_text,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
            logger.info(f"Chunk {i} (length {len(chunk)}): {chunk[:100]}...")
        
        return chunks