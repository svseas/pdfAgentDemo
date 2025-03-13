"""Semantic text splitting implementation."""
from typing import List, Optional
import re
from src.domain.interfaces import TextSplitterInterface
from src.domain.exceptions import ChunkingError

class SemanticTextSplitter(TextSplitterInterface):
    """Split text into semantic units like paragraphs and sentences."""
    
    def __init__(
        self,
        max_characters: int = 1500,
        semantic_units: Optional[List[str]] = None,
        break_mode: str = "sentence",
        flex: float = 0.1
    ):
        """
        Initialize text splitter.
        
        Args:
            max_characters: Maximum characters per chunk
            semantic_units: List of semantic units to split on, in order of priority
            break_mode: How to break text when no semantic splits found ("word" or "sentence")
            flex: Flexibility ratio for chunk size (0-1)
        """
        self.max_characters = max_characters
        self.semantic_units = semantic_units or ["paragraph", "sentence"]
        self.break_mode = break_mode
        self.flex = flex
        
        # Regex patterns for different semantic units
        self.patterns = {
            "paragraph": r"\n\s*\n+",  # Double newline
            "sentence": r"(?<=[.!?])\s+(?=[A-Z])",  # Period + space + capital
            "word": r"\s+"  # Any whitespace
        }
        
    def _get_splits(self, text: str, unit: str) -> List[str]:
        """Split text by semantic unit."""
        if unit not in self.patterns:
            raise ChunkingError(f"Unknown semantic unit: {unit}")
            
        pattern = self.patterns[unit]
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]
        
    def _merge_small_chunks(self, chunks: List[str], min_size: int) -> List[str]:
        """Merge chunks smaller than min_size with neighbors."""
        if not chunks:
            return chunks
            
        result = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            if len(current) < min_size:
                current += "\n\n" + chunk
            else:
                result.append(current)
                current = chunk
                
        result.append(current)
        return result
        
    def split(self, text: str) -> List[str]:
        """
        Split text into chunks using semantic units.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
            
        Raises:
            ChunkingError: If chunking fails
        """
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Calculate size bounds
            min_size = int(self.max_characters * (1 - self.flex))
            max_size = int(self.max_characters * (1 + self.flex))
            
            chunks = [text]
            
            # Try each semantic unit in order
            for unit in self.semantic_units:
                new_chunks = []
                
                for chunk in chunks:
                    # If chunk is within bounds, keep it
                    if len(chunk) <= max_size:
                        new_chunks.append(chunk)
                        continue
                        
                    # Split oversized chunk
                    splits = self._get_splits(chunk, unit)
                    new_chunks.extend(splits)
                    
                chunks = new_chunks
                
            # Final pass: merge small chunks and split oversized ones
            chunks = self._merge_small_chunks(chunks, min_size)
            
            # Split any remaining oversized chunks
            if self.break_mode == "word":
                pattern = self.patterns["word"]
            else:  # sentence
                pattern = self.patterns["sentence"]
                
            final_chunks = []
            for chunk in chunks:
                if len(chunk) <= max_size:
                    final_chunks.append(chunk)
                else:
                    # Split on words/sentences and join until max size
                    parts = re.split(pattern, chunk)
                    current = parts[0]
                    
                    for part in parts[1:]:
                        if len(current) + len(part) <= max_size:
                            current += " " + part
                        else:
                            final_chunks.append(current)
                            current = part
                            
                    if current:
                        final_chunks.append(current)
                        
            return final_chunks
            
        except Exception as e:
            raise ChunkingError(f"Failed to split text: {str(e)}")