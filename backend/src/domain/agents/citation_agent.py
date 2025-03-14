"""Citation extraction and processing agent."""
from typing import Dict, Any, List, Optional
import re
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.citation_repository import CitationRepository
from src.repositories.context_repository import ContextRepository
from src.domain.exceptions import AgentError
from .base_agent import BaseAgent

class CitationAgent(BaseAgent):
    """Agent that extracts, verifies, and formats citations.
    
    This agent is responsible for:
    - Extracting citations from text using LLM or regex
    - Normalizing citation formats
    - Determining citation authority levels
    - Creating citation records and linking them to responses
    
    Attributes:
        citation_repo: Repository for citation operations
        context_repo: Repository for context operations
    """
    
    def __init__(
        self,
        session: AsyncSession,
        citation_repo: CitationRepository,
        context_repo: ContextRepository,
        *args,
        **kwargs
    ):
        """Initialize citation agent with required repositories.
        
        Args:
            session: Database session
            citation_repo: Repository for citation operations
            context_repo: Repository for context operations
            *args, **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(session, *args, **kwargs)
        self.citation_repo = citation_repo
        self.context_repo = context_repo

    async def _process_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of citation processing logic.
        
        Args:
            input_data: Must contain:
                - context_results: List of context result IDs
                
        Returns:
            Dict containing:
                - citations: List of extracted citations
                - count: Number of citations extracted
                
        Raises:
            AgentError: If citation processing fails
        """
        try:
            context_results = input_data.get("context_results", [])
            extracted_citations = []
            
            for context_id in context_results:
                context = await self._get_context(context_id)
                if not context:
                    continue
                
                # Extract citations from context text
                text = await self._get_context_text(context)
                citations = await self._extract_citations(text)
                
                # Process each citation
                for citation in citations:
                    citation_id = await self._create_citation(
                        context,
                        citation,
                        context_id
                    )
                    extracted_citations.append({
                        "id": citation_id,
                        **citation
                    })
            
            return {
                "citations": extracted_citations,
                "count": len(extracted_citations)
            }
            
        except Exception as e:
            raise AgentError(f"Citation processing failed: {str(e)}") from e

    async def _get_context(self, context_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve context by ID."""
        try:
            return await self.context_repo.get_by_id(context_id)
        except Exception as e:
            raise AgentError(f"Failed to get context {context_id}: {str(e)}") from e

    async def _get_context_text(self, context: Dict[str, Any]) -> str:
        """Extract text from context result."""
        if context.get("chunk_id"):
            return context.get("chunk_text", "")
        elif context.get("summary_id"):
            return context.get("summary_text", "")
        return ""

    async def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations from text using LLM or regex fallback."""
        if self.llm and self.prompt_manager:
            try:
                return await self._extract_citations_llm(text)
            except Exception:
                # Fallback to regex on LLM failure
                return await self._extract_citations_regex(text)
        return await self._extract_citations_regex(text)

    async def _extract_citations_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations using LLM."""
        prompt = self.prompt_manager.format_prompt(
            "citation_extraction",
            language="vi",
            text=text
        )
        
        result = await self.llm.generate_completion([
            {"role": "system", "content": "You are a citation extraction expert."},
            {"role": "user", "content": prompt}
        ])
        
        return self._parse_llm_citations(result)

    def _parse_llm_citations(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse citations from LLM response."""
        citations = []
        for line in llm_response.split("\n"):
            if "Mục" in line or "Phần" in line or "Điều" in line:
                match = re.search(
                    r"(Mục|Phần|Điều)\s+(\d+(?:\.\d+)?):?\s*(.*)",
                    line
                )
                if match:
                    citations.append({
                        "text": match.group(3).strip(),
                        "type": "section",
                        "section_number": match.group(2),
                        "section_type": match.group(1).lower(),
                        "relevance": 0.9
                    })
        return citations

    async def _extract_citations_regex(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations using regex patterns."""
        citations = []
        section_pattern = r'(?:Mục|Phần|Điều)\s+(\d+(?:\.\d+)?)\s*[:.]\s*([^.!?\n]+)'
        
        for match in re.finditer(section_pattern, text, re.IGNORECASE):
            citations.append({
                "text": match.group(2).strip(),
                "type": "section",
                "section_number": match.group(1),
                "section_type": "muc",
                "relevance": 0.85
            })
        
        return citations

    async def _create_citation(
        self,
        context: Dict[str, Any],
        citation: Dict[str, Any],
        context_id: int
    ) -> int:
        """Create citation record and link to response."""
        try:
            # Create citation record
            citation_id = await self.citation_repo.create_citation(
                document_id=context["document_id"],
                chunk_id=context.get("chunk_id"),
                citation_text=citation["text"],
                citation_type=citation["type"],
                normalized_format=await self._normalize_citation(citation),
                authority_level=await self._determine_authority(citation),
                metadata={}
            )
            
            # Link citation to response
            await self.citation_repo.create_response_citation(
                citation_id,
                context_id,
                citation.get("relevance", 0.8)
            )
            
            return citation_id
            
        except Exception as e:
            raise AgentError(f"Failed to create citation: {str(e)}") from e

    async def _normalize_citation(self, citation: Dict[str, Any]) -> str:
        """Normalize citation format."""
        if citation["type"] == "section":
            section_type_map = {
                "muc": "Mục",
                "phan": "Phần",
                "dieu": "Điều"
            }
            section_type = section_type_map.get(
                citation["section_type"],
                citation["section_type"].title()
            )
            return f"{section_type} {citation['section_number']}: {citation['text']}"
        return citation["text"]

    async def _determine_authority(self, citation: Dict[str, Any]) -> int:
        """Determine citation authority level."""
        if citation["type"] != "section":
            return 3  # Default for non-section citations
            
        section_num = citation["section_number"]
        if "." not in section_num:
            # Main sections have higher authority
            section_type = citation["section_type"].lower()
            if section_type == "phan":
                return 5  # Highest for main parts
            elif section_type in ["muc", "dieu"]:
                return 4  # High for main sections and articles
            return 3  # Default for other main sections
        else:
            return 2  # Lower for subsections