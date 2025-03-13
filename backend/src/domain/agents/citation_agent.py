"""Citation extraction and processing agent."""
from typing import Dict, Any, List
import re
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.workflow_repository import (
    SQLCitationRepository,
    SQLContextRepository
)
from .base_agent import BaseAgent

class CitationAgent(BaseAgent):
    """Agent that extracts, verifies, and formats citations."""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return output."""
        workflow_run_id = input_data.get("workflow_run_id")
        context_results = input_data.get("context_results", [])
        
        # Log step start
        agent_step_id = await self.log_step(
            workflow_run_id,
            None,
            input_data,
            {},
            "running"
        )
        
        try:
            citation_repo = SQLCitationRepository(self.session)
            context_repo = SQLContextRepository(self.session)
            
            # Process each context result
            extracted_citations = []
            
            for context_id in context_results:
                context = await context_repo.get_by_id(context_id)
                if not context:
                    continue
                
                # Extract citations from text
                text = await self._get_context_text(context)
                citations = await self._extract_citations(text)
                
                for citation in citations:
                    # Create citation record
                    citation_id = await citation_repo.create_citation(
                        document_id=context["document_id"],
                        chunk_id=context.get("chunk_id"),
                        citation_text=citation["text"],
                        citation_type=citation["type"],
                        normalized_format=await self._normalize_citation(citation),
                        authority_level=await self._determine_authority(citation),
                        metadata={}
                    )
                    
                    # Link to response
                    await citation_repo.create_response_citation(
                        workflow_run_id,
                        citation_id,
                        context_id,
                        citation.get("relevance", 0.8)
                    )
                    
                    extracted_citations.append({
                        "id": citation_id,
                        **citation
                    })
            
            output_data = {
                "citations": extracted_citations,
                "count": len(extracted_citations)
            }
            
            # Update step status
            await self._update_step_status(
                agent_step_id,
                "success",
                output_data
            )
            
            return output_data
            
        except Exception as e:
            await self._update_step_status(
                agent_step_id,
                "failed",
                {"error": str(e)}
            )
            raise
    
    async def _get_context_text(self, context: Dict[str, Any]) -> str:
        """Get text from context result."""
        if context.get("chunk_id"):
            return context.get("chunk_text", "")
        elif context.get("summary_id"):
            return context.get("summary_text", "")
        return ""
    
    async def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations from text using LLM."""
        if not self.llm or not self.prompt_manager:
            # Fallback to regex-based extraction if LLM not available
            return await self._extract_citations_regex(text)
            
        try:
            # Use LLM to extract citations
            prompt = self.prompt_manager.format_prompt(
                "citation_extraction",
                language="vi",  # Use Vietnamese for document processing
                text=text
            )
            
            result = await self.llm.generate_completion([
                {"role": "system", "content": "You are a citation extraction expert."},
                {"role": "user", "content": prompt}
            ])
            
            # Parse LLM response to extract citations
            citations = []
            for line in result.split("\n"):
                # Match Vietnamese section patterns
                if "Mục" in line or "Phần" in line or "Điều" in line:
                    # Extract section number and text
                    match = re.search(r"(Mục|Phần|Điều)\s+(\d+(?:\.\d+)?):?\s*(.*)", line)
                    if match:
                        section_type = match.group(1)
                        section_num = match.group(2)
                        section_text = match.group(3).strip()
                        
                        citations.append({
                            "text": section_text,
                            "type": "section",
                            "section_number": section_num,
                            "section_type": section_type.lower(),
                            "relevance": 0.9
                        })
            
            return citations
            
        except Exception as e:
            print(f"LLM citation extraction failed: {e}")
            return await self._extract_citations_regex(text)
    
    async def _extract_citations_regex(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations using regex patterns."""
        citations = []
        
        # Match Vietnamese section patterns
        section_pattern = r'(?:Mục|Phần|Điều)\s+(\d+(?:\.\d+)?)\s*[:.]\s*([^.!?\n]+)'
        matches = re.finditer(section_pattern, text, re.IGNORECASE)
        
        for match in matches:
            section_num = match.group(1)
            section_text = match.group(2).strip()
            
            citations.append({
                "text": section_text,
                "type": "section",
                "section_number": section_num,
                "section_type": "muc",  # Default to 'muc' for Vietnamese sections
                "relevance": 0.85
            })
        
        return citations
    
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
        """Determine citation relevance level."""
        if citation["type"] != "section":
            return 3  # Default for non-section citations
            
        # Higher authority for main sections vs subsections
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
            # Subsections have lower authority
            return 2  # Lower for subsections