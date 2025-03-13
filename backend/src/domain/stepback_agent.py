"""Stepback prompting implementation."""
import logging
from typing import Optional
from src.domain.interfaces import LLMInterface
from src.domain.exceptions import LLMError

logger = logging.getLogger(__name__)

class StepbackPromptBuilder:
    """Build prompts for stepback prompting."""
    
    @staticmethod
    def build_stepback_prompt(context: str, query: str, language: str = "vi") -> str:
        """Build prompt for generating stepback perspective."""
        if language == "vi":
            return f"""Hãy phân tích vấn đề ở mức độ tổng quan hơn:

Nội dung:
{context}

Câu hỏi:
{query}

Hãy:
1. Xác định các khái niệm và nguyên tắc cơ bản liên quan
2. Phân tích mối quan hệ giữa các yếu tố
3. Đưa ra góc nhìn tổng thể về vấn đề
4. KHÔNG đưa ra câu trả lời cụ thể cho câu hỏi"""
        else:
            return f"""Analyze the issue from a broader perspective:

Content:
{context}

Question:
{query}

Please:
1. Identify relevant core concepts and principles
2. Analyze relationships between elements
3. Provide a holistic view of the issue
4. DO NOT provide specific answers to the question"""

    @staticmethod
    def build_enhance_prompt(
        context: str,
        query: str,
        stepback_result: str,
        initial_answer: str,
        language: str = "vi"
    ) -> str:
        """Build prompt for enhancing answer with stepback perspective."""
        if language == "vi":
            return f"""Dựa trên phân tích tổng quan và câu trả lời ban đầu, hãy đưa ra câu trả lời chi tiết và đầy đủ hơn.

Nội dung:
{context}

Câu hỏi:
{query}

Phân tích tổng quan:
{stepback_result}

Câu trả lời ban đầu:
{initial_answer}

Hãy:
1. Kết hợp góc nhìn tổng thể với chi tiết cụ thể
2. Đảm bảo câu trả lời đầy đủ và chính xác
3. Giải thích rõ mối liên hệ giữa các khía cạnh
4. Nêu rõ nếu thiếu thông tin quan trọng"""
        else:
            return f"""Based on the broader analysis and initial answer, provide a more detailed and comprehensive response.

Content:
{context}

Question:
{query}

Broader Analysis:
{stepback_result}

Initial Answer:
{initial_answer}

Please:
1. Combine holistic perspective with specific details
2. Ensure comprehensive and accurate response
3. Explain relationships between aspects clearly
4. Indicate if important information is missing"""

class StepbackAgent:
    """Agent that implements stepback prompting to improve answer quality."""
    
    def __init__(self, llm_service: LLMInterface):
        """
        Initialize stepback agent.
        
        Args:
            llm_service: Service for LLM interactions
        """
        self.llm_service = llm_service
        self.prompt_builder = StepbackPromptBuilder()
        
    async def generate_stepback(
        self,
        context: str,
        query: str,
        language: str = "vi",
        temperature: float = 0.7
    ) -> str:
        """
        Generate higher-level perspective on query.
        
        Args:
            context: Document context
            query: User query
            language: Response language
            temperature: LLM temperature
            
        Returns:
            Stepback perspective
            
        Raises:
            LLMError: If generation fails
        """
        try:
            # Build stepback prompt
            prompt = self.prompt_builder.build_stepback_prompt(
                context=context,
                query=query,
                language=language
            )
            
            # Generate stepback perspective
            return await self.llm_service.generate_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            
        except Exception as e:
            logger.error(f"Error generating stepback perspective: {str(e)}")
            raise LLMError(f"Failed to generate stepback perspective: {str(e)}")

    async def enhance_answer(
        self,
        context: str,
        query: str,
        initial_answer: str,
        language: str = "vi",
        temperature: float = 0.7
    ) -> str:
        """
        Enhance initial answer using stepback prompting.
        
        Args:
            context: Document context
            query: User query
            initial_answer: Initial response to enhance
            language: Response language
            temperature: LLM temperature
            
        Returns:
            Enhanced answer
            
        Raises:
            LLMError: If enhancement fails
        """
        try:
            # Generate broader perspective
            stepback_result = await self.generate_stepback(
                context=context,
                query=query,
                language=language,
                temperature=temperature
            )
            
            # Build enhance prompt
            prompt = self.prompt_builder.build_enhance_prompt(
                context=context,
                query=query,
                stepback_result=stepback_result,
                initial_answer=initial_answer,
                language=language
            )
            
            # Generate enhanced answer
            enhanced_answer = await self.llm_service.generate_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            
            # Format final response
            final_response = f"""{"Phân tích tổng quan" if language == "vi" else "Broader Analysis"}:
{stepback_result}

{"Câu trả lời chi tiết" if language == "vi" else "Detailed Answer"}:
{enhanced_answer}"""
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error enhancing answer: {str(e)}")
            raise LLMError(f"Failed to enhance answer: {str(e)}")