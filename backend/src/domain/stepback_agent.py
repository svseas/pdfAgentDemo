import logging
import httpx
from src.core.config import settings

logger = logging.getLogger(__name__)

class StepbackAgent:
    """Agent that implements stepback prompting to improve answer quality.
    
    Takes the initial context and query, generates a higher-level perspective,
    and uses that to enhance the final answer.
    """
    
    def __init__(self):
        self.llm_url = f"{settings.LMSTUDIO_BASE_URL}/chat/completions"
        
        self.stepback_prompt = """Bạn là một trợ lý AI chuyên nghiệp. Hãy phân tích câu hỏi và ngữ cảnh được cung cấp theo các bước sau:

1. Xác định chủ đề/lĩnh vực rộng hơn mà câu hỏi này thuộc về
2. Xác định các khái niệm chính cần thiết để hiểu chủ đề này
3. Tạo một câu hỏi tổng quát hơn để giúp hiểu rõ bối cảnh của câu hỏi cụ thể

Ngữ cảnh:
{context}

Câu hỏi:
{query}

Hãy phân tích từng bước:"""

        self.enhance_prompt = """Bạn là một trợ lý AI chuyên nghiệp. Dựa trên:

1. Ngữ cảnh ban đầu
2. Câu hỏi gốc
3. Góc nhìn tổng quan vừa được phân tích
4. Câu trả lời ban đầu

Hãy tạo một câu trả lời toàn diện hơn, kết hợp cả chi tiết cụ thể và hiểu biết tổng quan.

Ngữ cảnh ban đầu:
{context}

Câu hỏi gốc:
{query}

Góc nhìn tổng quan:
{stepback_result}

Câu trả lời ban đầu:
{initial_answer}

Hãy đưa ra câu trả lời toàn diện hơn:"""

    async def generate_stepback(self, context: str, query: str) -> str:
        """Generate a higher-level perspective on the query."""
        # Format prompt
        prompt = self.stepback_prompt.format(
            context=context,
            query=query
        )
        
        # Prepare payload
        payload = {
            "model": "llama3-docchat-1.0-8b-i1",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }
        
        # Make request to LMStudio
        async with httpx.AsyncClient(timeout=settings.LMSTUDIO_TIMEOUT) as client:
            try:
                response = await client.post(
                    self.llm_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )
                response.raise_for_status()
                response_data = response.json()
                
                if "choices" not in response_data:
                    logger.error(f"Unexpected response format: {response_data}")
                    raise Exception(f"Unexpected response format: {response_data}")
                
                return response_data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Error generating stepback perspective: {str(e)}")
                return ""

    async def enhance_answer(self, 
                           context: str, 
                           query: str,
                           initial_answer: str) -> str:
        """Enhance the initial answer using stepback prompting."""
        # Generate broader perspective
        stepback_result = await self.generate_stepback(context, query)
        
        if not stepback_result:
            logger.warning("Failed to generate stepback perspective, returning initial answer")
            return initial_answer
        
        # Format enhance prompt
        prompt = self.enhance_prompt.format(
            context=context,
            query=query,
            stepback_result=stepback_result,
            initial_answer=initial_answer
        )
        
        # Prepare payload
        payload = {
            "model": "llama3-docchat-1.0-8b-i1",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }
        
        # Make request to LMStudio
        async with httpx.AsyncClient(timeout=settings.LMSTUDIO_TIMEOUT) as client:
            try:
                response = await client.post(
                    self.llm_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )
                response.raise_for_status()
                response_data = response.json()
                
                if "choices" not in response_data:
                    logger.error(f"Unexpected response format: {response_data}")
                    return initial_answer
                
                enhanced_answer = response_data["choices"][0]["message"]["content"]
                
                # Format final response to include reasoning
                final_response = f"""Phân tích tổng quan:
{stepback_result}

Câu trả lời chi tiết:
{enhanced_answer}"""
                
                return final_response
            except Exception as e:
                logger.error(f"Error enhancing answer: {str(e)}")
                return initial_answer