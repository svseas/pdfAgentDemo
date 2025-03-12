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

    async def _try_models(self, prompt: str) -> str:
        """Try different models in sequence until one works."""
        models = [
            "qwen2.5-7b-instruct-1m",  # Try Qwen first
            "llama3-docchat-1.0-8b-i1",  # Fallback to Llama if Qwen fails
        ]
        
        last_error = None
        for model in models:
            try:
                logger.info(f"Attempting to use model: {model}")
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": -1,
                    "stream": False
                }
                
                async with httpx.AsyncClient(timeout=settings.LMSTUDIO_TIMEOUT) as client:
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
                        raise Exception(f"Unexpected response format: {response_data}")
                    
                    logger.info(f"Successfully generated response using model: {model}")
                    return response_data["choices"][0]["message"]["content"]
                    
            except Exception as e:
                logger.error(f"Error with model {model}: {str(e)}")
                last_error = e
                continue
        
        # If all models failed
        logger.error("All models failed to generate response")
        return ""  # Return empty string as fallback

    async def generate_stepback(self, context: str, query: str) -> str:
        """Generate a higher-level perspective on the query."""
        prompt = self.stepback_prompt.format(
            context=context,
            query=query
        )
        
        return await self._try_models(prompt)

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
        
        # Try to generate enhanced answer with fallback
        enhanced_answer = await self._try_models(prompt)
        
        if not enhanced_answer:
            logger.warning("Failed to generate enhanced answer, returning initial answer")
            return initial_answer
            
        # Format final response to include reasoning
        final_response = f"""Phân tích tổng quan:
{stepback_result}

Câu trả lời chi tiết:
{enhanced_answer}"""
        
        return final_response