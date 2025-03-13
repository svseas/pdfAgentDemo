from abc import ABC, abstractmethod
from typing import List, Dict, Any
import httpx
import logging
from .config import settings

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False
    ) -> Dict[str, Any]:
        pass

class LMStudioProvider(LLMProvider):
    def __init__(self):
        self.base_url = settings.LMSTUDIO_BASE_URL
        self.model = settings.LMSTUDIO_MODEL
        self.timeout = settings.LMSTUDIO_TIMEOUT

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": stream
                    }
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Error in LMStudio completion: {str(e)}")
                raise

class OpenRouterProvider(LLMProvider):
    def __init__(self):
        self.base_url = settings.OPENROUTER_BASE_URL
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.OPENROUTER_MODEL
        self.timeout = settings.OPENROUTER_TIMEOUT

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Required headers as per OpenRouter API
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://your-site.com",  # Replace with actual site URL
                    "X-Title": "PdfAgentDemo"  # Replace with actual site name
                }

                # Request body with required model format
                json_data = {
                    "model": "qwen/qwq-32b:free",  # Use the specific free model
                    "messages": messages
                }

                # Add optional parameters if specified
                if temperature != 0.7:
                    json_data["temperature"] = temperature
                if max_tokens > 0:
                    json_data["max_tokens"] = max_tokens
                if stream:
                    json_data["stream"] = stream

                logger.debug(f"OpenRouter request: {json_data}")
                response = await client.post(
                    f"{self.base_url}/chat/completions",  # Use configured base URL with correct endpoint
                    headers=headers,
                    json=json_data
                )

                if not response.is_success:
                    logger.error(f"OpenRouter error response: {response.content}")
                response.raise_for_status()
                
                return response.json()
            except Exception as e:
                logger.error(f"Error in OpenRouter completion: {str(e)}")
                if isinstance(e, httpx.HTTPError):
                    logger.error(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
                raise

class LLMService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if settings.LLM_PROVIDER == "openrouter":
                cls._instance.provider = OpenRouterProvider()
            else:
                cls._instance.provider = LMStudioProvider()
        return cls._instance

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False
    ) -> str:
        """Generate completion and return the content."""
        try:
            # Ensure messages have the correct format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    formatted_messages.append(msg)
                else:
                    # Default to user role if not specified
                    formatted_messages.append({
                        "role": "user",
                        "content": str(msg) if isinstance(msg, (str, dict)) else ""
                    })

            response = await self.provider.generate_completion(
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error in completion generation: {str(e)}")
            raise

    def get_prompt(self, prompt_type: str, language: str = "vi") -> str:
        """Get prompt template based on type and language."""
        prompts = {
            "system": {
                "vi": settings.SYSTEM_PROMPT_VI,
                "en": settings.SYSTEM_PROMPT_EN
            },
            "stepback": {
                "vi": """Bạn là một trợ lý AI chuyên nghiệp. Khi được đưa ra một câu hỏi, hãy tạo ra một câu hỏi tổng quát hơn 
để giúp trả lời câu hỏi gốc tốt hơn. Ví dụ, nếu được hỏi 'Harry nói gì với Voldemort trong trận chiến cuối cùng?', 
bạn có thể đặt câu hỏi tổng quát hơn là 'Cuộc đối đầu cuối cùng giữa Harry Potter và Voldemort nói về điều gì?'. 
Chỉ trả lời bằng câu hỏi tổng quát, không thêm bất kỳ nội dung nào khác.

Ngữ cảnh:
{context}

Câu hỏi:
{query}""",
                "en": """You are an AI assistant. When given a question, help break it down into a more general 'step-back' question 
that will help better answer the original query. For example, if asked 'What did Harry say to Voldemort in their final battle?', 
you might step back and ask 'What was the final confrontation between Harry Potter and Voldemort about?'. 
Respond only with the step-back question, no other text.

Context:
{context}

Question:
{query}"""
            },
            "enhance": {
                "vi": """Bạn là một trợ lý AI chuyên nghiệp. Dựa trên:

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
{initial_answer}""",
                "en": """You are an AI assistant. Based on:

1. Initial context
2. Original question
3. The broader perspective just analyzed
4. Initial answer

Create a more comprehensive answer that combines both specific details and broader understanding.

Initial context:
{context}

Original question:
{query}

Broader perspective:
{stepback_result}

Initial answer:
{initial_answer}"""
            }
        }
        return prompts[prompt_type][language]