"""LLM provider implementations."""
import os
from typing import List, Dict, Any, Optional
import httpx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from backend.src.core.llm.interfaces import LLMInterface
from backend.src.core.config import Settings

class OpenRouterLLM(LLMInterface):
    """OpenRouter LLM implementation."""
    
    def __init__(self, settings: Settings):
        self.api_key = settings.OPENROUTER_API_KEY
        self.api_base = settings.OPENROUTER_BASE_URL
        self.default_model = settings.OPENROUTER_MODEL
        self.timeout = settings.OPENROUTER_TIMEOUT
        
        self.http_client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:8000",  # Default referer
                "X-Title": settings.PROJECT_NAME
            },
            timeout=self.timeout
        )
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion using OpenRouter."""
        try:
            response = await self.http_client.post(
                "/chat/completions",
                json={
                    "model": self.default_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenRouter."""
        try:
            response = await self.http_client.post(
                "/embeddings",
                json={
                    "model": "text-embedding-ada-002",
                    "input": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
            
        except Exception as e:
            raise Exception(f"OpenRouter embedding error: {str(e)}")

class LocalLLM(LLMInterface):
    """Local LLM implementation using HuggingFace models."""
    
    def __init__(self, settings: Settings):
        self.model_path = settings.LMSTUDIO_MODEL
        self.base_url = settings.LMSTUDIO_BASE_URL
        self.timeout = settings.LMSTUDIO_TIMEOUT
        
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion using local model."""
        try:
            response = await self.http_client.post(
                "/chat/completions",
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens or 512,
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise Exception(f"Local LLM error: {str(e)}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model."""
        try:
            response = await self.http_client.post(
                "/embeddings",
                json={
                    "input": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
            
        except Exception as e:
            raise Exception(f"Local embedding error: {str(e)}")

class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(settings: Settings) -> LLMInterface:
        """Create LLM instance based on settings."""
        if settings.LLM_PROVIDER == "openrouter":
            return OpenRouterLLM(settings)
        elif settings.LLM_PROVIDER == "lmstudio":
            return LocalLLM(settings)
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")