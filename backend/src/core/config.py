from typing import Optional, Any, List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PDF Chat API"

    # CORS Origins
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # LLM Provider Settings
    LLM_PROVIDER: str = Field(
        default="lmstudio",
        description="LLM provider to use (lmstudio or openrouter)"
    )
    
    # LMStudio settings
    LMSTUDIO_BASE_URL: str = Field(
        default="http://localhost:1234/v1",
        description="LM Studio API base URL"
    )
    LMSTUDIO_MODEL: str = Field(
        default="llama3-docchat-1.0-8b-i1",
        description="Model to use for chat completions"
    )
    LMSTUDIO_TIMEOUT: float = Field(
        default=30.0,
        description="Timeout for LM Studio API requests"
    )

    # OpenRouter settings
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",  # Correct base URL
        description="OpenRouter API base URL"
    )
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="OpenRouter API key"
    )
    OPENROUTER_MODEL: str = Field(
        default="qwen/qwq-32b:free",  # Updated to correct model ID
        description="OpenRouter model to use"
    )
    OPENROUTER_TIMEOUT: float = Field(
        default=30.0,
        description="Timeout for OpenRouter API requests"
    )

    # Database Settings
    POSTGRES_HOST: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    POSTGRES_PORT: str = Field(
        default="5432",
        description="PostgreSQL port"
    )
    POSTGRES_USER: str = Field(
        default="postgres",
        description="PostgreSQL user"
    )
    POSTGRES_PASSWORD: str = Field(
        default="postgres",
        description="PostgreSQL password"
    )
    POSTGRES_DB: str = Field(
        default="pdf_chat",
        description="PostgreSQL database name"
    )
    
    # Construct database URL
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Vector Settings
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Dimension of text embeddings"
    )
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en",
        description="Model to use for embeddings"
    )
    CHUNK_SIZE: int = Field(
        default=500,
        description="Number of characters per text chunk"
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="Number of overlapping characters between chunks"
    )

    # Chunking Settings
    CHUNKING_METHOD: Literal["semantic", "agentic"] = Field(
        default="semantic",
        description="Method to use for text chunking (semantic or agentic)"
    )
    AGENTIC_CHUNKING_LANGUAGE: str = Field(
        default="vietnamese",
        description="Language for agentic chunking (vietnamese or english)"
    )
    AGENTIC_MIN_CHUNK_SIZE: int = Field(
        default=100,
        description="Minimum size for agentic chunks"
    )
    AGENTIC_MAX_CHUNK_SIZE: int = Field(
        default=1500,
        description="Maximum size for agentic chunks"
    )

    # RAG Settings
    TOP_K_MATCHES: int = Field(
        default=20,  # Increased to get more context
        description="Number of chunks to retrieve for context"
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.0,  # No threshold to get all chunks
        description="Minimum similarity score for context retrieval"
    )

    # Prompts
    SYSTEM_PROMPT_VI: str = """Bạn là một trợ lý AI chuyên nghiệp, giúp người dùng hiểu nội dung văn bản. 
Khi trả lời câu hỏi, hãy:
1. Tập trung vào thông tin được hỏi
2. Trích dẫn các con số cụ thể nếu có
3. Liệt kê đầy đủ các đối tượng được đề cập
4. Sắp xếp thông tin một cách logic
5. Sử dụng ngôn ngữ rõ ràng, chính xác

Nếu văn bản không có thông tin cần thiết, hãy nêu rõ điều này."""

    SYSTEM_PROMPT_EN: str = """You are a professional AI assistant helping users understand document content.
When answering questions:
1. Focus on the requested information
2. Quote specific numbers when available
3. List all mentioned entities
4. Organize information logically
5. Use clear and precise language

If the text lacks necessary information, clearly state this."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Create global settings instance
settings = Settings()