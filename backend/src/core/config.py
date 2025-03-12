from typing import Optional, Any, List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PDF Chat API"

    # CORS Origins
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # LM Studio Settings
    LMSTUDIO_BASE_URL: str = Field(
        default="http://localhost:1234/v1",
        description="LM Studio API base URL"
    )
    LMSTUDIO_MODEL: str = Field(
        default="llama3-docchat-1.0-8b-i1",
        description="Model to use for chat completions"
    )
    LMSTUDIO_TIMEOUT: float = Field(
        default=120.0,
        description="Timeout for LM Studio API requests (increased for larger models)"
    )

    # Database Settings
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: str = Field(default="5432", description="PostgreSQL port")
    POSTGRES_USER: str = Field(..., description="PostgreSQL user")
    POSTGRES_PASSWORD: str = Field(..., description="PostgreSQL password")
    POSTGRES_DB: str = Field(default="pdf_chat", description="PostgreSQL database name")
    
    # Construct database URL
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Vector Settings
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Dimension of text embeddings"
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Create global settings instance
settings = Settings()