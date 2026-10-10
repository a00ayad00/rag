"""
Configuration management for Bilingual Document Q&A RAG System.
Supports loading from environment variables and .env file.
"""

from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    ENGLISH_DATA_DIR: Path = BASE_DIR / "data" / "english"
    ARABIC_DATA_DIR: Path = BASE_DIR / "data" / "arabic"

    # Ingestion & Chunking
    CHUNK_SIZE: int = Field(default=600, description="Target chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=120, description="Overlap between consecutive chunks in characters")
    NORMALIZE_ARABIC: bool = Field(default=True, description="Whether to normalize Arabic orthography")

    # Qdrant Vector Store
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant host")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant port")
    QDRANT_URL: Optional[str] = Field(default=None, description="Direct Qdrant URL if cloud or remote")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API Key if authenticated")
    QDRANT_COLLECTION: str = Field(default="bilingual_rag_docs", description="Qdrant collection name")

    # Embedding Model (Baseline Dense)
    # Default: BAAI/bge-m3 (SOTA multilingual, handles Arabic + English natively, 1024-dim)
    EMBEDDING_MODEL_NAME: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model ID for multilingual embeddings"
    )
    EMBEDDING_DIM: int = Field(default=1024, description="Embedding vector dimension")
    EMBEDDING_DEVICE: str = Field(default="cpu", description="Torch device: cpu or cuda")

    # Improvement: Cross-Encoder Reranker
    # Default: BAAI/bge-reranker-v2-m3 (Multilingual cross-encoder SOTA)
    RERANKER_MODEL_NAME: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-Encoder model for reranking candidates"
    )
    USE_RERANKER: bool = Field(default=True, description="Default retrieval mode")
    RETRIEVAL_TOP_K: int = Field(default=5, description="Final number of retrieved chunks for generation")
    RETRIEVAL_CANDIDATE_K: int = Field(default=15, description="Initial candidates retrieved before reranking")

    # LLM Provider & Generation
    # Supported: "groq", "gemini", "ollama", "mock"
    LLM_PROVIDER: Literal["groq", "gemini", "ollama", "mock"] = Field(
        default="groq",
        description="LLM provider: groq (free fast), gemini (free tier), ollama (local), mock (testing)"
    )
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API key")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")

    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini model name")

    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    OLLAMA_MODEL: str = Field(default="llama3.2", description="Local Ollama model name")

    TEMPERATURE: float = Field(default=0.0, description="LLM sampling temperature for deterministic factual Q&A")
    MAX_TOKENS: int = Field(default=1024, description="Maximum response tokens")

    # Service & Logging
    API_HOST: str = Field(default="0.0.0.0", description="FastAPI host")
    API_PORT: int = Field(default=8000, description="FastAPI port")
    LOG_LEVEL: str = Field(default="INFO", description="Application log level")


settings = Settings()
