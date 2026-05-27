"""
Application configuration for the grounded RAG API.

Centralized settings keep deployment environments (local, Docker, cloud)
consistent without scattering magic paths across modules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (parent of api/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-driven settings with sensible defaults for local dev."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Grounded RAG API"
    app_version: str = "1.0.0"
    debug: bool = False

    chroma_dir: Path = Field(default=PROJECT_ROOT / "chroma_db")
    chunks_path: Path = Field(default=PROJECT_ROOT / "processed_chunks.json")
    collection_name: str = "quality_articles"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    default_candidate_k: int = 25
    default_top_k: int = 5
    max_candidate_k: int = 50
    max_top_k: int = 20

    # LLM (Groq default; Ollama optional for local CLI)
    llm_provider: str = "groq"
    model_name: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: float = 60.0
    groq_api_base_url: str = "https://api.groq.com/openai/v1"

    ollama_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 120.0

    log_level: str = "INFO"

    # Uvicorn bind (Docker / production)
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (safe to call on every request)."""
    return Settings()
