"""Configuration management using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Semantic Highlighting
    semantic_highlight_model: str = "zilliz/semantic-highlight-bilingual-v1"
    semantic_threshold: float = 0.5

    # HHEM
    hhem_model: str = "vectara/hallucination_evaluation_model"
    hhem_threshold: float = 0.5

    # API
    upload_dir: str = "./uploads"
    max_file_size: int = 10485760  # 10MB

    # Computed properties
    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
