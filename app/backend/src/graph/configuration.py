from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and environment variables."""

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0

    openai_api_key: str

    database_url: str
    database_sslmode: str = "disable"

    embedding_model: str = "text-embedding-3-large"
    top_k_docs: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
