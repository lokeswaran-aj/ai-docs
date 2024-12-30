from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and environment variables."""

    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0

    openai_api_key: str

    database_url: str
    database_sslmode: str = "disable"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
