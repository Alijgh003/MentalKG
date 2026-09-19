from pydantic_settings import BaseSettings, SettingsConfigDict

from .env import PROJECT_ENV_FILE


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "openai/gpt-oss-120b"
    llm_api_base: str = "http://192.168.1.204:43375/v1"
    llm_api_key: str = "not-needed"
    llm_timeout: int = 30
    llm_retry_attempts: int = 5

    # Only when use RateLimitedLM in data_ingestion.core.llm
    llm_max_concurrent: int = 50


settings = Settings()


class DatabaseSettings(BaseSettings):
    """DSM knowledge-graph database settings."""

    model_config = SettingsConfigDict(
        env_prefix="DSM_KG_",
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    database_url: str
