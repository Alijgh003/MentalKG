from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    llm_model: str = "openai/gpt-oss-120b"
    llm_api_base: str = "http://192.168.1.204:43375/v1"
    llm_api_key: str = "not-needed"
    llm_timeout: int = 30
    llm_retry_attempts: int = 5

    # Only when use RateLimitedLM in data_ingestion.core.llm
    llm_max_concurrent: int = 50


settings = Settings()
