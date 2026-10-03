from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    corpus_dir: Path = Path("corpus")
    index_dir: Path = Path("var/index")
    chunk_size: int = 180
    chunk_overlap: int = 40
    top_k: int = 4
    rrf_k: int = 60
    prompt_version: str = "grounded-v1"
    provider: str = "extractive"
    llm_url: str = ""
    llm_api_key: str = ""
    max_context_chars: int = 2400
    request_timeout_s: float = 8.0
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
