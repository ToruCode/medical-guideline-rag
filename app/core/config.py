"""Application settings loaded from environment variables and .env files."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration.

    Environment variables use the MEDICAL_RAG_ prefix and are matched
    case-insensitively. Unknown environment variables are ignored so
    that unrelated variables in the environment do not break startup.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEDICAL_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Medical Guideline RAG"
    app_version: str = "0.1.0"
    environment: Literal["local", "test", "staging", "production"] = "local"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_v1_prefix: str = "/api/v1"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # "pymupdf" (default; better Japanese text-layer extraction quality
    # and retrieval accuracy than pypdf in a local single-document
    # comparison) or "pypdf" (kept for rollback/compatibility). See
    # docs/adr/0017-pdf-extraction-comparison-tooling.md for the
    # comparison methodology and
    # docs/adr/0018-adopt-pymupdf-for-production-pdf-extraction.md for
    # the production adoption decision, including an open PyMuPDF
    # license follow-up (AGPL-3.0/commercial dual license) that must be
    # resolved before any commercial/public deployment.
    pdf_extractor: Literal["pymupdf", "pypdf"] = "pymupdf"
    # "fake" (default, no dependencies) or "sentence_transformers" (local
    # multilingual model, downloaded on first use). See
    # docs/adr/0011-real-embedding-and-llm-adapters.md.
    embedding_provider: Literal["fake", "sentence_transformers"] = "fake"
    embedding_model_name: str = "intfloat/multilingual-e5-base"
    # "fake" (default, no dependencies, no network access) or "openai".
    # llm_api_key is a SecretStr (not str) so that accidentally logging or
    # printing the Settings object never leaks its value; call
    # .get_secret_value() to obtain the raw string when constructing an
    # OpenAiLlm. See docs/adr/0011-real-embedding-and-llm-adapters.md.
    llm_provider: Literal["fake", "openai"] = "fake"
    llm_model_name: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
