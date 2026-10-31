"""Environment loading and validated settings for SupportPearlz.

Run as ``python -m src.config`` for a smoke check (secrets masked).
Fails loudly at import/load time if a required key is missing.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.errors import ConfigurationError

# Project root: parent of ``src/``
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DISCLAIMER = (
    "SupportPearlz answers from Pearlz product documentation only. "
    "It is not a medical device advisor and must not be used for clinical decisions. "
    "Never invent health claims, certifications, or treatment advice. "
    "If water quality or health concerns persist, contact a qualified professional "
    "and Pearlz support."
)


class Settings(BaseSettings):
    """Single validated settings object. All tunables live here — nowhere else."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    llm_provider: Literal["openai", "groq", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_api_key: SecretStr | None = None
    openai_api_base: str | None = None

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, gt=0)

    # Vector store
    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    collection_name: str = "pearlz_kb"
    knowledge_base_path: Path = PROJECT_ROOT / "data" / "knowledge_base"

    # Chunking
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)

    # Retrieval
    retrieval_k: int = Field(default=4, gt=0)
    retrieval_strategy: Literal["similarity", "mmr", "hybrid"] = "hybrid"
    # Interpreted as max acceptable Chroma cosine *distance* (lower distance = better).
    score_threshold: float = Field(default=0.75, ge=0.0, le=2.0)
    mmr_fetch_k: int = Field(default=12, gt=0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    # C1 / C2 / C5 bonuses
    enable_hybrid_search: bool = True
    enable_rerank: bool = True
    rerank_candidates: int = Field(default=20, gt=0)
    enable_query_expansion: bool = True
    # C6 role: customer sees public only; agent sees public+internal
    default_access_role: Literal["customer", "agent"] = "customer"
    # C7 tools
    enable_tools: bool = True
    # C9 query-result cache
    enable_query_cache: bool = True

    # Context & history
    history_max_turns: int = Field(default=6, gt=0)
    context_max_chars: int = Field(default=12000, gt=0)
    context_max_chunks: int = Field(default=6, gt=0)

    # Caching strategy 1 — LangChain cache
    enable_langchain_cache: bool = True
    langchain_cache_path: Path = PROJECT_ROOT / "data" / "cache" / "langchain.db"

    # Streaming (UI)
    enable_streaming: bool = True

    # Support / safety
    support_escalation_email: str = "support@pearlzhome.example"
    medical_safety_disclaimer: str = DEFAULT_DISCLAIMER

    # Logging
    log_level: str = "INFO"
    log_file: Path = PROJECT_ROOT / "logs" / "supportpearlz.log"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:  # noqa: ANN001
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v

    @model_validator(mode="after")
    def require_api_key_for_hosted(self) -> Settings:
        """Hosted OpenAI-compatible providers need a key at startup — fail loud."""
        needs_key = self.llm_provider in ("openai", "groq")
        key = self.openai_api_key.get_secret_value() if self.openai_api_key else ""
        if needs_key and not key.strip():
            raise ConfigurationError(
                "Missing required OPENAI_API_KEY. "
                "Copy .env.example to .env and set your API key "
                "(used for OpenAI and OpenAI-compatible providers such as Groq)."
            )
        return self

    def masked_dict(self) -> dict[str, object]:
        """Settings for display/logging with secrets redacted."""
        data: dict[str, object] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                raw = value.get_secret_value()
                data[name] = _mask_secret(raw) if raw else None
            elif isinstance(value, Path):
                data[name] = str(value)
            else:
                data[name] = value
        return data

    def resolve_path(self, path: Path) -> Path:
        """Resolve relative paths against the project root."""
        return path if path.is_absolute() else (PROJECT_ROOT / path)


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once. Raises ConfigurationError on validation failure."""
    try:
        return Settings()
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic ValidationError, etc.
        raise ConfigurationError(
            f"Invalid configuration: {exc}",
            detail=str(exc),
        ) from exc


def clear_settings_cache() -> None:
    """Clear cached settings (for tests)."""
    get_settings.cache_clear()


def main() -> int:
    """Smoke check: print masked configuration."""
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"CONFIGURATION ERROR: {exc.message}", file=sys.stderr)
        return 1

    print("SupportPearlz configuration (secrets masked)")
    print("-" * 48)
    for key, value in sorted(settings.masked_dict().items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
