"""Typed, user-facing errors for SupportPearlz."""

from __future__ import annotations


class SupportPearlzError(Exception):
    """Base error with a short message safe to show in CLI/UI."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


class ConfigurationError(SupportPearlzError):
    """Missing or invalid configuration / secrets."""


class IndexError_(SupportPearlzError):
    """Vector index missing, corrupt, or embedding-model mismatch."""


class IngestionError(SupportPearlzError):
    """A single file failed during ingestion (may be non-fatal when skipped)."""


class RetrievalError(SupportPearlzError):
    """Retriever or relevance-gate failure."""


class GenerationError(SupportPearlzError):
    """LLM timeout, rate limit, or unrecoverable generation failure."""


class SchemaValidationError(SupportPearlzError):
    """Model output failed structured-schema validation after retry."""
