"""Shared pytest fixtures for SupportPearlz."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from src.config import clear_settings_cache


@pytest.fixture
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide a minimal valid env and clear the settings LRU cache."""
    clear_settings_cache()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-unit-tests-only")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    # Avoid reading a developer's real .env unexpectedly for assertions
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    yield
    clear_settings_cache()


@pytest.fixture
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure OPENAI_API_KEY is absent for negative tests."""
    clear_settings_cache()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # pydantic-settings may still load from .env file — point env_file away in test
    # by setting an empty key after disabling file via monkeypatch of cwd .env is hard;
    # tests that need absolute absence use monkeypatch on Settings. See test_config.
    yield
    clear_settings_cache()
