"""Phase 0 — configuration smoke and failure-mode tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings, clear_settings_cache, get_settings
from src.utils.errors import ConfigurationError


def test_settings_load_with_api_key(clean_settings_env: None) -> None:
    settings = get_settings()
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_temperature == 0.0
    assert settings.chunk_size == 800
    assert settings.retrieval_k == 4
    assert settings.enable_langchain_cache is True
    assert settings.enable_streaming is True
    assert "medical" in settings.medical_safety_disclaimer.lower() or (
        "clinical" in settings.medical_safety_disclaimer.lower()
    )


def test_masked_dict_hides_secret(clean_settings_env: None) -> None:
    settings = get_settings()
    masked = settings.masked_dict()
    assert masked["openai_api_key"] != "sk-test-key-for-unit-tests-only"
    assert masked["openai_api_key"] is not None
    assert "…" in str(masked["openai_api_key"]) or str(masked["openai_api_key"]) == "***"


def test_missing_api_key_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_cache()
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # Instantiate Settings directly with _env_file=None to ignore project .env
    with pytest.raises(ConfigurationError) as exc_info:
        Settings(_env_file=None, openai_api_key=None, llm_provider="openai")
    assert "OPENAI_API_KEY" in exc_info.value.message


def test_overlap_must_be_less_than_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_cache()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises((ValidationError, ConfigurationError)):
        Settings(
            _env_file=None,
            openai_api_key="sk-test",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_get_settings_is_cached(clean_settings_env: None) -> None:
    a = get_settings()
    b = get_settings()
    assert a is b
