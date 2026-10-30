"""Chat model factory (OpenAI-compatible)."""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_chat_model(settings: Settings | None = None, *, streaming: bool = False) -> ChatOpenAI:
    cfg = settings or get_settings()
    kwargs: dict = {
        "model": cfg.llm_model,
        "temperature": cfg.llm_temperature,
        "api_key": cfg.openai_api_key.get_secret_value() if cfg.openai_api_key else None,
        "streaming": streaming,
        "max_retries": 2,
        "timeout": 60,
    }
    if cfg.openai_api_base:
        kwargs["base_url"] = cfg.openai_api_base
    logger.info("Chat model ready: %s temperature=%s", cfg.llm_model, cfg.llm_temperature)
    return ChatOpenAI(**kwargs)
