"""LangChain cache setup (caching strategy 1). Strategy 2 is Streamlit @st.cache_resource."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def setup_langchain_cache(settings: Settings | None = None) -> bool:
    """Enable SQLite cache for LLM/embedding calls when configured.

    Returns True if cache was enabled.
    """
    cfg = settings or get_settings()
    if not cfg.enable_langchain_cache:
        logger.info("LangChain cache disabled via ENABLE_LANGCHAIN_CACHE")
        return False

    path = cfg.langchain_cache_path
    if not path.is_absolute():
        from src.config import PROJECT_ROOT

        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from langchain_community.cache import SQLiteCache
        from langchain.globals import set_llm_cache

        set_llm_cache(SQLiteCache(database_path=str(path)))
        logger.info("LangChain SQLite cache enabled at %s", path)
        return True
    except Exception as exc:
        logger.warning("Could not enable LangChain cache: %s", exc)
        return False
