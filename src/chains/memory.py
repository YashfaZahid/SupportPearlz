"""Chat history windowing and query condensation."""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.chains.prompts import CONDENSE_PROMPT
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def window_history(
    history: Sequence[BaseMessage] | Sequence[tuple[str, str]],
    *,
    max_turns: int | None = None,
    settings: Settings | None = None,
) -> list[BaseMessage]:
    """Keep the last N user/assistant turns (2 messages per turn)."""
    cfg = settings or get_settings()
    max_turns = max_turns or cfg.history_max_turns
    messages = _to_messages(history)
    max_messages = max_turns * 2
    if len(messages) <= max_messages:
        return list(messages)
    return list(messages[-max_messages:])


def _to_messages(
    history: Sequence[BaseMessage] | Sequence[tuple[str, str]],
) -> list[BaseMessage]:
    if not history:
        return []
    first = history[0]
    if isinstance(first, BaseMessage):
        return list(history)  # type: ignore[arg-type]
    out: list[BaseMessage] = []
    for user, assistant in history:  # type: ignore[misc]
        out.append(HumanMessage(content=user))
        out.append(AIMessage(content=assistant))
    return out


def condense_question(
    question: str,
    history: Sequence[BaseMessage] | Sequence[tuple[str, str]],
    llm: BaseChatModel,
    *,
    settings: Settings | None = None,
) -> str:
    """Rewrite follow-ups into a standalone retrieval query. Logs both forms."""
    cfg = settings or get_settings()
    messages = window_history(history, settings=cfg)
    if not messages:
        logger.info("query_original=%r query_rewritten=%r (no history)", question, question)
        return question

    chain = CONDENSE_PROMPT | llm
    result = chain.invoke({"chat_history": messages, "question": question})
    rewritten = getattr(result, "content", str(result)).strip() or question
    logger.info("query_original=%r query_rewritten=%r", question, rewritten)
    return rewritten
