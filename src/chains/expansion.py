"""C5 query expansion: generate variants and fuse retrieval sets."""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_EXPAND = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the customer support question into exactly 2 alternative search queries. "
            "Keep product names and error codes. One line per query. No numbering.",
        ),
        ("human", "{question}"),
    ]
)


def expand_queries(question: str, llm: BaseChatModel) -> list[str]:
    """Return [original] + up to 2 variants."""
    try:
        result = (_EXPAND | llm).invoke({"question": question})
        text = getattr(result, "content", str(result))
        variants = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
        variants = [v for v in variants if v and v.lower() != question.lower()][:2]
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
        variants = []
    out = [question, *variants]
    logger.info("query_expansion base=%r variants=%s", question, variants)
    return out


def fuse_documents(doc_lists: Sequence[Sequence[Document]], *, limit: int = 20) -> list[Document]:
    """RRF-style fusion across expanded query result lists."""
    scores: dict[str, float] = {}
    keep: dict[str, Document] = {}
    for docs in doc_lists:
        for rank, doc in enumerate(docs, start=1):
            key = f"{doc.metadata.get('source','')}|{doc.page_content[:120]}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            keep[key] = doc
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [keep[k] for k, _ in ordered[:limit]]
