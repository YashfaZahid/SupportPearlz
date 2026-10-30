"""Context formatting with [Sn] labels and budget enforcement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.retrieval.retriever import ScoredChunk

logger = logging.getLogger(__name__)


@dataclass
class LabeledContext:
    text: str
    labels: dict[str, Document]  # "S1" -> Document
    truncated: bool


def format_context(
    chunks: list[ScoredChunk] | list[Document],
    *,
    settings: Settings | None = None,
) -> LabeledContext:
    """Concatenate retrieved chunks as labelled blocks within the context budget."""
    cfg = settings or get_settings()
    max_chunks = cfg.context_max_chunks
    max_chars = cfg.context_max_chars

    docs: list[Document] = []
    for item in chunks[:max_chunks]:
        if isinstance(item, ScoredChunk):
            docs.append(item.document)
        else:
            docs.append(item)

    parts: list[str] = []
    labels: dict[str, Document] = {}
    used = 0
    truncated = False

    for i, doc in enumerate(docs, start=1):
        label = f"S{i}"
        source = doc.metadata.get("source", "unknown")
        location = doc.metadata.get("location", "")
        header = f"[{label}] source={source} location={location}"
        body = doc.page_content.strip()
        block = f"{header}\n{body}"
        # +2 for joining newlines
        addition = len(block) + (2 if parts else 0)
        if used + addition > max_chars:
            truncated = True
            logger.info(
                "Context truncated at chunk %s (budget=%s chars)",
                i,
                max_chars,
            )
            break
        parts.append(block)
        labels[label] = doc
        used += addition

    if len(docs) > max_chunks:
        truncated = True

    text = "\n\n".join(parts) if parts else "(no context)"
    return LabeledContext(text=text, labels=labels, truncated=truncated)
