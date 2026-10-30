"""Validate and de-duplicate model-claimed citations against supplied labels."""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

from src.chains.schemas import Confidence, GroundedResponse, SourceCitation

logger = logging.getLogger(__name__)

_LABEL_RE = re.compile(r"\[?\s*S(\d+)\s*\]?", re.IGNORECASE)


def normalize_label(raw: str) -> str | None:
    match = _LABEL_RE.search(raw or "")
    if not match:
        return None
    return f"S{int(match.group(1))}"


def validate_and_dedupe_citations(
    response: GroundedResponse,
    supplied: dict[str, Document],
) -> GroundedResponse:
    """Keep only citations whose labels were actually supplied; dedupe by source+location."""
    valid: list[SourceCitation] = []
    seen: set[tuple[str, str]] = set()
    invalid_labels: list[str] = []

    for cite in response.sources:
        label = normalize_label(cite.label) or normalize_label(cite.source)
        if not label or label not in supplied:
            invalid_labels.append(cite.label)
            continue
        doc = supplied[label]
        source = str(doc.metadata.get("source", cite.source or ""))
        location = str(doc.metadata.get("location", cite.location or ""))
        key = (source, location)
        if key in seen:
            continue
        seen.add(key)
        snippet = cite.snippet or doc.page_content[:180].replace("\n", " ")
        valid.append(
            SourceCitation(
                label=label,
                source=source,
                location=location,
                snippet=snippet,
            )
        )

    flags = list(response.flags)
    if invalid_labels:
        logger.warning("Dropped invalid citations: %s", invalid_labels)
        flags.append("invalid_citations_dropped")

    confidence = response.confidence
    answered = response.answered

    if response.answered and not valid:
        # Model answered but no valid citations — lower trust
        flags.append("missing_valid_citations")
        if confidence == Confidence.high:
            confidence = Confidence.partial

    return response.model_copy(
        update={
            "sources": valid,
            "flags": flags,
            "confidence": confidence,
            "answered": answered,
        }
    )
