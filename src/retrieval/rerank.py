"""C2 reranking: retrieve many, keep top-n with a local fusion score.

Uses lexical overlap + reciprocal rank — no heavy cross-encoder download required,
while preserving the retrieve-many-then-rerank pattern from the assignment.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9\-]+", re.I)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1}


def fusion_score(query: str, doc: Document, rank: int) -> float:
    q = _tokens(query)
    d = _tokens(doc.page_content)
    if not q:
        overlap = 0.0
    else:
        overlap = len(q & d) / len(q)
    # Exact-ish boosts for SKUs / error codes
    boost = 0.0
    for tok in q:
        if re.match(r"^(e-\d+|ap\d+|sku)$", tok, re.I) or tok.upper().startswith("AP"):
            if tok.lower() in doc.page_content.lower():
                boost += 0.25
    rrf = 1.0 / (60 + rank)
    return 0.55 * overlap + 0.35 * rrf + boost


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    *,
    top_n: int = 4,
) -> list[Document]:
    scored = [
        (fusion_score(query, doc, i + 1), doc)
        for i, doc in enumerate(documents)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    kept = [doc for _, doc in scored[:top_n]]
    logger.info(
        "Reranked %s → %s candidates (fusion reranker)",
        len(documents),
        len(kept),
    )
    return kept
