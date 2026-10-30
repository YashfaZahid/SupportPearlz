"""Retriever policies, metadata filters, and relevance gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

RetrievalStrategy = Literal["similarity", "mmr"]


@dataclass
class ScoredChunk:
    document: Document
    score: float  # Chroma cosine *distance* (lower = more similar)
    rank: int


@dataclass
class RetrievalResult:
    query: str
    chunks: list[ScoredChunk]
    strategy: str
    relevant: bool
    gate_reason: str
    best_score: float | None

    @property
    def documents(self) -> list[Document]:
        return [c.document for c in self.chunks]


def build_retriever(
    vector_store: Chroma,
    *,
    settings: Settings | None = None,
    strategy: RetrievalStrategy | None = None,
    k: int | None = None,
    filter_dict: dict[str, Any] | None = None,
):
    """Construct a LangChain retriever (similarity or MMR) with optional metadata filter."""
    cfg = settings or get_settings()
    strategy = strategy or cfg.retrieval_strategy
    k = k or cfg.retrieval_k
    search_kwargs: dict[str, Any] = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    if strategy == "mmr":
        search_kwargs["fetch_k"] = cfg.mmr_fetch_k
        search_kwargs["lambda_mult"] = cfg.mmr_lambda
        return vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def retrieve_with_scores(
    vector_store: Chroma,
    query: str,
    *,
    settings: Settings | None = None,
    strategy: RetrievalStrategy | None = None,
    k: int | None = None,
    filter_dict: dict[str, Any] | None = None,
) -> list[ScoredChunk]:
    """Run retrieval and attach scores.

    Note: with ``hnsw:space=cosine``, Chroma returns **distance** (lower is better).
    MMR path may not return scores from the retriever API — we fall back to a
    similarity_search_with_score pass for gating when needed.
    """
    cfg = settings or get_settings()
    strategy = strategy or cfg.retrieval_strategy
    k = k or cfg.retrieval_k

    if strategy == "similarity":
        kwargs: dict[str, Any] = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        pairs = vector_store.similarity_search_with_score(query, **kwargs)
        return [
            ScoredChunk(document=doc, score=float(score), rank=i)
            for i, (doc, score) in enumerate(pairs, start=1)
        ]

    # MMR: retrieve diverse set, then score the returned docs via similarity for the gate
    retriever = build_retriever(
        vector_store,
        settings=cfg,
        strategy="mmr",
        k=k,
        filter_dict=filter_dict,
    )
    docs = retriever.invoke(query)
    # Score each by a fresh similarity search window and match on content+source
    scored_pool = vector_store.similarity_search_with_score(query, k=max(k * 3, cfg.mmr_fetch_k))
    score_by_key: dict[tuple[str, str], float] = {}
    for doc, score in scored_pool:
        key = (str(doc.metadata.get("source", "")), doc.page_content[:160])
        score_by_key[key] = float(score)

    chunks: list[ScoredChunk] = []
    for i, doc in enumerate(docs, start=1):
        key = (str(doc.metadata.get("source", "")), doc.page_content[:160])
        score = score_by_key.get(key, 1.0)
        chunks.append(ScoredChunk(document=doc, score=score, rank=i))
    return chunks


def relevance_gate(
    chunks: list[ScoredChunk],
    *,
    settings: Settings | None = None,
    max_distance: float | None = None,
) -> tuple[bool, str]:
    """Decide whether retrieval succeeded.

    Gate rule (empirically calibrated for this Chroma + cosine-distance setup):
    - Fail if no chunks
    - Fail if best (lowest) distance exceeds ``SCORE_THRESHOLD`` (interpreted as
      **maximum acceptable distance**)

    Observed good hits for in-KB questions often land ~0.35–0.55 distance.
    Unrelated questions tend to be higher. Default threshold 0.75 rejects weak matches.
    """
    cfg = settings or get_settings()
    threshold = max_distance if max_distance is not None else cfg.score_threshold

    if not chunks:
        return False, "empty_result_set"

    best = min(c.score for c in chunks)
    if best > threshold:
        return False, f"best_distance={best:.4f}_exceeds_threshold={threshold}"

    return True, f"best_distance={best:.4f}_within_threshold={threshold}"


def retrieve(
    vector_store: Chroma,
    query: str,
    *,
    settings: Settings | None = None,
    strategy: RetrievalStrategy | None = None,
    k: int | None = None,
    filter_dict: dict[str, Any] | None = None,
) -> RetrievalResult:
    """Full retrieve + gate. Logs chunk ids and scores for the turn contract."""
    cfg = settings or get_settings()
    strategy = strategy or cfg.retrieval_strategy
    chunks = retrieve_with_scores(
        vector_store,
        query,
        settings=cfg,
        strategy=strategy,
        k=k,
        filter_dict=filter_dict,
    )
    relevant, reason = relevance_gate(chunks, settings=cfg)
    best = min((c.score for c in chunks), default=None)

    for ch in chunks:
        src = ch.document.metadata.get("source")
        loc = ch.document.metadata.get("location")
        logger.info(
            "retrieved rank=%s score=%.4f source=%s location=%s",
            ch.rank,
            ch.score,
            src,
            loc,
        )
    logger.info(
        "relevance_gate relevant=%s reason=%s strategy=%s query=%r",
        relevant,
        reason,
        strategy,
        query,
    )

    return RetrievalResult(
        query=query,
        chunks=chunks if relevant else chunks,  # keep chunks for debugging; caller may ignore
        strategy=strategy,
        relevant=relevant,
        gate_reason=reason,
        best_score=best,
    )


def hit_at_k(
    vector_store: Chroma,
    questions: list[dict[str, Any]],
    *,
    k: int = 4,
    strategy: RetrievalStrategy = "similarity",
    settings: Settings | None = None,
) -> float:
    """Simple hit@k: fraction of questions where any top-k chunk source is in expected_sources."""
    hits = 0
    for item in questions:
        q = item["question"]
        expected = set(item.get("expected_sources") or [])
        must_contain = item.get("must_contain")
        chunks = retrieve_with_scores(
            vector_store, q, settings=settings, strategy=strategy, k=k
        )
        ok = False
        for ch in chunks:
            src = str(ch.document.metadata.get("source", ""))
            if expected and src in expected:
                if must_contain:
                    if str(must_contain).lower() in ch.document.page_content.lower():
                        ok = True
                        break
                else:
                    ok = True
                    break
        if ok:
            hits += 1
    return hits / max(len(questions), 1)
