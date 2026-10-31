"""Retriever policies, hybrid search, rerank, metadata filters, and relevance gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import Settings, get_settings
from src.retrieval.hybrid import build_hybrid_retriever
from src.retrieval.rerank import rerank_documents

logger = logging.getLogger(__name__)

RetrievalStrategy = Literal["similarity", "mmr", "hybrid"]


@dataclass
class ScoredChunk:
    document: Document
    score: float  # Chroma cosine *distance* (lower = more similar) when available
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


def apply_access_filter(
    filter_dict: dict[str, Any] | None,
    *,
    role: str,
) -> dict[str, Any] | None:
    """Pass through doc_type/product filters only.

    ``access_level`` is enforced **after** retrieval (see ``retrieve``) so indexes
    built before C6 metadata still work. After ``build_index --rebuild``, chunks
    carry ``access_level`` and post-filter drops internal for customers.
    """
    _ = role  # role applied post-retrieval
    return dict(filter_dict) if filter_dict else None


def to_chroma_where(filter_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chroma allows one top-level operator; combine multi-field filters with ``$and``."""
    if not filter_dict:
        return None
    if len(filter_dict) == 1:
        key = next(iter(filter_dict))
        if str(key).startswith("$"):
            return dict(filter_dict)
        return dict(filter_dict)
    return {"$and": [{key: value} for key, value in filter_dict.items()]}


def build_retriever(
    vector_store: Chroma,
    *,
    settings: Settings | None = None,
    strategy: RetrievalStrategy | None = None,
    k: int | None = None,
    filter_dict: dict[str, Any] | None = None,
):
    """Construct a LangChain retriever with optional metadata filter."""
    cfg = settings or get_settings()
    strategy = strategy or cfg.retrieval_strategy
    k = k or cfg.retrieval_k
    search_kwargs: dict[str, Any] = {"k": k}
    where = to_chroma_where(filter_dict)
    if where:
        search_kwargs["filter"] = where

    if strategy == "hybrid" or (cfg.enable_hybrid_search and strategy == "hybrid"):
        return build_hybrid_retriever(
            vector_store, k=k, filter_dict=filter_dict
        )

    if strategy == "mmr":
        search_kwargs["fetch_k"] = cfg.mmr_fetch_k
        search_kwargs["lambda_mult"] = cfg.mmr_lambda
        return vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def _score_docs(
    vector_store: Chroma,
    query: str,
    docs: list[Document],
    *,
    pool_k: int,
) -> list[ScoredChunk]:
    scored_pool = vector_store.similarity_search_with_score(query, k=pool_k)
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


def retrieve_with_scores(
    vector_store: Chroma,
    query: str,
    *,
    settings: Settings | None = None,
    strategy: RetrievalStrategy | None = None,
    k: int | None = None,
    filter_dict: dict[str, Any] | None = None,
) -> list[ScoredChunk]:
    """Run retrieval (optionally hybrid + rerank) and attach scores for the gate."""
    cfg = settings or get_settings()
    strategy = strategy or cfg.retrieval_strategy
    k = k or cfg.retrieval_k
    candidate_k = cfg.rerank_candidates if cfg.enable_rerank else k

    if strategy == "similarity" and not cfg.enable_hybrid_search:
        kwargs: dict[str, Any] = {"k": candidate_k}
        where = to_chroma_where(filter_dict)
        if where:
            kwargs["filter"] = where
        pairs = vector_store.similarity_search_with_score(query, **kwargs)
        docs = [doc for doc, _ in pairs]
        if cfg.enable_rerank:
            docs = rerank_documents(query, docs, top_n=k)
            return _score_docs(vector_store, query, docs, pool_k=max(candidate_k, 12))
        return [
            ScoredChunk(document=doc, score=float(score), rank=i)
            for i, (doc, score) in enumerate(pairs[:k], start=1)
        ]

    # hybrid or mmr path
    use_strategy: RetrievalStrategy = strategy
    if cfg.enable_hybrid_search and strategy in ("hybrid", "similarity"):
        use_strategy = "hybrid"

    if use_strategy == "hybrid":
        retriever = build_hybrid_retriever(
            vector_store, k=candidate_k, filter_dict=filter_dict
        )
        docs = list(retriever.invoke(query))
    else:
        retriever = build_retriever(
            vector_store,
            settings=cfg,
            strategy="mmr",
            k=candidate_k,
            filter_dict=filter_dict,
        )
        docs = list(retriever.invoke(query))

    if cfg.enable_rerank:
        docs = rerank_documents(query, docs, top_n=k)
    else:
        docs = docs[:k]

    return _score_docs(vector_store, query, docs, pool_k=max(candidate_k, 12))


def relevance_gate(
    chunks: list[ScoredChunk],
    *,
    settings: Settings | None = None,
    max_distance: float | None = None,
) -> tuple[bool, str]:
    """Fail if empty or best cosine distance exceeds SCORE_THRESHOLD."""
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
    role: str | None = None,
) -> RetrievalResult:
    """Full retrieve + access filter + gate. Logs chunk ids and scores."""
    cfg = settings or get_settings()
    role = role or cfg.default_access_role
    strategy = strategy or cfg.retrieval_strategy
    filtered = apply_access_filter(filter_dict, role=role)

    chunks = retrieve_with_scores(
        vector_store,
        query,
        settings=cfg,
        strategy=strategy,
        k=k,
        filter_dict=filtered,
    )
    # Defense in depth: drop internal if customer role somehow retrieved them
    if role == "customer":
        chunks = [
            c
            for c in chunks
            if str(c.document.metadata.get("access_level", "public")) != "internal"
        ]

    relevant, reason = relevance_gate(chunks, settings=cfg)
    best = min((c.score for c in chunks), default=None)

    for ch in chunks:
        logger.info(
            "retrieved rank=%s score=%.4f source=%s location=%s access=%s",
            ch.rank,
            ch.score,
            ch.document.metadata.get("source"),
            ch.document.metadata.get("location"),
            ch.document.metadata.get("access_level"),
        )
    logger.info(
        "relevance_gate relevant=%s reason=%s strategy=%s role=%s query=%r",
        relevant,
        reason,
        strategy,
        role,
        query,
    )

    return RetrievalResult(
        query=query,
        chunks=chunks,
        strategy=str(strategy),
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
    """hit@k without LLM."""
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
