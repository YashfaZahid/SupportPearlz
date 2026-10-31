"""C1 hybrid search: BM25 lexical + dense vector ensemble."""

from __future__ import annotations

import logging
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


def _to_chroma_where(filter_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chroma allows one top-level operator; combine multi-field filters with ``$and``."""
    if not filter_dict:
        return None
    if len(filter_dict) == 1:
        return dict(filter_dict)
    return {"$and": [{key: value} for key, value in filter_dict.items()]}


def documents_from_chroma(vector_store: Chroma) -> list[Document]:
    raw = vector_store.get()
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    docs: list[Document] = []
    for text, meta in zip(texts, metas):
        if not text:
            continue
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def build_hybrid_retriever(
    vector_store: Chroma,
    *,
    k: int = 4,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
    filter_dict: dict[str, Any] | None = None,
) -> BaseRetriever:
    """Ensemble BM25 + similarity retriever (helps SKUs / error codes like E-03)."""
    docs = documents_from_chroma(vector_store)
    if filter_dict:
        docs = [
            d
            for d in docs
            if all(str(d.metadata.get(k)) == str(v) for k, v in filter_dict.items())
        ]
    where = _to_chroma_where(filter_dict)
    if not docs:
        logger.warning("Hybrid BM25 corpus empty after filter; falling back to dense only")
        search_kwargs: dict[str, Any] = {"k": k}
        if where:
            search_kwargs["filter"] = where
        return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    search_kwargs = {"k": k}
    if where:
        search_kwargs["filter"] = where
    dense = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    logger.info(
        "Hybrid retriever ready: bm25_docs=%s weights=(%.2f, %.2f) k=%s",
        len(docs),
        bm25_weight,
        dense_weight,
        k,
    )
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_weight, dense_weight],
    )
