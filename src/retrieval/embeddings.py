"""Embedding model factory and similarity diagnostics."""

from __future__ import annotations

import logging
import math
from typing import Sequence

from langchain_openai import OpenAIEmbeddings

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """Return the configured embedding model (same instance family for index + query)."""
    cfg = settings or get_settings()
    kwargs: dict = {
        "model": cfg.embedding_model,
        "api_key": cfg.openai_api_key.get_secret_value() if cfg.openai_api_key else None,
    }
    if cfg.openai_api_base:
        kwargs["base_url"] = cfg.openai_api_base
    embeddings = OpenAIEmbeddings(**kwargs)
    logger.info(
        "Embedding model ready: %s (expected dim=%s)",
        cfg.embedding_model,
        cfg.embedding_dimension,
    )
    return embeddings


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def run_similarity_diagnostic(
    embeddings: OpenAIEmbeddings | None = None,
) -> dict[str, float]:
    """Embed two paraphrases + one unrelated sentence; return pairwise cosines."""
    emb = embeddings or get_embeddings()
    s1 = "The AquaPearl 500 Pro beeps three times when the sediment filter is full."
    s2 = "Three short beeps on the AquaPearl mean the pre-filter is saturated."
    s3 = "Pearlz ships free when the order total exceeds fifteen thousand rupees."
    v1, v2, v3 = emb.embed_documents([s1, s2, s3])
    scores = {
        "paraphrase_pair": cosine_similarity(v1, v2),
        "unrelated_1_3": cosine_similarity(v1, v3),
        "unrelated_2_3": cosine_similarity(v2, v3),
        "dimension": float(len(v1)),
    }
    logger.info("Similarity diagnostic: %s", scores)
    return scores
