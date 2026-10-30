"""Phase 5 — retriever + relevance gate (uses persisted index when present)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import clear_settings_cache, get_settings
from src.retrieval.retriever import hit_at_k, relevance_gate, retrieve, retrieve_with_scores
from src.retrieval.vector_store import index_exists, load_vector_store
from src.retrieval.retriever import ScoredChunk
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]


def _index_ready() -> bool:
    clear_settings_cache()
    if not os.getenv("OPENAI_API_KEY") and not (ROOT / ".env").exists():
        return False
    try:
        settings = get_settings()
    except Exception:
        return False
    return index_exists(settings=settings)


pytestmark = pytest.mark.skipif(not _index_ready(), reason="Persisted index + API key required")


@pytest.fixture(scope="module")
def store():
    clear_settings_cache()
    return load_vector_store()


def test_warranty_query_hits_warranty_source(store) -> None:
    chunks = retrieve_with_scores(store, "how long is the warranty on AquaPearl 500 Pro", k=4)
    assert chunks
    sources = [c.document.metadata.get("source") for c in chunks]
    assert "warranty_policy.md" in sources or "faq.md" in sources


def test_relevance_gate_empty() -> None:
    ok, reason = relevance_gate([])
    assert ok is False
    assert reason == "empty_result_set"


def test_relevance_gate_rejects_high_distance() -> None:
    fake = [
        ScoredChunk(
            document=Document(page_content="x", metadata={"source": "a.md", "doc_type": "faq"}),
            score=0.95,
            rank=1,
        )
    ]
    ok, reason = relevance_gate(fake, max_distance=0.75)
    assert ok is False
    assert "exceeds_threshold" in reason


def test_relevance_gate_accepts_good_distance() -> None:
    fake = [
        ScoredChunk(
            document=Document(page_content="x", metadata={"source": "a.md", "doc_type": "faq"}),
            score=0.40,
            rank=1,
        )
    ]
    ok, _ = relevance_gate(fake, max_distance=0.75)
    assert ok is True


def test_retrieve_unanswerable_often_fails_gate_or_weak(store) -> None:
    result = retrieve(store, "Do you offer student discounts for universities?")
    # Soft assertion: either gate fails, or best score is weaker than a factual query
    factual = retrieve(store, "How long is the warranty on the AquaPearl 500 Pro?")
    assert factual.best_score is not None
    if result.best_score is not None and factual.best_score is not None:
        # Unanswerable should not be *much* closer than a clear factual hit
        assert result.relevant is False or result.best_score >= factual.best_score - 0.05


def test_hit_at_k_sample(store) -> None:
    questions = [
        {
            "question": "How long is the warranty on the AquaPearl 500 Pro?",
            "expected_sources": ["warranty_policy.md", "faq.md"],
            "must_contain": "24",
        },
        {
            "question": "How long does delivery take to Skardu?",
            "expected_sources": ["shipping_policy.md", "faq.md"],
            "must_contain": "5-7",
        },
        {
            "question": "What does three beeps mean?",
            "expected_sources": ["troubleshooting_guide.md", "product_manual_aquapearl500.pdf", "faq.md"],
        },
    ]
    score = hit_at_k(store, questions, k=4, strategy="similarity")
    assert score >= 2 / 3


def test_metadata_filter_policy(store) -> None:
    chunks = retrieve_with_scores(
        store,
        "warranty period",
        k=3,
        strategy="similarity",
        filter_dict={"doc_type": "policy"},
    )
    assert chunks
    assert all(c.document.metadata.get("doc_type") == "policy" for c in chunks)
