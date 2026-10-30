"""Phase 4 — unit tests that do not require live embeddings."""

from __future__ import annotations

from pathlib import Path

from src.retrieval.embeddings import cosine_similarity
from src.retrieval.vector_store import (
    fingerprint_path,
    index_exists,
    read_fingerprint,
    write_fingerprint,
)
from src.config import Settings


def test_cosine_similarity_identical() -> None:
    v = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal() -> None:
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_fingerprint_roundtrip(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test-fingerprint",
        vector_store_path=tmp_path,
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        collection_name="test_kb",
        chunk_size=800,
        chunk_overlap=120,
    )
    write_fingerprint(tmp_path, settings, chunk_count=42)
    assert fingerprint_path(tmp_path).exists()
    fp = read_fingerprint(tmp_path)
    assert fp is not None
    assert fp["chunk_count"] == 42
    assert fp["embedding_model"] == "text-embedding-3-small"
    assert index_exists(tmp_path, settings) is True
