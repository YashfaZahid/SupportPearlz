"""Phase 3 — chunking preserves metadata; experiment produces multiple configs."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.chunking import ChunkingConfig, run_chunking_experiment, split_documents
from src.ingestion.loaders import load_knowledge_base

KB = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


def test_recursive_chunks_preserve_source_and_doc_type() -> None:
    docs, _ = load_knowledge_base(KB)
    chunks = split_documents(docs, ChunkingConfig("recursive", 800, 120))
    assert len(chunks) > len(docs)
    for ch in chunks:
        assert ch.metadata.get("source")
        assert ch.metadata.get("doc_type")


def test_csv_passthrough_keeps_pricing_rows_intact() -> None:
    docs, _ = load_knowledge_base(KB)
    pricing_docs = [d for d in docs if d.metadata.get("doc_type") == "pricing"]
    chunks = split_documents(docs, ChunkingConfig("csv_passthrough", 800, 120))
    pricing_chunks = [c for c in chunks if c.metadata.get("doc_type") == "pricing"]
    assert len(pricing_chunks) == len(pricing_docs)
    assert any("AP500-SED" in c.page_content for c in pricing_chunks)


def test_markdown_header_strategy_runs() -> None:
    docs, _ = load_knowledge_base(KB)
    chunks = split_documents(docs, ChunkingConfig("markdown_headers", 800, 120))
    assert chunks
    assert all(c.metadata.get("source") for c in chunks)


def test_experiment_has_at_least_six_configs() -> None:
    docs, _ = load_knowledge_base(KB)
    # Use a subset for speed: warranty + pricing + faq
    subset = [
        d
        for d in docs
        if d.metadata.get("source")
        in {"warranty_policy.md", "pricing_guide.csv", "faq.md", "shipping_policy.md"}
    ]
    assert subset
    stats = run_chunking_experiment(subset)
    assert len(stats) >= 6
    # Smaller chunk size should generally produce more chunks than larger (same strategy)
    recursive = [s for s in stats if s.config.strategy == "recursive"]
    by_size = {s.config.chunk_size: s.chunk_count for s in recursive if s.config.chunk_overlap == 0}
    if 300 in by_size and 1500 in by_size:
        assert by_size[300] >= by_size[1500]
