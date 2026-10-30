"""Chunking strategies and experiment helpers (Task 03)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["recursive", "markdown_headers", "csv_passthrough"]


@dataclass
class ChunkingConfig:
    strategy: ChunkStrategy = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass
class ChunkStats:
    config: ChunkingConfig
    chunk_count: int
    mean_length: float
    max_length: int


def _recursive_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_documents(
    documents: list[Document],
    config: ChunkingConfig | None = None,
) -> list[Document]:
    """Split documents; metadata from parents is preserved/copied onto chunks.

    - ``recursive``: RecursiveCharacterTextSplitter (baseline for all docs)
    - ``markdown_headers``: structure-aware for ``.md`` sources; others recursive
    - ``csv_passthrough``: leave ``doc_type=pricing`` rows intact; recursive elsewhere
    """
    cfg = config or ChunkingConfig()
    if cfg.strategy == "recursive":
        splitter = _recursive_splitter(cfg.chunk_size, cfg.chunk_overlap)
        chunks = splitter.split_documents(documents)
        _ensure_metadata(chunks)
        return chunks

    if cfg.strategy == "csv_passthrough":
        pricing = [d for d in documents if d.metadata.get("doc_type") == "pricing"]
        others = [d for d in documents if d.metadata.get("doc_type") != "pricing"]
        splitter = _recursive_splitter(cfg.chunk_size, cfg.chunk_overlap)
        chunks = list(pricing) + splitter.split_documents(others)
        _ensure_metadata(chunks)
        return chunks

    if cfg.strategy == "markdown_headers":
        return _split_markdown_aware(documents, cfg)

    raise ValueError(f"Unknown chunk strategy: {cfg.strategy}")


def _split_markdown_aware(documents: list[Document], cfg: ChunkingConfig) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "heading1"), ("##", "heading2"), ("###", "heading3")],
        strip_headers=False,
    )
    recursive = _recursive_splitter(cfg.chunk_size, cfg.chunk_overlap)
    out: list[Document] = []
    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        if source.endswith(".md"):
            try:
                md_docs = header_splitter.split_text(doc.page_content)
            except Exception as exc:
                logger.warning("Markdown header split failed for %s: %s", source, exc)
                out.extend(recursive.split_documents([doc]))
                continue
            # Re-attach parent metadata; header splitter returns limited meta
            enriched: list[Document] = []
            for md in md_docs:
                meta = dict(doc.metadata)
                meta.update({k: v for k, v in md.metadata.items()})
                heading = md.metadata.get("heading2") or md.metadata.get("heading1")
                if heading:
                    meta["location"] = f"section:{heading}"
                enriched.append(Document(page_content=md.page_content, metadata=meta))
            # Secondary recursive split if a section is still huge
            out.extend(recursive.split_documents(enriched))
        else:
            out.extend(recursive.split_documents([doc]))
    _ensure_metadata(out)
    return out


def _ensure_metadata(chunks: list[Document]) -> None:
    for i, chunk in enumerate(chunks):
        if "source" not in chunk.metadata or not chunk.metadata["source"]:
            raise RuntimeError(f"Chunk {i} lost source metadata")
        if "doc_type" not in chunk.metadata or not chunk.metadata["doc_type"]:
            raise RuntimeError(f"Chunk {i} lost doc_type metadata")
        chunk.metadata.setdefault("chunk_index", i)


def measure_chunk_stats(chunks: list[Document], config: ChunkingConfig) -> ChunkStats:
    lengths = [len(c.page_content) for c in chunks] or [0]
    return ChunkStats(
        config=config,
        chunk_count=len(chunks),
        mean_length=sum(lengths) / max(len(lengths), 1),
        max_length=max(lengths),
    )


def run_chunking_experiment(
    documents: list[Document],
    configs: list[ChunkingConfig] | None = None,
) -> list[ChunkStats]:
    """Compare multiple chunking configs (report table input)."""
    configs = configs or [
        ChunkingConfig("recursive", 300, 0),
        ChunkingConfig("recursive", 300, 60),
        ChunkingConfig("recursive", 800, 0),
        ChunkingConfig("recursive", 800, 120),
        ChunkingConfig("recursive", 1500, 0),
        ChunkingConfig("recursive", 1500, 225),
        ChunkingConfig("markdown_headers", 800, 120),
        ChunkingConfig("csv_passthrough", 800, 120),
    ]
    results: list[ChunkStats] = []
    for cfg in configs:
        chunks = split_documents(documents, cfg)
        stats = measure_chunk_stats(chunks, cfg)
        results.append(stats)
        logger.info(
            "chunk experiment strategy=%s size=%s overlap=%s count=%s mean=%.1f max=%s",
            cfg.strategy,
            cfg.chunk_size,
            cfg.chunk_overlap,
            stats.chunk_count,
            stats.mean_length,
            stats.max_length,
        )
    return results
