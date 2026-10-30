"""Chroma vector store: create, persist, load, rebuild, delete-by-source upsert."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.config import PROJECT_ROOT, Settings, get_settings
from src.retrieval.embeddings import get_embeddings
from src.utils.errors import IndexError_

logger = logging.getLogger(__name__)

FINGERPRINT_FILENAME = "index_fingerprint.json"


def _resolve_store_path(settings: Settings) -> Path:
    path = settings.vector_store_path
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def fingerprint_path(store_path: Path) -> Path:
    return store_path / FINGERPRINT_FILENAME


def write_fingerprint(store_path: Path, settings: Settings, *, chunk_count: int) -> None:
    store_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "collection_name": settings.collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunk_count": chunk_count,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    fingerprint_path(store_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_fingerprint(store_path: Path) -> dict[str, Any] | None:
    path = fingerprint_path(store_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def index_exists(store_path: Path | None = None, settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    path = store_path or _resolve_store_path(cfg)
    fp = read_fingerprint(path)
    # Chroma persist dir typically contains chroma.sqlite3
    sqlite = path / "chroma.sqlite3"
    return bool(fp and (sqlite.exists() or any(path.glob("*"))))


def assert_embedding_model_matches(settings: Settings, store_path: Path | None = None) -> None:
    path = store_path or _resolve_store_path(settings)
    fp = read_fingerprint(path)
    if not fp:
        return
    recorded = fp.get("embedding_model")
    if recorded and recorded != settings.embedding_model:
        raise IndexError_(
            f"Embedding model mismatch: index was built with '{recorded}' but "
            f"config has '{settings.embedding_model}'. Rebuild the index "
            f"(python -m src.ingestion.build_index --rebuild)."
        )


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str, int, float, bool metadata values."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def create_vector_store(
    documents: list[Document],
    *,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    rebuild: bool = False,
) -> Chroma:
    """Embed documents and persist to disk. Use rebuild=True to wipe first."""
    cfg = settings or get_settings()
    store_path = _resolve_store_path(cfg)
    emb = embeddings or get_embeddings(cfg)

    if rebuild and store_path.exists():
        logger.info("Removing existing vector store at %s", store_path)
        shutil.rmtree(store_path)

    store_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    logger.info(
        "Building vector store: %s chunks → %s (model=%s)",
        len(documents),
        store_path,
        cfg.embedding_model,
    )

    safe_docs = [
        Document(page_content=d.page_content, metadata=_sanitize_metadata(dict(d.metadata)))
        for d in documents
    ]

    # Batch via Chroma.from_documents (LangChain handles batching internally)
    vector_store = Chroma.from_documents(
        documents=safe_docs,
        embedding=emb,
        collection_name=cfg.collection_name,
        persist_directory=str(store_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
    elapsed = time.perf_counter() - started
    write_fingerprint(store_path, cfg, chunk_count=len(documents))
    logger.info(
        "Vector store persisted (%s chunks, %.1fs). Embeddings generated this run.",
        len(documents),
        elapsed,
    )
    return vector_store


def load_vector_store(
    *,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Load a persisted index from disk (no re-embedding)."""
    cfg = settings or get_settings()
    store_path = _resolve_store_path(cfg)
    if not index_exists(store_path, cfg):
        raise IndexError_(
            f"No vector index found at {store_path}. "
            "Run: python -m src.ingestion.build_index"
        )
    assert_embedding_model_matches(cfg, store_path)
    emb = embeddings or get_embeddings(cfg)
    logger.info("Loading vector store from disk: %s (no re-embed)", store_path)
    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=emb,
        persist_directory=str(store_path),
    )


def load_or_build(
    documents: list[Document] | None = None,
    *,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    rebuild: bool = False,
) -> Chroma:
    """Load persisted index if valid; otherwise build from ``documents``."""
    cfg = settings or get_settings()
    store_path = _resolve_store_path(cfg)
    if not rebuild and index_exists(store_path, cfg):
        return load_vector_store(settings=cfg, embeddings=embeddings)
    if documents is None:
        raise IndexError_(
            "Index missing and no documents provided to build. "
            "Run build_index or pass documents."
        )
    return create_vector_store(
        documents,
        settings=cfg,
        embeddings=embeddings,
        rebuild=rebuild,
    )


def delete_by_source(vector_store: Chroma, source: str) -> int:
    """Remove all chunks whose metadata.source matches ``source`` (for upsert)."""
    try:
        existing = vector_store.get(where={"source": source})
    except Exception as exc:
        logger.warning("delete_by_source get failed: %s", exc)
        return 0
    ids = existing.get("ids") or []
    if not ids:
        return 0
    vector_store.delete(ids=ids)
    logger.info("Deleted %s chunks for source=%s", len(ids), source)
    return len(ids)


def upsert_documents_by_source(
    vector_store: Chroma,
    documents: list[Document],
    *,
    settings: Settings | None = None,
) -> None:
    """Delete-by-source then reinsert — chosen incremental update strategy."""
    cfg = settings or get_settings()
    sources = sorted({str(d.metadata.get("source", "")) for d in documents if d.metadata.get("source")})
    for source in sources:
        delete_by_source(vector_store, source)
    if documents:
        vector_store.add_documents(documents)
    store_path = _resolve_store_path(cfg)
    # Refresh fingerprint chunk count approximately via collection count
    try:
        count = vector_store._collection.count()  # noqa: SLF001 — chromadb count
    except Exception:
        count = len(documents)
    write_fingerprint(store_path, cfg, chunk_count=count)
    logger.info("Upserted %s docs across sources %s", len(documents), sources)


def similarity_search_debug(
    query: str,
    *,
    k: int | None = None,
    settings: Settings | None = None,
    vector_store: Chroma | None = None,
) -> list[tuple[Document, float]]:
    """Search and return (doc, score) for debugging — prints-friendly helper."""
    cfg = settings or get_settings()
    store = vector_store or load_vector_store(settings=cfg)
    k = k or cfg.retrieval_k
    # Chroma with cosine space: score is distance; convert carefully for display
    pairs = store.similarity_search_with_score(query, k=k)
    results: list[tuple[Document, float]] = []
    for doc, score in pairs:
        results.append((doc, float(score)))
    return results


def format_search_hits(hits: list[tuple[Document, float]], preview: int = 200) -> str:
    lines: list[str] = []
    for i, (doc, score) in enumerate(hits, start=1):
        meta = doc.metadata
        snippet = doc.page_content[:preview].replace("\n", " ")
        lines.append(
            f"[{i}] score={score:.4f} source={meta.get('source')} "
            f"loc={meta.get('location')} doc_type={meta.get('doc_type')}\n"
            f"    {snippet}"
        )
    return "\n".join(lines) if lines else "(no hits)"
