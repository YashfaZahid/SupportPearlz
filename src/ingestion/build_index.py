"""CLI: build / rebuild the persistent vector index (offline ingestion path)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from src.config import get_settings
from src.ingestion.chunking import ChunkingConfig, split_documents
from src.ingestion.loaders import load_knowledge_base
from src.retrieval.embeddings import get_embeddings, run_similarity_diagnostic
from src.retrieval.vector_store import (
    create_vector_store,
    format_search_hits,
    index_exists,
    load_vector_store,
    similarity_search_debug,
)
from src.utils.cache_setup import setup_langchain_cache
from src.utils.errors import ConfigurationError, IndexError_, SupportPearlzError
from src.utils.logging_setup import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SupportPearlz index tools")
logger = logging.getLogger(__name__)


def _bootstrap() -> None:
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    setup_langchain_cache(settings)


@app.command("build")
def build(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Delete existing index and re-embed everything",
    ),
    knowledge_base: Optional[Path] = typer.Option(
        None,
        "--kb",
        help="Override knowledge base directory",
    ),
) -> None:
    """Ingest the knowledge base and persist the Chroma vector store."""
    try:
        _bootstrap()
        settings = get_settings()
        kb_path = knowledge_base or settings.knowledge_base_path
        if not kb_path.is_absolute():
            from src.config import PROJECT_ROOT

            kb_path = PROJECT_ROOT / kb_path

        if index_exists(settings=settings) and not rebuild:
            typer.echo(
                "Index already exists on disk. Loaded path will be used at query time "
                "without re-embedding. Pass --rebuild to wipe and re-embed."
            )
            from src.retrieval.vector_store import _resolve_store_path, read_fingerprint

            path = _resolve_store_path(settings)
            typer.echo(f"Store: {path}")
            fp = read_fingerprint(path) or {}
            for k, v in fp.items():
                typer.echo(f"  {k}: {v}")
            return

        docs, report = load_knowledge_base(kb_path)
        for line in report.summary_lines():
            typer.echo(line)
            logger.info(line)

        if not docs:
            raise IndexError_("No documents loaded; aborting index build.")

        chunk_cfg = ChunkingConfig(
            strategy="csv_passthrough",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = split_documents(docs, chunk_cfg)
        typer.echo(f"Chunks after split: {len(chunks)} (strategy={chunk_cfg.strategy})")

        embeddings = get_embeddings(settings)
        create_vector_store(
            chunks,
            settings=settings,
            embeddings=embeddings,
            rebuild=True,
        )
        typer.echo("Index build complete.")
        typer.echo(f"Store: {settings.vector_store_path}")
    except (ConfigurationError, IndexError_, SupportPearlzError) as exc:
        typer.secho(exc.message, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Natural-language search query"),
    k: int = typer.Option(4, "--k", help="Number of hits"),
) -> None:
    """Similarity-search helper for debugging retrieval without the LLM."""
    try:
        _bootstrap()
        settings = get_settings()
        store = load_vector_store(settings=settings)
        hits = similarity_search_debug(query, k=k, settings=settings, vector_store=store)
        typer.echo(format_search_hits(hits))
    except (ConfigurationError, IndexError_, SupportPearlzError) as exc:
        typer.secho(exc.message, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("diagnostic")
def diagnostic() -> None:
    """Embed paraphrase vs unrelated sentences and print cosine similarities."""
    try:
        _bootstrap()
        scores = run_similarity_diagnostic()
        typer.echo("Embedding similarity diagnostic")
        for key, value in scores.items():
            typer.echo(f"  {key}: {value}")
        if scores["paraphrase_pair"] <= max(scores["unrelated_1_3"], scores["unrelated_2_3"]):
            typer.secho(
                "Warning: paraphrase pair did not clearly outscore unrelated pairs.",
                fg=typer.colors.YELLOW,
            )
    except (ConfigurationError, SupportPearlzError) as exc:
        typer.secho(exc.message, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("status")
def status() -> None:
    """Show whether a persisted index exists and its fingerprint."""
    try:
        _bootstrap()
        settings = get_settings()
        from src.retrieval.vector_store import _resolve_store_path, read_fingerprint

        path = _resolve_store_path(settings)
        exists = index_exists(path, settings)
        typer.echo(f"path: {path}")
        typer.echo(f"exists: {exists}")
        fp = read_fingerprint(path)
        if fp:
            for k, v in fp.items():
                typer.echo(f"  {k}: {v}")
    except ConfigurationError as exc:
        typer.secho(exc.message, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def main() -> None:
    # `python -m src.ingestion.build_index` and `... build_index --rebuild`
    # map bare --rebuild onto the build command for assignment UX.
    import sys

    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        # Default subcommand: build
        sys.argv = [sys.argv[0], "build", *argv]
    app()


if __name__ == "__main__":
    main()
