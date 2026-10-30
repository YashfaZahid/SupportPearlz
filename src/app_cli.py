"""Interactive CLI chat for SupportPearlz."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.chains.rag_chain import SupportPearlzAgent
from src.config import get_settings
from src.retrieval.vector_store import index_exists, read_fingerprint, _resolve_store_path
from src.utils.cache_setup import setup_langchain_cache
from src.utils.errors import ConfigurationError, GenerationError, IndexError_, SupportPearlzError
from src.utils.logging_setup import setup_logging

app = typer.Typer(add_completion=False, help="SupportPearlz CLI")
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def chat(
    debug: bool = typer.Option(False, "--debug", help="Show rewritten query hints when available"),
) -> None:
    """Start an interactive multi-turn support session."""
    try:
        settings = get_settings()
        setup_logging(level=settings.log_level, log_file=settings.log_file)
        setup_langchain_cache(settings)

        if not index_exists(settings=settings):
            raise IndexError_(
                "Vector index missing. Run: python -m src.ingestion.build_index --rebuild"
            )

        fp = read_fingerprint(_resolve_store_path(settings)) or {}
        console.print(
            Panel.fit(
                f"[bold]SupportPearlz[/bold]\n"
                f"knowledge base: {settings.collection_name} | "
                f"{fp.get('chunk_count', '?')} chunks\n\n"
                f"[dim]{settings.medical_safety_disclaimer}[/dim]",
                border_style="teal",
            )
        )
        console.print(
            "Commands: /reset  /quit  /debug\n"
            f"Escalate: {settings.support_escalation_email}"
        )

        agent = SupportPearlzAgent(settings=settings)
        show_debug = debug

        while True:
            try:
                user = console.input("[bold cyan]You >[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nSession ended.")
                break

            if not user:
                continue
            if user.lower() in {"/quit", "/exit", "quit", "exit"}:
                break
            if user.lower() == "/reset":
                agent.reset()
                console.print("[yellow]Session reset.[/yellow]")
                continue
            if user.lower() == "/debug":
                show_debug = not show_debug
                console.print(f"debug={'on' if show_debug else 'off'}")
                continue

            try:
                response = agent.ask(user)
            except GenerationError as exc:
                console.print(f"[red]{exc.message}[/red]")
                continue
            except SupportPearlzError as exc:
                console.print(f"[red]{exc.message}[/red]")
                continue

            console.print()
            console.print(Markdown(response.answer))
            if response.sources:
                console.print("[bold]Sources:[/bold]")
                for i, src in enumerate(response.sources, start=1):
                    console.print(
                        f"  [{i}] {src.source} — {src.location or 'n/a'}"
                    )
            else:
                console.print("[dim]Sources: (none — refusal or uncited)[/dim]")
            console.print(f"Confidence: [bold]{response.confidence.value}[/bold]")
            if response.flags:
                console.print(f"[dim]flags: {', '.join(response.flags)}[/dim]")
            console.print()

    except (ConfigurationError, IndexError_) as exc:
        console.print(f"[red]{exc.message}[/red]")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
