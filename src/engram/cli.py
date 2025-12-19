"""CLI interface for Engram."""

import click
from rich.console import Console
from rich.table import Table

from .core.db import init_db, save_observation
from .core.vector import VectorStore
from .search.semantic import SemanticSearch

console = Console()


@click.group()
@click.version_option()
def main():
    """Engram - Semantic memory for Claude Code sessions."""
    pass


@main.command()
@click.argument("content")
@click.option("--type", "-t", "obs_type", default="discovery",
              type=click.Choice(["decision", "bugfix", "feature", "refactor", "discovery"]),
              help="Type of observation")
@click.option("--session", "-s", default="manual", help="Session ID")
def save(content: str, obs_type: str, session: str):
    """Save an observation to memory."""
    db = init_db()
    vector = VectorStore()

    # Check if embedder is ready
    if not vector.is_ready():
        console.print("[yellow]Warning: Ollama not available. Run 'ollama pull bge-m3' first.[/yellow]")
        console.print("[dim]Saving to SQLite only (no semantic search)...[/dim]")
        obs_id = save_observation(db, session, obs_type, content)
    else:
        obs_id = save_observation(db, session, obs_type, content)
        vector.add(obs_id, content, {"type": obs_type, "session_id": session})

    console.print(f"[green]Saved observation #{obs_id}[/green]")


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Maximum results")
@click.option("--mode", "-m", default="hybrid",
              type=click.Choice(["semantic", "keyword", "hybrid"]),
              help="Search mode")
def search(query: str, limit: int, mode: str):
    """Search observations semantically."""
    searcher = SemanticSearch()

    if not searcher.is_ready():
        console.print("[yellow]Warning: Ollama not available. Using keyword search only.[/yellow]")
        mode = "keyword"

    results = searcher.search(query, limit=limit, mode=mode)

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    table = Table(title=f"Search Results ({mode} mode)")
    table.add_column("#", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Content", max_width=60)
    table.add_column("Score", style="green")
    table.add_column("Source", style="dim")

    for i, r in enumerate(results, 1):
        content = r.get("content", r.get("compressed", ""))[:100]
        if len(content) == 100:
            content += "..."
        table.add_row(
            str(i),
            r.get("type", r.get("metadata", {}).get("type", "?")),
            content,
            f"{r.get('score', 0):.3f}",
            r.get("source", "?"),
        )

    console.print(table)


@main.command()
def status():
    """Check Engram system status."""
    from .core.embedder import OllamaEmbedder

    db = init_db()
    vector = VectorStore()
    embedder = OllamaEmbedder()

    table = Table(title="Engram Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    # SQLite
    cursor = db.execute("SELECT COUNT(*) FROM observations")
    obs_count = cursor.fetchone()[0]
    table.add_row("SQLite", "[green]OK[/green]", f"{obs_count} observations")

    # ChromaDB
    chroma_count = vector.count()
    table.add_row("ChromaDB", "[green]OK[/green]", f"{chroma_count} vectors")

    # Ollama
    if embedder.is_available():
        table.add_row("Ollama (bge-m3)", "[green]OK[/green]", "Model ready")
    else:
        table.add_row("Ollama (bge-m3)", "[red]NOT READY[/red]", "Run: ollama pull bge-m3")

    console.print(table)


@main.command()
def init():
    """Initialize Engram (pull embedding model)."""
    from .core.embedder import OllamaEmbedder

    console.print("[bold]Initializing Engram...[/bold]")

    # Initialize DB
    init_db()
    console.print("[green]SQLite database initialized[/green]")

    # Initialize vector store
    VectorStore()
    console.print("[green]ChromaDB initialized[/green]")

    # Check/pull embedding model
    embedder = OllamaEmbedder()
    if embedder.is_available():
        console.print("[green]Embedding model (bge-m3) ready[/green]")
    else:
        console.print("[yellow]Pulling embedding model (bge-m3)...[/yellow]")
        if embedder.pull_model():
            console.print("[green]Model pulled successfully[/green]")
        else:
            console.print("[red]Failed to pull model. Run manually: ollama pull bge-m3[/red]")

    console.print("\n[bold green]Engram is ready![/bold green]")


if __name__ == "__main__":
    main()
