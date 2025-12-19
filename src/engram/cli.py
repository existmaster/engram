"""CLI interface for Engram."""

import click
from rich.console import Console
from rich.table import Table

from .core.db import init_db, save_observation
from .core.project import get_project_path, get_project_name
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

    # Auto-detect project
    project_path = get_project_path()
    project_name = get_project_name()

    # Check if embedder is ready
    if not vector.is_ready():
        console.print("[yellow]Warning: Ollama not available. Run 'ollama pull bge-m3' first.[/yellow]")
        console.print("[dim]Saving to SQLite only (no semantic search)...[/dim]")
        obs_id = save_observation(db, session, obs_type, content, project_path=project_path)
    else:
        obs_id = save_observation(db, session, obs_type, content, project_path=project_path)
        vector.add(obs_id, content, {
            "type": obs_type,
            "session_id": session,
            "project_path": project_path,
            "project_name": project_name,
        })

    console.print(f"[green]Saved observation #{obs_id}[/green] [dim]({project_name})[/dim]")


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Maximum results")
@click.option("--mode", "-m", default="hybrid",
              type=click.Choice(["semantic", "keyword", "hybrid"]),
              help="Search mode")
@click.option("--project", "-p", is_flag=True, help="Filter by current project only")
def search(query: str, limit: int, mode: str, project: bool):
    """Search observations semantically."""
    searcher = SemanticSearch()

    # Get project filter if requested
    project_filter = get_project_path() if project else None

    if not searcher.is_ready():
        console.print("[yellow]Warning: Ollama not available. Using keyword search only.[/yellow]")
        mode = "keyword"

    results = searcher.search(query, limit=limit, mode=mode, project_path=project_filter)

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    title = f"Search Results ({mode} mode)"
    if project_filter:
        title += f" [dim]({get_project_name()} only)[/dim]"

    table = Table(title=title)
    table.add_column("#", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Content", max_width=50)
    table.add_column("Project", style="magenta", max_width=15)
    table.add_column("Score", style="green")

    for i, r in enumerate(results, 1):
        content = r.get("content", r.get("compressed", ""))[:80]
        if len(content) == 80:
            content += "..."
        proj = r.get("metadata", {}).get("project_name", r.get("project_path", "?"))
        if isinstance(proj, str) and "/" in proj:
            proj = proj.split("/")[-1]  # Just show folder name
        table.add_row(
            str(i),
            r.get("type", r.get("metadata", {}).get("type", "?")),
            content,
            proj[:15] if proj else "?",
            f"{r.get('score', 0):.3f}",
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
