"""CLI interface for Engram."""

import json
import sys

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
@click.option("--hooks/--no-hooks", default=True, help="Install Claude Code hooks")
def init(hooks: bool):
    """Initialize Engram - check dependencies and install hooks."""
    from .core.embedder import OllamaEmbedder
    from .setup import check_ollama, check_model, check_global_install, install_hooks

    console.print("[bold]Engram Setup[/bold]\n")

    all_ok = True

    # Step 1: Check Ollama
    console.print("[bold]1. Checking Ollama...[/bold]")
    ok, msg = check_ollama()
    if ok:
        console.print(f"   [green]OK[/green] {msg}")
    else:
        console.print(f"   [red]FAIL[/red] {msg}")
        all_ok = False

    # Step 2: Check embedding model
    console.print("[bold]2. Checking embedding model...[/bold]")
    ok, msg = check_model("bge-m3")
    if ok:
        console.print(f"   [green]OK[/green] {msg}")
    else:
        console.print(f"   [yellow]MISSING[/yellow] {msg}")
        console.print("   [dim]Attempting to pull model...[/dim]")
        embedder = OllamaEmbedder()
        if embedder.pull_model():
            console.print("   [green]OK[/green] Model pulled successfully")
        else:
            console.print("   [red]FAIL[/red] Could not pull model")
            all_ok = False

    # Step 3: Check global install
    console.print("[bold]3. Checking global installation...[/bold]")
    ok, msg = check_global_install()
    if ok:
        console.print(f"   [green]OK[/green] {msg}")
    else:
        console.print(f"   [yellow]MISSING[/yellow] {msg}")
        console.print("   [dim]Run: uv tool install -e .[/dim]")

    # Step 4: Initialize DB
    console.print("[bold]4. Initializing database...[/bold]")
    init_db()
    VectorStore()
    console.print("   [green]OK[/green] SQLite + ChromaDB ready")

    # Step 5: Install hooks (optional)
    if hooks:
        console.print("[bold]5. Installing Claude Code hooks...[/bold]")
        ok, msg, changes = install_hooks()
        if ok:
            for hook_type, status in changes.items():
                if status == "added":
                    console.print(f"   [green]+[/green] {hook_type} hook added")
                else:
                    console.print(f"   [dim]=[/dim] {hook_type} already installed")
        else:
            console.print(f"   [red]FAIL[/red] {msg}")
    else:
        console.print("[bold]5. Skipping hooks[/bold] (--no-hooks)")

    # Summary
    console.print()
    if all_ok:
        console.print("[bold green]Engram is ready![/bold green]")
        console.print("[dim]Try: engram save \"test observation\"[/dim]")
    else:
        console.print("[bold yellow]Setup incomplete. Fix issues above.[/bold yellow]")


@main.command()
@click.option("--hook", type=click.Choice(["post-tool-use", "stop"]), required=True,
              help="Hook type that triggered this capture")
def capture(hook: str):
    """Capture observations from Claude Code hooks (internal use)."""
    # Read JSON from stdin (Claude Code sends session data)
    try:
        if not sys.stdin.isatty():
            input_data = sys.stdin.read()
            if input_data.strip():
                data = json.loads(input_data)
            else:
                data = {}
        else:
            data = {}
    except json.JSONDecodeError:
        data = {}

    # Get project info
    project_path = get_project_path()
    project_name = get_project_name()

    # Determine observation type and content based on hook
    if hook == "post-tool-use":
        # Extract tool usage info
        tool_name = data.get("tool_name", "unknown")
        tool_input = data.get("tool_input", {})

        # Skip trivial tools
        if tool_name in ("Read", "Glob", "Grep", "LS"):
            return  # Don't save read-only operations

        # Build observation content
        content = f"[{tool_name}] "
        if tool_name == "Bash":
            content += tool_input.get("command", "")[:200]
        elif tool_name in ("Edit", "Write"):
            file_path = tool_input.get("file_path", "")
            content += f"Modified {file_path}"
        else:
            content += json.dumps(tool_input)[:200]

        obs_type = "change"

    elif hook == "stop":
        # Session end - could summarize the session
        # For now, just note the session ended
        content = f"Session ended in {project_name}"
        obs_type = "discovery"

    else:
        return

    # Save to engram (silently)
    try:
        db = init_db()
        vector = VectorStore()

        obs_id = save_observation(
            db, "hook", obs_type, content, project_path=project_path
        )

        if vector.is_ready():
            vector.add(obs_id, content, {
                "type": obs_type,
                "hook": hook,
                "project_path": project_path,
                "project_name": project_name,
            })
    except Exception:
        pass  # Fail silently - don't interrupt Claude Code


if __name__ == "__main__":
    main()
