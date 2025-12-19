# Engram

Semantic memory system for Claude Code sessions.

Captures observations during coding sessions, stores them with vector embeddings, and enables semantic search to retrieve relevant context.

## Features

- **Automatic Capture**: Hooks into Claude Code to capture tool usage and decisions
- **Semantic Search**: Find relevant past observations using natural language
- **Hybrid Search**: Combines vector similarity with full-text search
- **Local-first**: SQLite + ChromaDB + Ollama, no external APIs required

## Prerequisites

1. **Python 3.11+** and [uv](https://docs.astral.sh/uv/)
2. **Ollama** running locally: https://ollama.ai/
3. **bge-m3** embedding model:
   ```bash
   ollama pull bge-m3
   ```

## Installation

```bash
# Clone the repository
git clone https://github.com/existmaster/engram.git
cd engram

# Install globally (recommended)
uv tool install -e .

# Verify installation
engram status
```

### Alternative: Development Mode

```bash
# Install in current project's venv
uv pip install -e .

# Run with uv
uv run engram status
```

## Usage

```bash
# Initialize (creates DB, pulls embedding model)
engram init

# Check status
engram status

# Save an observation
engram save "Fixed authentication bug by adding token refresh"
engram save "Decided to use Chroma for vector storage" --type decision

# Search
engram search "authentication"
engram search "vector database" --mode semantic
engram search "token" --mode keyword
```

## Architecture

```
~/.engram/
├── engram.db      # SQLite (metadata + FTS5)
└── chroma/        # ChromaDB (vectors)

Hooks (Claude Code):
├── PostToolUse    # Capture observations
├── Stop           # Compress & save session
└── SessionStart   # Inject relevant context
```

## Tech Stack

- **Python 3.11+** with uv
- **SQLite** with FTS5 for full-text search
- **ChromaDB** for vector storage
- **Ollama bge-m3** for embeddings
- **Click + Rich** for CLI

## License

MIT
