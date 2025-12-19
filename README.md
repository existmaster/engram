# Engram

Semantic memory system for Claude Code sessions.

Captures observations during coding sessions, stores them with vector embeddings, and enables semantic search to retrieve relevant context.

## Features

- **Automatic Capture**: Hooks into Claude Code to capture tool usage and decisions
- **Semantic Search**: Find relevant past observations using natural language
- **Hybrid Search**: Combines vector similarity with full-text search
- **Local-first**: SQLite + ChromaDB + Ollama, no external APIs required

## Installation

```bash
# Install with uv
uv pip install -e .

# Or install globally
uv tool install .
```

## Prerequisites

- [Ollama](https://ollama.ai/) running locally
- bge-m3 embedding model: `ollama pull bge-m3`

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
