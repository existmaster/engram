"""SQLite database for metadata and full-text search."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path.home() / ".engram" / "engram.db"


def get_db_path() -> Path:
    """Get database path, creating directory if needed."""
    db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize database with schema."""
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        -- Sessions table
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            project_path TEXT,
            summary TEXT
        );

        -- Observations table
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            type TEXT NOT NULL,  -- decision, bugfix, feature, refactor, discovery
            content TEXT NOT NULL,
            compressed TEXT,     -- AI-compressed version
            file_refs TEXT,      -- JSON array of file paths
            created_at TEXT NOT NULL,
            token_count INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        -- Full-text search index
        CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
            content,
            compressed,
            content='observations',
            content_rowid='id'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
            INSERT INTO observations_fts(rowid, content, compressed)
            VALUES (new.id, new.content, new.compressed);
        END;

        CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
            INSERT INTO observations_fts(observations_fts, rowid, content, compressed)
            VALUES ('delete', old.id, old.content, old.compressed);
        END;

        CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
            INSERT INTO observations_fts(observations_fts, rowid, content, compressed)
            VALUES ('delete', old.id, old.content, old.compressed);
            INSERT INTO observations_fts(rowid, content, compressed)
            VALUES (new.id, new.content, new.compressed);
        END;
    """)

    conn.commit()
    return conn


def save_observation(
    conn: sqlite3.Connection,
    session_id: str,
    obs_type: str,
    content: str,
    file_refs: list[str] | None = None,
    compressed: str | None = None,
) -> int:
    """Save an observation to the database."""
    import json

    cursor = conn.execute(
        """
        INSERT INTO observations (session_id, type, content, compressed, file_refs, created_at, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            obs_type,
            content,
            compressed,
            json.dumps(file_refs) if file_refs else None,
            datetime.now().isoformat(),
            len(content.split()),  # rough token estimate
        ),
    )
    conn.commit()
    return cursor.lastrowid


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Full-text search on observations."""
    cursor = conn.execute(
        """
        SELECT o.*, rank
        FROM observations_fts
        JOIN observations o ON observations_fts.rowid = o.id
        WHERE observations_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    )
    return [dict(row) for row in cursor.fetchall()]
