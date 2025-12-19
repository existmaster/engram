"""Semantic search combining FTS and vector search."""

from typing import Any

from ..core.db import init_db, search_fts
from ..core.vector import VectorStore


class SemanticSearch:
    """Hybrid search combining full-text and vector search."""

    def __init__(self):
        self._db = init_db()
        self._vector = VectorStore()

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """
        Search observations using specified mode.

        Args:
            query: Search query
            limit: Maximum results
            mode: "semantic" (vector only), "keyword" (FTS only), "hybrid" (both)
        """
        results = []

        if mode in ("semantic", "hybrid"):
            vector_results = self._vector.search(query, limit=limit)
            for r in vector_results:
                r["source"] = "semantic"
                r["score"] = 1 - r.get("distance", 0)  # Convert distance to similarity
            results.extend(vector_results)

        if mode in ("keyword", "hybrid"):
            fts_results = search_fts(self._db, query, limit=limit)
            for r in fts_results:
                r["source"] = "keyword"
                r["score"] = abs(r.get("rank", 0))  # FTS5 rank is negative
            results.extend(fts_results)

        if mode == "hybrid":
            # Deduplicate by id, keeping highest score
            seen = {}
            for r in results:
                obs_id = r.get("id")
                if obs_id not in seen or r.get("score", 0) > seen[obs_id].get("score", 0):
                    seen[obs_id] = r
            results = list(seen.values())

        # Sort by score descending
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]

    def is_ready(self) -> bool:
        """Check if search system is ready."""
        return self._vector.is_ready()
