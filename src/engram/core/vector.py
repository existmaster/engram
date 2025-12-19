"""Vector storage using ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .embedder import OllamaEmbedder

DEFAULT_CHROMA_PATH = Path.home() / ".engram" / "chroma"


class VectorStore:
    """ChromaDB-based vector storage for semantic search."""

    def __init__(
        self,
        path: Path | None = None,
        collection_name: str = "observations",
    ):
        self.path = path or DEFAULT_CHROMA_PATH
        self.path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = OllamaEmbedder()

    def add(
        self,
        observation_id: int,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an observation to the vector store."""
        embedding = self._embedder.embed(content)

        self._collection.add(
            ids=[str(observation_id)],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}],
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search for similar observations."""
        query_embedding = self._embedder.embed(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten results
        observations = []
        if results["ids"] and results["ids"][0]:
            for i, obs_id in enumerate(results["ids"][0]):
                observations.append({
                    "id": int(obs_id),
                    "content": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

        return observations

    def delete(self, observation_id: int) -> None:
        """Delete an observation from the vector store."""
        self._collection.delete(ids=[str(observation_id)])

    def count(self) -> int:
        """Get total number of observations."""
        return self._collection.count()

    def is_ready(self) -> bool:
        """Check if vector store and embedder are ready."""
        return self._embedder.is_available()
