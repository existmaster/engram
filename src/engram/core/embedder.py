"""Embedding generation using Ollama."""

import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "bge-m3"


class OllamaEmbedder:
    """Generate embeddings using Ollama's local models."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self._client = httpx.Client(timeout=60.0)

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def pull_model(self) -> bool:
        """Pull the embedding model if not available."""
        try:
            response = self._client.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=300.0,  # 5 min timeout for download
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
