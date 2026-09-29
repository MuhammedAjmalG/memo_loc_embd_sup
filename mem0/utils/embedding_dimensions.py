from typing import Optional


DEFAULT_EMBEDDING_DIMS = 1536

_KNOWN_DIMS = {
    "openai": {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    },
    "ollama": {
        "nomic-embed-text": 768,
        "nomic-embed-text:latest": 768,
    },
}

_PROVIDER_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}


def infer_embedding_dims(
    provider: Optional[str], model: Optional[str], explicit: Optional[int] = None
) -> int:
    """Return an embedding dimension for common bundled models.

    Explicit dimensions always win. Unknown providers/models fall back to 1536,
    which matches Mem0's historical default.
    """
    if explicit is not None:
        return explicit

    normalized_provider = (provider or "").strip().lower()
    normalized_model = (model or "").strip().lower() or _PROVIDER_DEFAULT_MODELS.get(
        normalized_provider, ""
    )

    if normalized_provider == "ollama" and normalized_model.startswith("nomic-embed-text"):
        return 768

    return _KNOWN_DIMS.get(normalized_provider, {}).get(normalized_model, DEFAULT_EMBEDDING_DIMS)
