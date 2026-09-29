import json
import logging
import threading
from copy import deepcopy
from typing import Any, Callable, Dict

from mem0 import Memory
from mem0.utils.embedding_dimensions import infer_embedding_dims

_state_lock = threading.RLock()
_current_config: Dict[str, Any] = {}
_memory_instance: Memory | None = None
_session_factory: Callable | None = None
_VECTOR_STORES_WITH_EMBEDDING_DIMS = {
    "elasticsearch",
    "faiss",
    "milvus",
    "pgvector",
    "qdrant",
    "redis",
    "weaviate",
}


def set_session_factory(factory: Callable) -> None:
    global _session_factory
    _session_factory = factory


def _load_overrides() -> Dict[str, Any]:
    try:
        if _session_factory is None:
            return {}
        from models import Settings

        session = _session_factory()
        try:
            row = session.get(Settings, "config_overrides")
            if row is None:
                return {}
            return json.loads(row.value)
        finally:
            session.close()
    except Exception:
        return {}


def _save_overrides(overrides: Dict[str, Any]) -> None:
    try:
        if _session_factory is None:
            return
        from models import Settings
        from sqlalchemy.dialects.postgresql import insert

        session = _session_factory()
        try:
            serialized = json.dumps(overrides)
            stmt = (
                insert(Settings)
                .values(key="config_overrides", value=serialized)
                .on_conflict_do_update(
                    index_elements=[Settings.key],
                    set_={"value": serialized},
                )
            )
            session.execute(stmt)
            session.commit()
        finally:
            session.close()
    except Exception:
        logging.warning("Failed to persist config overrides to database", exc_info=True)


def _merge_config(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def _sync_vector_store_embedding_dims(config: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    vector_store = config.get("vector_store")
    if not isinstance(vector_store, dict):
        return config

    provider = vector_store.get("provider")
    if provider not in _VECTOR_STORES_WITH_EMBEDDING_DIMS:
        return config

    vector_config = vector_store.get("config")
    if not isinstance(vector_config, dict):
        return config

    if not force and vector_config.get("embedding_model_dims") is not None:
        return config

    embedder = config.get("embedder", {})
    embedder_provider = embedder.get("provider") if isinstance(embedder, dict) else None
    embedder_config = embedder.get("config", {}) if isinstance(embedder, dict) else {}
    embedder_model = embedder_config.get("model") if isinstance(embedder_config, dict) else None

    vector_config["embedding_model_dims"] = infer_embedding_dims(embedder_provider, embedder_model)
    return config


def initialize_state(default_config: Dict[str, Any]) -> None:
    global _current_config, _memory_instance
    with _state_lock:
        _current_config = deepcopy(default_config)
        overrides = _load_overrides()
        if overrides:
            _current_config = _merge_config(_current_config, overrides)
        _current_config = _sync_vector_store_embedding_dims(_current_config)
        _memory_instance = Memory.from_config(_current_config)


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    global _current_config, _memory_instance
    with _state_lock:
        next_config = _merge_config(_current_config, updates)
        vector_store_updates = (
            updates.get("vector_store", {}) if isinstance(updates.get("vector_store"), dict) else {}
        )
        vector_store_config_updates = (
            vector_store_updates.get("config", {}) if isinstance(vector_store_updates.get("config"), dict) else {}
        )
        should_force_dims = (
            "embedder" in updates or "vector_store" in updates
        ) and "embedding_model_dims" not in vector_store_config_updates
        next_config = _sync_vector_store_embedding_dims(next_config, force=should_force_dims)
        _current_config = next_config
        _memory_instance = Memory.from_config(next_config)
        overrides = _load_overrides()
        overrides = _merge_config(overrides, updates)
        _save_overrides(overrides)
        return deepcopy(_current_config)


def get_current_config() -> Dict[str, Any]:
    with _state_lock:
        return deepcopy(_current_config)


def get_memory_instance() -> Memory:
    with _state_lock:
        if _memory_instance is None:
            raise RuntimeError("Mem0 runtime has not been initialized.")
        return _memory_instance
