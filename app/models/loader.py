import os
import pickle
from typing import Any, Dict, Optional

import joblib

from app.logging.structured_logger import logger
from app.models.registry import active_marker_mtime, read_active_model_path

# In-memory cache: cache_key -> estimator
active_models: Dict[str, Any] = {}
# Last seen active_model.txt mtime per model_id (for hot reload)
_marker_mtimes: Dict[str, float] = {}


def load_model_from_file(file_path: str) -> Any:
    if not os.path.exists(file_path):
        logger.error(f"Model file not found at path: {file_path}")
        raise FileNotFoundError(f"Model file not found at path: {file_path}")

    logger.info(f"Loading serialized model from {file_path}...")
    try:
        if file_path.endswith(".joblib"):
            model = joblib.load(file_path)
        else:
            with open(file_path, "rb") as f:
                model = pickle.load(f)
        logger.info(f"Model successfully loaded from {file_path}.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {file_path}: {e}")
        raise RuntimeError(f"Failed to deserialize model binary: {e}")


def _invalidate_if_marker_changed(model_id: str) -> None:
    """Drop cached estimators when active_model.txt changes."""
    current = active_marker_mtime(model_id)
    previous = _marker_mtimes.get(model_id)
    if previous is not None and current != previous:
        keys = [k for k in active_models if k.startswith(f"{model_id}:")]
        for key in keys:
            del active_models[key]
        logger.info(
            f"active_model.txt changed for {model_id}; cleared {len(keys)} cached estimator(s)."
        )
    _marker_mtimes[model_id] = current


def get_active_model(model_id: str, version: str, file_path: str) -> Any:
    """
    Load estimator from disk, reloading when active_model.txt changes.
    """
    _invalidate_if_marker_changed(model_id)

    # Prefer path from active_model.txt if it matches this deployment
    resolved = read_active_model_path(model_id) or file_path
    cache_key = f"{model_id}:{version}:{os.path.abspath(resolved)}"

    if cache_key in active_models:
        return active_models[cache_key]

    model = load_model_from_file(resolved)
    active_models[cache_key] = model
    return model


def get_active_model_by_marker(model_id: str) -> Any:
    """Load strictly from active_model.txt (no version hint)."""
    _invalidate_if_marker_changed(model_id)
    path = read_active_model_path(model_id)
    if not path:
        raise FileNotFoundError(f"No active model configured for '{model_id}'")
    cache_key = f"{model_id}:active:{os.path.abspath(path)}"
    if cache_key in active_models:
        return active_models[cache_key]
    model = load_model_from_file(path)
    active_models[cache_key] = model
    return model


def clear_model_cache(model_id: Optional[str] = None, version: Optional[str] = None):
    global active_models, _marker_mtimes
    if model_id and version:
        prefix = f"{model_id}:{version}:"
        keys = [k for k in active_models if k.startswith(prefix) or k.startswith(f"{model_id}:active:")]
        for key in keys:
            del active_models[key]
        logger.info(f"Cleared cache for {model_id} version {version}.")
    elif model_id:
        keys = [k for k in active_models if k.startswith(f"{model_id}:")]
        for key in keys:
            del active_models[key]
        _marker_mtimes.pop(model_id, None)
        logger.info(f"Cleared all cache entries for {model_id}.")
    else:
        active_models.clear()
        _marker_mtimes.clear()
        logger.info("Cleared all models from memory cache.")
