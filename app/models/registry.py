"""
Filesystem-backed manual model registry.

Layout per model:
  models/{model_id}/model_v1.joblib
  models/{model_id}/model_v2.joblib
  models/{model_id}/active_model.txt   # filename of the deployed artifact
"""
import os
from typing import Optional

MODELS_ROOT = os.getenv("MODELS_ROOT", "models")
ACTIVE_MARKER = "active_model.txt"


def model_dir(model_id: str) -> str:
    return os.path.join(MODELS_ROOT, model_id)


def active_marker_path(model_id: str) -> str:
    return os.path.join(model_dir(model_id), ACTIVE_MARKER)


def version_to_filename(version: str, ext: str = ".joblib") -> str:
    """Map version tag v1 -> model_v1.joblib"""
    ver = version if version.startswith("v") else f"v{version}"
    return f"model_{ver}{ext}"


def build_model_path(model_id: str, version: str, ext: str = ".joblib") -> str:
    filename = version_to_filename(version, ext)
    return os.path.join(model_dir(model_id), filename)


def write_active_model(model_id: str, filename: str) -> None:
    os.makedirs(model_dir(model_id), exist_ok=True)
    path = active_marker_path(model_id)
    with open(path, "w", encoding="utf-8") as f:
        f.write(filename.strip() + "\n")


def read_active_filename(model_id: str) -> Optional[str]:
    path = active_marker_path(model_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    return line or None


def read_active_model_path(model_id: str) -> Optional[str]:
    filename = read_active_filename(model_id)
    if not filename:
        return None
    full = os.path.join(model_dir(model_id), filename)
    return full if os.path.exists(full) else None


def active_marker_mtime(model_id: str) -> float:
    path = active_marker_path(model_id)
    if os.path.exists(path):
        return os.path.getmtime(path)
    return 0.0


def list_artifact_files(model_id: str) -> list[str]:
    directory = model_dir(model_id)
    if not os.path.isdir(directory):
        return []
    return sorted(
        f
        for f in os.listdir(directory)
        if f.endswith((".joblib", ".pkl")) and f != ACTIVE_MARKER
    )
