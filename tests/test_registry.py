"""Filesystem registry helpers."""
import os

from app.models.registry import (
    read_active_filename,
    version_to_filename,
    write_active_model,
)


def test_version_to_filename():
    assert version_to_filename("v1", ".joblib") == "model_v1.joblib"
    assert version_to_filename("v12", ".pkl") == "model_v12.pkl"


def test_active_model_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path))
    from app.models import registry

    monkeypatch.setattr(registry, "MODELS_ROOT", str(tmp_path))

    write_active_model("demo", "model_v2.joblib")
    assert read_active_filename("demo") == "model_v2.joblib"
    assert os.path.exists(os.path.join(tmp_path, "demo", "active_model.txt"))
