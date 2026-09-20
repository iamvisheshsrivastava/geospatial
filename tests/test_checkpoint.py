from __future__ import annotations

import pytest
import torch

from src.checkpoint import load_checkpoint


def test_loads_plain_checkpoint(tmp_path):
    p = tmp_path / "c.pt"
    torch.save({"class_names": ["a", "b"], "model_state_dict": {"w": torch.zeros(2)}}, p)
    assert load_checkpoint(p, map_location="cpu")["class_names"] == ["a", "b"]


class _Custom:
    pass


def test_unsafe_fallback_and_strict_mode(tmp_path, monkeypatch):
    p = tmp_path / "c.pt"
    torch.save({"obj": _Custom()}, p)
    monkeypatch.setenv("ALLOW_UNSAFE_CHECKPOINTS", "1")
    assert "obj" in load_checkpoint(p, map_location="cpu")
    monkeypatch.setenv("ALLOW_UNSAFE_CHECKPOINTS", "0")
    with pytest.raises(RuntimeError):
        load_checkpoint(p, map_location="cpu")
