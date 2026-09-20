"""Safe checkpoint loading.

Tries ``torch.load(weights_only=True)`` first so a tampered checkpoint cannot
execute code. Falls back to the legacy pickle loader (with a warning) only when
``ALLOW_UNSAFE_CHECKPOINTS`` is not set to ``0``/``false``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _unsafe_allowed() -> bool:
    return os.environ.get("ALLOW_UNSAFE_CHECKPOINTS", "1").strip().lower() not in {"0", "false", "no"}


def load_checkpoint(path: str | Path, map_location: Any = None, **kwargs: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)
    except Exception as exc:
        if not _unsafe_allowed():
            raise RuntimeError(
                f"Checkpoint {path} failed safe loading and ALLOW_UNSAFE_CHECKPOINTS is disabled: {exc}"
            ) from exc
        logger.warning("Safe load of %s failed (%s); falling back to pickle loader.", path, exc)
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
