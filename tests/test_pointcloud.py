from __future__ import annotations

import numpy as np
import pytest

from src import pointcloud


def test_process_las_file_rejects_empty_point_cloud(tmp_path, monkeypatch):
    """A LAS/LAZ file with a 0-point header is spec-valid but every downstream
    step (normalise_height's np.percentile, compute_stats' division by
    len(heights)) assumes at least one point. Before the fix this surfaced as
    an opaque ZeroDivisionError/IndexError instead of a clean message.
    """
    monkeypatch.setattr(
        pointcloud, "read_las", lambda path, max_points=None: np.zeros((0, 3), dtype=np.float32)
    )

    dummy_path = tmp_path / "empty.las"
    dummy_path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        pointcloud.process_las_file(dummy_path)
