from __future__ import annotations

"""LiDAR point cloud processing for forestry applications.

Reads LAS/LAZ files (standard LiDAR formats), normalises the point cloud,
and extracts per-tree structural metrics: height, crown radius, stem density,
and canopy cover fraction.

These are exactly the metrics forestry companies (e.g. 44moles) need for
automated forest inventories from airborne LiDAR surveys.

Dependencies: open3d, laspy (both pure-Python, no C++ build required).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Optional heavy imports — graceful fallback so the rest of the API stays up
try:
    import laspy
    _HAS_LASPY = True
except ImportError:  # pragma: no cover
    _HAS_LASPY = False

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover
    _HAS_OPEN3D = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PointCloudStats:
    """Summary statistics extracted from a normalised LiDAR point cloud."""
    num_points: int
    bbox_min: list[float]       # [x, y, z]
    bbox_max: list[float]       # [x, y, z]
    mean_height: float          # metres above ground (CHM mean)
    max_height: float           # canopy height model peak
    canopy_cover_fraction: float  # proportion of first-returns above 2 m
    stem_density_per_ha: float  # estimated stems per hectare (local maxima)


@dataclass
class TreeSegment:
    """A single tree detected by the individual tree segmentation algorithm."""
    tree_id: int
    centroid_xy: list[float]    # [x, y] in point-cloud CRS
    height_m: float             # max Z of the segment
    crown_radius_m: float       # estimated crown radius
    num_points: int


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_las(path: Path, max_points: int | None = None) -> np.ndarray:
    """Read a LAS/LAZ file and return an (N, 3) float32 XYZ array.

    `max_points`, when given, is checked against the header's point count
    *before* the (potentially expensive, decompression-heavy) full read, so an
    oversized or maliciously dense LAZ file is rejected cheaply instead of
    being fully decompressed into memory first.
    """
    if not _HAS_LASPY:
        raise ImportError("laspy is required: pip install laspy")
    with laspy.open(path) as f:
        if max_points is not None and f.header.point_count > max_points:
            raise ValueError(
                f"Point cloud has {f.header.point_count:,} points, which exceeds "
                f"the {max_points:,} point limit."
            )
        las = f.read()
    xyz = np.stack([
        np.asarray(las.x, dtype=np.float32),
        np.asarray(las.y, dtype=np.float32),
        np.asarray(las.z, dtype=np.float32),
    ], axis=1)
    return xyz


def normalise_height(xyz: np.ndarray, ground_percentile: float = 1.0) -> np.ndarray:
    """Subtract the ground surface estimate so Z represents height above ground."""
    ground_z = float(np.percentile(xyz[:, 2], ground_percentile))
    normalised = xyz.copy()
    normalised[:, 2] -= ground_z
    normalised[:, 2] = np.maximum(normalised[:, 2], 0.0)
    return normalised


# ---------------------------------------------------------------------------
# Canopy metrics
# ---------------------------------------------------------------------------

def compute_stats(xyz: np.ndarray, canopy_threshold_m: float = 2.0) -> PointCloudStats:
    """Compute stand-level forest structure metrics from a normalised point cloud."""
    heights = xyz[:, 2]
    canopy_mask = heights >= canopy_threshold_m
    canopy_cover = float(canopy_mask.sum()) / len(heights)

    # Estimate stem density from local Z-maxima on a 1 m grid
    xy = xyz[:, :2]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    cell = 1.0  # 1-metre grid cells
    nx = max(1, int((x_max - x_min) / cell))
    ny = max(1, int((y_max - y_min) / cell))
    chm = np.zeros((ny, nx), dtype=np.float32)
    ix = np.clip(((xy[:, 0] - x_min) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - y_min) / cell).astype(int), 0, ny - 1)
    np.maximum.at(chm, (iy, ix), heights)

    # Local maxima as tree tops (simple non-max suppression on 3×3 window)
    from scipy.ndimage import maximum_filter
    local_max = (chm == maximum_filter(chm, size=5)) & (chm > canopy_threshold_m)
    num_trees = int(local_max.sum())
    area_ha = (x_max - x_min) * (y_max - y_min) / 10_000
    stem_density = num_trees / max(area_ha, 1e-6)

    return PointCloudStats(
        num_points=len(xyz),
        bbox_min=xyz.min(axis=0).tolist(),
        bbox_max=xyz.max(axis=0).tolist(),
        mean_height=float(heights[canopy_mask].mean()) if canopy_mask.any() else 0.0,
        max_height=float(heights.max()),
        canopy_cover_fraction=round(canopy_cover, 4),
        stem_density_per_ha=round(stem_density, 1),
    )


# ---------------------------------------------------------------------------
# Individual Tree Segmentation (ITS) — watershed on CHM
# ---------------------------------------------------------------------------

def segment_trees(
    xyz: np.ndarray,
    cell_size: float = 0.5,
    min_tree_height: float = 3.0,
    min_points: int = 20,
) -> list[TreeSegment]:
    """Segment individual trees from a normalised LiDAR point cloud.

    Algorithm:
    1. Rasterise point cloud into a Canopy Height Model (CHM) at `cell_size` resolution.
    2. Detect local maxima (tree tops) with a 3×3 non-max suppression window.
    3. Assign each point to the nearest detected tree top (Voronoi / nearest-centroid).
    4. Compute per-tree metrics: height, crown radius, centroid.

    This mirrors production ITS pipelines used in airborne LiDAR surveys.
    """
    from scipy.ndimage import maximum_filter, label

    if len(xyz) == 0:
        return []

    xy = xyz[:, :2]
    heights = xyz[:, 2]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    nx = max(1, int(np.ceil((x_max - x_min) / cell_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell_size)))

    # Build CHM
    chm = np.zeros((ny, nx), dtype=np.float32)
    ix = np.clip(((xy[:, 0] - x_min) / cell_size).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - y_min) / cell_size).astype(int), 0, ny - 1)
    np.maximum.at(chm, (iy, ix), heights)

    # Detect tree tops as local maxima
    local_max_mask = (
        (chm == maximum_filter(chm, size=5))
        & (chm >= min_tree_height)
    )
    tree_top_positions = np.argwhere(local_max_mask)  # (K, 2) in grid coords [row, col]

    if len(tree_top_positions) == 0:
        return []

    # Convert tree-top grid positions back to world XY
    top_x = tree_top_positions[:, 1] * cell_size + x_min
    top_y = tree_top_positions[:, 0] * cell_size + y_min
    top_xy = np.stack([top_x, top_y], axis=1)  # (K, 2)

    # Assign each point to nearest tree top
    from scipy.spatial import cKDTree
    tree = cKDTree(top_xy)
    _, tree_ids = tree.query(xy, k=1, workers=1)

    segments: list[TreeSegment] = []
    for tid in range(len(top_xy)):
        mask = tree_ids == tid
        if mask.sum() < min_points:
            continue
        seg_xy = xy[mask]
        seg_z = heights[mask]
        centroid = seg_xy.mean(axis=0)
        crown_radius = float(np.linalg.norm(seg_xy - centroid, axis=1).max())
        segments.append(TreeSegment(
            tree_id=tid,
            centroid_xy=centroid.tolist(),
            height_m=round(float(seg_z.max()), 2),
            crown_radius_m=round(crown_radius, 2),
            num_points=int(mask.sum()),
        ))

    return segments


# ---------------------------------------------------------------------------
# Open3D visualisation helper (optional — used offline, not in API)
# ---------------------------------------------------------------------------

def to_open3d(xyz: np.ndarray, colour_by_height: bool = True):  # type: ignore[return]
    """Convert an XYZ array to an Open3D PointCloud object for visualisation."""
    if not _HAS_OPEN3D:
        raise ImportError("open3d is required: pip install open3d")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if colour_by_height and len(xyz):
        z = xyz[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-9)
        # Map height to a blue→green→yellow colour gradient
        colours = np.zeros((len(z), 3), dtype=np.float32)
        colours[:, 1] = z_norm          # green channel scales with height
        colours[:, 2] = 1.0 - z_norm   # blue channel inversely
        pcd.colors = o3d.utility.Vector3dVector(colours.astype(np.float64))
    return pcd


# ---------------------------------------------------------------------------
# Public pipeline — single call used by the API
# ---------------------------------------------------------------------------

def process_las_file(path: Path, max_points: int | None = None) -> dict[str, Any]:
    """Full pipeline: read → normalise → stats → segment trees.

    `max_points` caps the header-reported point count before the full
    decompress/read happens (see `read_las`) — guards against a small
    compressed LAZ file that decompresses to an unbounded number of points.

    Returns a JSON-serialisable dict ready for the /pointcloud API endpoint.
    """
    xyz_raw = read_las(path, max_points=max_points)
    xyz = normalise_height(xyz_raw)
    stats = compute_stats(xyz)
    trees = segment_trees(xyz)

    return {
        "stats": {
            "num_points": stats.num_points,
            "bbox_min": stats.bbox_min,
            "bbox_max": stats.bbox_max,
            "mean_canopy_height_m": stats.mean_height,
            "max_canopy_height_m": stats.max_height,
            "canopy_cover_fraction": stats.canopy_cover_fraction,
            "stem_density_per_ha": stats.stem_density_per_ha,
        },
        "trees": [
            {
                "tree_id": t.tree_id,
                "centroid_xy": t.centroid_xy,
                "height_m": t.height_m,
                "crown_radius_m": t.crown_radius_m,
                "num_points": t.num_points,
            }
            for t in trees
        ],
        "num_trees_detected": len(trees),
    }
