"""Generate notebooks/03_lidar_open3d_forest.ipynb"""
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "id": f"md{len(cells)}", "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "id": f"c{len(cells)}", "metadata": {}, "source": src, "outputs": [], "execution_count": None})

md("""# Notebook 03 — LiDAR Forest Inventory with Open3D

End-to-end pipeline for **airborne LiDAR forest inventory** using the tools that
power production forestry software (e.g. [44moles](https://44moles.com)):

1. **Load & inspect** a LAS/LAZ point cloud with `laspy`
2. **Normalise** heights to ground-relative values
3. **Visualise** the raw cloud with **Open3D** (height-coloured)
4. **Build a Canopy Height Model (CHM)** — rasterise to 0.5 m grid
5. **Detect tree tops** via local maxima on the CHM
6. **Segment individual trees** (ITS — watershed / nearest-centroid)
7. **Visualise segmented trees** with per-tree colours in Open3D
8. **Extract forest metrics**: stem density, mean height, canopy cover

This notebook uses **synthetic LAS data** (clearly labelled) so it runs without
downloading airborne survey data. Instructions for real data are included.

**Install:** `pip install open3d laspy scipy numpy matplotlib`""")

md("## Setup")

code("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path('..').resolve()))

FIGURES_DIR = Path('../figures')
FIGURES_DIR.mkdir(exist_ok=True)
np.random.seed(42)

# Check Open3D
try:
    import open3d as o3d
    print(f'Open3D {o3d.__version__} available')
    HAS_OPEN3D = True
except ImportError:
    print('Open3D not installed — run: pip install open3d')
    print('Matplotlib fallback will be used for all 3D plots.')
    HAS_OPEN3D = False

import laspy
print(f'laspy available')

from src.pointcloud import (
    normalise_height, compute_stats, segment_trees, to_open3d
)
print('src.pointcloud imported OK')""")

md("""## 1. Load a LAS file

### Real data
```python
# Any LAS/LAZ file from an airborne LiDAR survey works.
# Free sources:
#   OpenTopography:  https://opentopography.org  (global ALS surveys, free academic)
#   USGS 3DEP:       https://apps.nationalmap.gov/downloader/  (USA coverage)
#   AHN (Netherlands): https://www.ahn.nl  (1–8 pts/m², free download)
#   Finnish NLS:     https://www.maanmittauslaitos.fi/en  (2 pts/m², free)

import laspy
las = laspy.read('path/to/your/survey.las')
xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
```

### Synthetic data (used below)
Simulates a 100 m × 100 m forest stand with ~40 trees at ~8 pts/m².""")

code("""# ── SYNTHETIC LAS GENERATOR ───────────────────────────────────────────────
# Mimics a 100m x 100m airborne LiDAR scan of a mixed-age forest:
#   - Ground returns (DTM surface with slight slope)
#   - Understory returns (1-3 m)
#   - Canopy returns (~15-25 m, clustered around individual trees)
# ─────────────────────────────────────────────────────────────────────────

AREA_M = 100          # plot size (metres)
POINT_DENSITY = 8     # pts / m²
N_TREES = 38          # individual trees

N_TOTAL = int(AREA_M**2 * POINT_DENSITY)
print(f'Simulating {N_TOTAL:,} points over {AREA_M}×{AREA_M} m plot...')

# Ground returns (70% of points, Z follows gentle slope)
n_ground = int(N_TOTAL * 0.70)
gx = np.random.uniform(0, AREA_M, n_ground)
gy = np.random.uniform(0, AREA_M, n_ground)
gz = 0.02 * gx + 0.01 * gy + np.random.normal(0, 0.05, n_ground)  # gentle slope + noise

# Tree canopy returns (25% of points, clustered around N_TREES centres)
n_canopy = int(N_TOTAL * 0.25)
tree_x = np.random.uniform(5, AREA_M - 5, N_TREES)
tree_y = np.random.uniform(5, AREA_M - 5, N_TREES)
tree_h = np.random.uniform(10, 28, N_TREES)   # tree heights in metres

assignments = np.random.randint(0, N_TREES, n_canopy)
cx = tree_x[assignments] + np.random.normal(0, 1.5, n_canopy)
cy = tree_y[assignments] + np.random.normal(0, 1.5, n_canopy)
crown_frac = np.random.beta(2, 3, n_canopy)   # returns distributed through crown depth
cz = gz.mean() + tree_h[assignments] * crown_frac + np.random.normal(0, 0.3, n_canopy)

# Understory returns (5%)
n_under = N_TOTAL - n_ground - n_canopy
ux = np.random.uniform(0, AREA_M, n_under)
uy = np.random.uniform(0, AREA_M, n_under)
uz = gz.mean() + np.random.uniform(0.5, 4.0, n_under)

# Stack all returns
xyz_raw = np.vstack([
    np.stack([gx, gy, gz], axis=1),
    np.stack([cx, cy, cz], axis=1),
    np.stack([ux, uy, uz], axis=1),
]).astype(np.float32)

print(f'Total points:  {len(xyz_raw):,}')
print(f'Z range:       {xyz_raw[:,2].min():.1f} m – {xyz_raw[:,2].max():.1f} m (absolute)')
print(f'XY extent:     {xyz_raw[:,0].ptp():.0f} m × {xyz_raw[:,1].ptp():.0f} m')
print(f'Point density: {len(xyz_raw) / AREA_M**2:.1f} pts/m²')""")

md("## 2. Height normalisation")

code("""# Subtract ground surface so Z = height above ground (not elevation above sea level)
xyz = normalise_height(xyz_raw, ground_percentile=1.0)

print('After normalisation:')
print(f'  Ground level: ~0 m (clipped to 0)')
print(f'  Max height:   {xyz[:,2].max():.1f} m')
print(f'  Mean canopy:  {xyz[xyz[:,2] > 2, 2].mean():.1f} m (returns above 2 m)')""")

md("""## 3. Open3D visualisation — raw point cloud

Open3D renders the full point cloud coloured by height.
In a Jupyter environment with Open3D installed, `o3d.visualization.draw_geometries`
opens an interactive 3D window. Here we also save a static matplotlib render
so the notebook displays correctly without a display server.""")

code("""def plot_cloud_matplotlib(xyz, title, colour_by='height', tree_ids=None, ax=None):
    \"\"\"Matplotlib fallback for static 3D point cloud rendering.\"\"\"
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    if colour_by == 'height':
        z = xyz[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-9)
        colours = cm.viridis(z_norm)
    elif colour_by == 'tree' and tree_ids is not None:
        n_trees = tree_ids.max() + 1
        palette = cm.tab20(np.linspace(0, 1, max(n_trees, 1)))
        colours = palette[tree_ids % len(palette)]
    else:
        colours = 'steelblue'

    # Subsample for speed (matplotlib is slow with >50k points)
    idx = np.random.choice(len(xyz), min(len(xyz), 30_000), replace=False)
    ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2],
               c=colours[idx] if isinstance(colours, np.ndarray) else colours,
               s=0.3, alpha=0.6, linewidths=0)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Height (m)')
    ax.set_title(title)
    if standalone:
        plt.tight_layout()
    return ax

# ── Open3D render ──────────────────────────────────────────────────────────
if HAS_OPEN3D:
    pcd = to_open3d(xyz, colour_by_height=True)
    print(f'Open3D PointCloud: {len(pcd.points):,} points')
    print('Opening interactive viewer... (close window to continue)')
    # In non-interactive environments, comment out the line below:
    # o3d.visualization.draw_geometries([pcd], window_name='LiDAR Forest — height coloured')

    # Save a headless screenshot via off-screen renderer
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1024, height=768)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(FIGURES_DIR / 'open3d_raw_cloud.png'))
        vis.destroy_window()
        print('Saved open3d_raw_cloud.png')
    except Exception as e:
        print(f'Off-screen render skipped ({e}) — matplotlib fallback used')

# Always show matplotlib version (works everywhere)
plot_cloud_matplotlib(xyz, 'Raw LiDAR point cloud — coloured by height (Open3D: viridis)')
plt.savefig(FIGURES_DIR / 'lidar_raw_cloud.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved lidar_raw_cloud.png')""")

md("## 4. Canopy Height Model (CHM)")

code("""# Rasterise point cloud to 0.5m CHM — standard resolution for tree-top detection
CELL = 0.5
x_min, y_min = xyz[:, :2].min(axis=0)
x_max, y_max = xyz[:, :2].max(axis=0)
nx = int(np.ceil((x_max - x_min) / CELL))
ny = int(np.ceil((y_max - y_min) / CELL))

chm = np.zeros((ny, nx), dtype=np.float32)
ix = np.clip(((xyz[:, 0] - x_min) / CELL).astype(int), 0, nx - 1)
iy = np.clip(((xyz[:, 1] - y_min) / CELL).astype(int), 0, ny - 1)
np.maximum.at(chm, (iy, ix), xyz[:, 2])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(chm, cmap='YlGn', origin='lower',
                     extent=[x_min, x_max, y_min, y_max])
axes[0].set_title(f'Canopy Height Model (CHM) — {CELL} m resolution')
axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
plt.colorbar(im0, ax=axes[0], label='Height (m)')

# Hillshade effect for visual clarity
from scipy.ndimage import gaussian_filter
smooth = gaussian_filter(chm, sigma=1)
axes[1].imshow(smooth, cmap='terrain', origin='lower',
               extent=[x_min, x_max, y_min, y_max])
axes[1].set_title('CHM — Gaussian-smoothed (for tree-top detection)')
axes[1].set_xlabel('X (m)')
plt.colorbar(plt.cm.ScalarMappable(cmap='terrain',
    norm=plt.Normalize(smooth.min(), smooth.max())), ax=axes[1], label='Height (m)')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'lidar_chm.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'CHM shape: {chm.shape}  ({ny * CELL:.0f} m × {nx * CELL:.0f} m at {CELL} m/cell)')""")

md("## 5. Individual Tree Segmentation (ITS)")

code("""# Segment trees using the production ITS pipeline from src/pointcloud.py
trees = segment_trees(xyz, cell_size=CELL, min_tree_height=5.0, min_points=15)
print(f'Trees detected: {len(trees)}  (planted {N_TREES} synthetic trees)')
print()
print(f'{"ID":>4}  {"Height (m)":>10}  {"Crown r (m)":>11}  {"Points":>8}')
print('-' * 42)
for t in sorted(trees, key=lambda x: -x.height_m)[:10]:
    print(f'{t.tree_id:>4}  {t.height_m:>10.1f}  {t.crown_radius_m:>11.2f}  {t.num_points:>8,}')
if len(trees) > 10:
    print(f'  ... {len(trees) - 10} more trees')""")

md("## 6. Open3D — segmented trees (per-tree colour)")

code("""# Assign each point a tree ID for colour-coded Open3D rendering
from scipy.spatial import cKDTree as _KDTree

centroids = np.array([t.centroid_xy for t in trees], dtype=np.float32)
if len(centroids):
    kd = _KDTree(centroids)
    _, point_tree_ids = kd.query(xyz[:, :2], k=1, workers=-1)
else:
    point_tree_ids = np.zeros(len(xyz), dtype=int)

# Per-tree colour palette (20-colour tab20)
palette = (cm.tab20(np.linspace(0, 1, max(len(trees), 1)))[:, :3])
colours_rgb = palette[point_tree_ids % len(palette)]

if HAS_OPEN3D:
    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd_seg.colors = o3d.utility.Vector3dVector(colours_rgb.astype(np.float64))
    print(f'Open3D segmented cloud: {len(pcd_seg.points):,} points, {len(trees)} tree colours')
    # Uncomment to open interactive viewer:
    # o3d.visualization.draw_geometries([pcd_seg], window_name='Individual Tree Segmentation')

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1024, height=768)
        vis.add_geometry(pcd_seg)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(str(FIGURES_DIR / 'open3d_segmented.png'))
        vis.destroy_window()
        print('Saved open3d_segmented.png')
    except Exception as e:
        print(f'Off-screen render skipped ({e})')

# Matplotlib version (always shown)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
plot_cloud_matplotlib(xyz, f'Individual Tree Segmentation — {len(trees)} trees detected',
                      colour_by='tree', tree_ids=point_tree_ids, ax=ax)

# Mark tree tops
for t in trees:
    ax.scatter(*t.centroid_xy, t.height_m, c='black', s=30, marker='^', zorder=5)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'lidar_segmented_trees.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved lidar_segmented_trees.png')""")

md("## 7. Forest inventory metrics")

code("""stats = compute_stats(xyz)

print('=' * 45)
print('  FOREST INVENTORY REPORT')
print('=' * 45)
print(f'  Plot area:            {AREA_M}m × {AREA_M}m = {AREA_M**2 / 10_000:.2f} ha')
print(f'  Total returns:        {stats.num_points:,}')
print(f'  Point density:        {stats.num_points / (AREA_M**2):.1f} pts/m²')
print(f'  Mean canopy height:   {stats.mean_height:.1f} m')
print(f'  Max canopy height:    {stats.max_height:.1f} m')
print(f'  Canopy cover:         {stats.canopy_cover_fraction:.1%}')
print(f'  Stem density:         {stats.stem_density_per_ha:.0f} trees/ha')
print(f'  Trees detected (ITS): {len(trees)}')
print('=' * 45)

# Summary chart
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Forest Inventory Summary — LiDAR + Open3D Pipeline', fontweight='bold')

# Height distribution
heights_canopy = xyz[xyz[:, 2] > 2, 2]
axes[0].hist(heights_canopy, bins=40, color='#2d6a4f', edgecolor='white', linewidth=0.4)
axes[0].axvline(stats.mean_height, color='orange', lw=2, label=f'Mean: {stats.mean_height:.1f} m')
axes[0].axvline(stats.max_height, color='red', lw=2, label=f'Max: {stats.max_height:.1f} m')
axes[0].set_xlabel('Height above ground (m)')
axes[0].set_ylabel('Point count')
axes[0].set_title('Canopy Height Distribution')
axes[0].legend(fontsize=8)

# Per-tree heights
tree_heights = sorted([t.height_m for t in trees], reverse=True)
axes[1].bar(range(len(tree_heights)), tree_heights,
            color=cm.YlGn(np.linspace(0.4, 0.9, len(tree_heights))))
axes[1].set_xlabel('Tree rank (tallest first)')
axes[1].set_ylabel('Height (m)')
axes[1].set_title(f'Individual Tree Heights (n={len(trees)})')

# Crown radius vs height scatter
crown_r = [t.crown_radius_m for t in trees]
heights_t = [t.height_m for t in trees]
sc = axes[2].scatter(heights_t, crown_r, c=heights_t, cmap='YlGn', s=60, edgecolors='k', lw=0.5)
axes[2].set_xlabel('Tree height (m)')
axes[2].set_ylabel('Crown radius (m)')
axes[2].set_title('Height vs Crown Radius')
plt.colorbar(sc, ax=axes[2], label='Height (m)')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'lidar_inventory_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved lidar_inventory_summary.png')""")

md("""## 8. Connecting to 44moles / production forestry

This notebook demonstrates the core technical stack used in production LiDAR forest inventory:

| Component | This notebook | Production equivalent |
|---|---|---|
| Point cloud I/O | `laspy` | laspy / PDAL / LAStools |
| 3D visualisation | **Open3D** | Open3D / CloudCompare |
| CHM rasterisation | numpy `maximum_filter` | PDAL writers.gdal |
| Tree detection | Local maxima on CHM | `lidR` (R) / custom CNN |
| ITS segmentation | Nearest-centroid Voronoi | Watershed / `lidR::segment_trees` |
| Metrics | Height, crown radius, density | Forest inventory database |

### Scaling to production (AWS pipeline)
```python
# Airflow DAG sketch for processing 10,000 LAS tiles/day
from airflow import DAG
from airflow.operators.python import PythonOperator

def process_tile(tile_s3_key: str) -> dict:
    # Download from S3
    local = download_from_s3(tile_s3_key)
    # Run pipeline
    result = process_las_file(local)
    # Write metrics to database
    write_to_rds(result, tile_s3_key)
    return result

with DAG('lidar_forest_inventory', schedule_interval='@daily') as dag:
    for tile in get_pending_tiles():
        PythonOperator(task_id=f'process_{tile}', python_callable=process_tile,
                       op_kwargs={'tile_s3_key': tile})
```

### References
- Dalponte, M., & Coomes, D. A. (2016). Tree-centric mapping of forest carbon density from airborne LiDAR. *Methods in Ecology and Evolution*.
- Zhen, Z., Quackenbush, L. J., & Zhang, L. (2016). Trends in automatic individual tree crown detection and delineation. *Remote Sensing*, 8(4).
- 44moles GmbH: https://44moles.com — next-generation LiDAR-based forest inventories""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

out = Path(__file__).parent.parent / "notebooks" / "03_lidar_open3d_forest.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {out}")
