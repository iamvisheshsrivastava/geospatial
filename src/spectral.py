"""RGB-based spectral index computation for satellite imagery.

Implements three vegetation/water/urban proxy indices computable from
standard RGB imagery — inspired by notebooks/06_multispectral_features.ipynb.

Note: True Sentinel-2 spectral indices require NIR and SWIR bands not present
in RGB images.  These RGB approximations use established visible-band indices
(VARI, ExWI, ExUI) that correlate well with their multispectral counterparts.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.preprocessing import assert_safe_image_pixels


# ── Colourmap helpers ─────────────────────────────────────────────────────────

def _to_heatmap_png(matrix: np.ndarray, cmap_name: str) -> str:
    """Convert a (H, W) float array in [-1, 1] to a base64 PNG heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    normed = (np.clip(matrix, -1.0, 1.0) + 1.0) / 2.0   # [0, 1]
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(normed) * 255).astype(np.uint8)
    img = Image.fromarray(rgba[:, :, :3])                  # drop alpha

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Index computation ─────────────────────────────────────────────────────────

def compute_spectral_indices(image_path: Path, output_size: int = 64) -> dict:
    """Compute VARI, ExWI and ExUI indices from an RGB image.

    Args:
        image_path:   Path to the input image (any format PIL can read).
        output_size:  Spatial resolution of the returned heatmaps.

    Returns:
        Dict with keys:
          vegetation_b64, water_b64, urban_b64  — base64 PNG heatmaps
          vegetation_mean, water_mean, urban_mean — scalar index averages
          interpretation                          — human-readable summary
    """
    img = Image.open(image_path)
    assert_safe_image_pixels(*img.size)
    img = img.convert("RGB").resize((output_size, output_size))
    arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3), [0, 1]

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    eps = 1e-6

    # VARI — Visible Atmospherically Resistant Index (vegetation proxy)
    # Range ~[-1, 1]; positive = green vegetation, negative = bare/urban
    vari = (G - R) / (G + R - B + eps)
    vari = np.clip(vari, -1.0, 1.0)

    # ExWI — Excess Water Index (water proxy)
    # Highlights blue-dominant pixels (water bodies)
    exwi = (2.0 * B - R - G) / (2.0 * B + R + G + eps)
    exwi = np.clip(exwi, -1.0, 1.0)

    # ExUI — Excess Urban Index (built-up / bare-soil proxy)
    # Red-dominant pixels → urban, red-tile roofs, bare earth
    exui = (R - G) / (R + G + eps)
    exui = np.clip(exui, -1.0, 1.0)

    veg_mean   = float(vari.mean())
    water_mean = float(exwi.mean())
    urban_mean = float(exui.mean())

    # Human-readable interpretation of dominant land signal
    scores = {
        "Vegetation-dominant": veg_mean,
        "Water-dominant":      water_mean,
        "Urban/bare-soil":     urban_mean,
    }
    dominant = max(scores, key=scores.get)  # type: ignore[arg-type]

    interpretation = (
        f"Dominant signal: {dominant}. "
        f"Vegetation index (VARI): {veg_mean:+.3f} "
        f"| Water index (ExWI): {water_mean:+.3f} "
        f"| Urban index (ExUI): {urban_mean:+.3f}"
    )

    return {
        "vegetation_b64":   _to_heatmap_png(vari,  "RdYlGn"),   # green = veg
        "water_b64":        _to_heatmap_png(exwi,  "Blues"),     # blue = water
        "urban_b64":        _to_heatmap_png(exui,  "Oranges"),   # orange = urban
        "vegetation_mean":  round(veg_mean,   4),
        "water_mean":       round(water_mean, 4),
        "urban_mean":       round(urban_mean, 4),
        "interpretation":   interpretation,
    }
