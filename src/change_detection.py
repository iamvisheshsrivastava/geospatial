from __future__ import annotations

"""Temporal change detection for satellite imagery.

Given two images of the same area at different times (before / after), this
module produces:
  - a per-pixel change heatmap in feature space (using the ResNet-50 encoder)
  - a scalar change score (0 = identical, 1 = maximum change)
  - an is_changed flag when a threshold is provided

Why feature-space diff instead of raw pixel diff?
  Raw pixel differences are sensitive to illumination, atmospheric conditions,
  and sensor noise. Comparing deep features from a pretrained encoder is more
  robust — the same technique used in production change-detection pipelines
  (e.g. Sentinel-2 land-cover monitoring).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from src.data.preprocessing import preprocess_image


# ---------------------------------------------------------------------------
# Feature extractor — shared encoder backbone
# ---------------------------------------------------------------------------

class ResNetEncoder(nn.Module):
    """ResNet-50 up to the last convolutional layer (before global pool)."""

    def __init__(self) -> None:
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Keep everything except avgpool and fc
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 2048, H/32, W/32)


_encoder: ResNetEncoder | None = None


def _get_encoder(device: torch.device) -> ResNetEncoder:
    global _encoder
    if _encoder is None:
        _encoder = ResNetEncoder().to(device).eval()
    return _encoder


# ---------------------------------------------------------------------------
# Core change detection
# ---------------------------------------------------------------------------

def detect_change(
    image_before: Path,
    image_after: Path,
    image_size: int,
    device: torch.device,
    threshold: float | None = None,
) -> dict:
    """Compare two satellite images and return a change map + scalar score.

    Args:
        image_before: Path to the earlier satellite image.
        image_after:  Path to the later satellite image.
        image_size:   Spatial size to resize both images to.
        device:       Torch device.
        threshold:    Optional float in [0, 1]. If provided, adds is_changed flag.

    Returns:
        dict with keys:
          - change_score: float in [0, 1]
          - change_map:   nested list (H_feat × W_feat), values in [0, 1]
          - is_changed:   bool (only when threshold is given)
    """
    encoder = _get_encoder(device)

    t_before = preprocess_image(image_before, image_size).unsqueeze(0).to(device)
    t_after  = preprocess_image(image_after,  image_size).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_before = encoder(t_before)  # (1, 2048, h, w)
        feat_after  = encoder(t_after)

        # Cosine distance per spatial position: 1 - cos_sim → [0, 2]
        # Normalise along channel dim
        feat_before_n = F.normalize(feat_before, dim=1)
        feat_after_n  = F.normalize(feat_after,  dim=1)
        cos_sim = (feat_before_n * feat_after_n).sum(dim=1)  # (1, h, w)
        change_map_raw = (1.0 - cos_sim).squeeze(0)          # (h, w), range [0, 2]

        # Normalise to [0, 1]
        change_map = (change_map_raw / 2.0).clamp(0.0, 1.0)

        # Scalar score: spatial mean
        change_score = float(change_map.mean().item())

    result: dict = {
        "change_score": round(change_score, 6),
        "change_map": change_map.cpu().numpy().tolist(),  # nested list
    }
    if threshold is not None:
        result["is_changed"] = change_score > threshold
        result["threshold"] = threshold
    return result


def change_map_to_rgb(change_map: list[list[float]]) -> np.ndarray:
    """Convert a normalised change map (H×W, [0,1]) to an RGB heatmap (H×W×3, uint8).

    Uses a blue→red colour gradient: blue = no change, red = high change.
    Useful for visualisation in the browser UI.
    """
    arr = np.array(change_map, dtype=np.float32)  # H×W
    r = (arr * 255).astype(np.uint8)
    g = np.zeros_like(r)
    b = (255 - r)
    return np.stack([r, g, b], axis=-1)  # H×W×3
