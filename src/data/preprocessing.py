from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


_PIL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_geospatial_rgb(path: Path) -> np.ndarray:
    """Read a GeoTIFF (or standard image) and return a float32 HWC array in [0, 1].

    For standard image formats (jpg/png) PIL is used directly — rasterio is only
    invoked for GeoTIFF/raster formats to avoid the slow NotGeoreferencedWarning
    fallback that makes EuroSAT JPEG loading ~50× slower than necessary.
    """
    if Path(path).suffix.lower() in _PIL_EXTENSIONS:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        data = np.array(img).transpose(2, 0, 1).astype(np.float32)
        return (data / 255.0).transpose(1, 2, 0).astype(np.float32)

    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read()  # (bands, H, W)
    else:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        data = np.array(img).transpose(2, 0, 1).astype(np.float32)
        return (data / 255.0).transpose(1, 2, 0).astype(np.float32)

    # Normalise to [0, 1] per-band using the actual dtype max
    max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
    rgb = data[:3].astype(np.float32) / float(max_val)
    return rgb.transpose(1, 2, 0)  # HWC


def preprocess_image(path: Path, image_size: int = 224) -> torch.Tensor:
    """Read an image file and return an ImageNet-normalised CHW tensor."""
    try:
        hwc = read_geospatial_rgb(path)
        pil_img = _hwc_to_pil(hwc)
    except Exception:
        from PIL import Image
        pil_img = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    return transform(pil_img)


def _hwc_to_pil(hwc: np.ndarray):
    from PIL import Image
    uint8 = (np.clip(hwc, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(uint8)
