from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.data.preprocessing import (
    MAX_IMAGE_PIXELS,
    assert_safe_image_pixels,
    preprocess_image,
    read_geospatial_rgb,
)


def test_read_geospatial_rgb_scales_multiband_tif(tmp_path):
    raster_path = tmp_path / "sample.tif"
    data = np.stack(
        [
            np.full((8, 8), 1000, dtype=np.uint16),
            np.full((8, 8), 2000, dtype=np.uint16),
            np.full((8, 8), 3000, dtype=np.uint16),
        ]
    )
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=3,
        dtype=data.dtype,
        transform=from_origin(0, 0, 10, 10),
    ) as dst:
        dst.write(data)

    image = read_geospatial_rgb(raster_path)

    assert image.shape == (8, 8, 3)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0


def test_preprocess_image_returns_resnet_tensor(tmp_path):
    raster_path = tmp_path / "sample.tif"
    data = np.random.randint(0, 255, size=(3, 16, 16), dtype=np.uint8)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=3,
        dtype=data.dtype,
        transform=from_origin(0, 0, 10, 10),
    ) as dst:
        dst.write(data)

    tensor = preprocess_image(raster_path, image_size=32)

    assert tuple(tensor.shape) == (3, 32, 32)
    assert tensor.is_floating_point()


# ---------------------------------------------------------------------------
# issue #33 — memory guard must account for band count, not just width*height
# ---------------------------------------------------------------------------

def test_assert_safe_image_pixels_passes_under_cap_single_band():
    assert_safe_image_pixels(1000, 1000, bands=1)  # 1M effective px — fine


def test_assert_safe_image_pixels_rejects_when_scaled_by_bands():
    # 7000x7000 (~49M px) alone is under the 50M cap, but 200 bands pushes
    # the effective pixel count to ~9.8B — must be rejected.
    with pytest.raises(ValueError, match="exceeds"):
        assert_safe_image_pixels(7000, 7000, bands=200)


def test_assert_safe_image_pixels_bands_defaults_to_one():
    # Backward compatible: callers that don't pass `bands` get the old
    # width*height-only behaviour.
    assert_safe_image_pixels(7000, 7000)  # ~49M px, under the 50M cap


def test_read_geospatial_rgb_only_decodes_first_three_bands(tmp_path):
    """A high band-count GeoTIFF under the width*height cap must not allocate
    bands x H x W in memory — read_geospatial_rgb should only decode band
    count up to 3 regardless of how many bands the file has."""
    raster_path = tmp_path / "multiband.tif"
    n_bands = 10
    data = np.stack(
        [np.full((8, 8), i * 100, dtype=np.uint16) for i in range(n_bands)]
    )
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=n_bands,
        dtype=data.dtype,
        transform=from_origin(0, 0, 10, 10),
    ) as dst:
        dst.write(data)

    image = read_geospatial_rgb(raster_path)

    assert image.shape == (8, 8, 3)
