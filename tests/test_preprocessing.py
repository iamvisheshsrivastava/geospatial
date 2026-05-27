from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.data.preprocessing import preprocess_image, read_geospatial_rgb


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
