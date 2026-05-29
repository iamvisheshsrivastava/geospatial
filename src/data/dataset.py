from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.preprocessing import preprocess_image

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ImageRecord:
    path: Path
    label: int


def discover_records(data_root: Path) -> tuple[list[ImageRecord], list[str]]:
    """Walk a folder structured as data_root/<class_name>/<image_file>."""
    class_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    class_names = [d.name for d in class_dirs]
    records: list[ImageRecord] = []
    for label, class_dir in enumerate(class_dirs):
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            for img_path in class_dir.glob(ext):
                records.append(ImageRecord(path=img_path, label=label))
    return records, class_names


def stratified_split(
    records: list[ImageRecord],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    by_label: dict[int, list[ImageRecord]] = {}
    for r in records:
        by_label.setdefault(r.label, []).append(r)

    train, val, test = [], [], []
    for recs in by_label.values():
        recs = recs.copy()
        rng.shuffle(recs)
        n = len(recs)
        n_test = max(1, round(n * test_size))
        n_val = max(1, round(n * val_size))
        test.extend(recs[:n_test])
        val.extend(recs[n_test: n_test + n_val])
        train.extend(recs[n_test + n_val:])
    return train, val, test


class SatelliteDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        records: list[ImageRecord],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.image_size = image_size
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        if self.transform is not None:
            from src.data.preprocessing import read_geospatial_rgb, _hwc_to_pil
            hwc = read_geospatial_rgb(record.path)
            pil = _hwc_to_pil(hwc)
            tensor = self.transform(pil)
        else:
            tensor = preprocess_image(record.path, self.image_size)
        return tensor, record.label


def create_datasets(
    data_root: Path,
    image_size: int = 224,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[SatelliteDataset, SatelliteDataset, SatelliteDataset, list[str]]:
    records, class_names = discover_records(data_root)
    train_recs, val_recs, test_recs = stratified_split(records, val_size, test_size, seed)
    train_ds = SatelliteDataset(train_recs, image_size=image_size, augment=True)
    val_ds = SatelliteDataset(val_recs, image_size=image_size, augment=False)
    test_ds = SatelliteDataset(test_recs, image_size=image_size, augment=False)
    return train_ds, val_ds, test_ds, class_names
