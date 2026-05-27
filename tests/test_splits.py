from __future__ import annotations

from pathlib import Path

from src.data.dataset import ImageRecord, stratified_split


def test_stratified_split_preserves_total_count():
    records = [
        ImageRecord(path=Path(f"class_{label}_{idx}.tif"), label=label)
        for label in range(2)
        for idx in range(20)
    ]

    train, val, test = stratified_split(records, val_size=0.2, test_size=0.2, seed=7)

    assert len(train) + len(val) + len(test) == len(records)
    assert {record.label for record in train} == {0, 1}
    assert {record.label for record in val} == {0, 1}
    assert {record.label for record in test} == {0, 1}
