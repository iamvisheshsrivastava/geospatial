from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchvision.datasets import EuroSAT


def download_eurosat(data_root: Path) -> Path:
    """Download EuroSAT RGB imagery and return the class-folder directory."""

    data_root.mkdir(parents=True, exist_ok=True)
    dataset = EuroSAT(root=str(data_root), download=True)
    class_to_idx = dataset.class_to_idx
    dataset_dir = data_root / "eurosat" / "2750"

    with (data_root / "eurosat_class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, sort_keys=True)

    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the EuroSAT public dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_eurosat(args.data_root)
    print(f"EuroSAT dataset ready at {dataset_dir}")


if __name__ == "__main__":
    main()
