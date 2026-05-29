#!/usr/bin/env python3
"""Download and unpack the EuroSAT RGB dataset.

EuroSAT is a public benchmark of 27,000 Sentinel-2 satellite patches
across 10 land-use / land-cover classes at 64×64 px resolution.

Source: https://github.com/phelber/EuroSAT
Paper:  Helber et al., "EuroSAT: A Novel Dataset and Deep Learning
        Benchmark for Land Use and Land Cover Classification", IEEE JSTARS 2019.

Usage:
    python scripts/download_eurosat.py --output-dir data/eurosat
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

EUROSAT_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
EUROSAT_ZIP_NAME = "EuroSAT.zip"

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def _progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  ({downloaded // 1_048_576} MB)")
        sys.stdout.flush()


def download(output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / EUROSAT_ZIP_NAME

    # Check if already extracted
    if not force and all((output_dir / cls).is_dir() for cls in CLASSES):
        print(f"EuroSAT already extracted at {output_dir}. Use --force to re-download.")
        return output_dir

    # Download
    if not zip_path.exists() or force:
        print(f"Downloading EuroSAT from {EUROSAT_URL} ...")
        urllib.request.urlretrieve(EUROSAT_URL, zip_path, reporthook=_progress)
        print()
    else:
        print(f"Zip already exists at {zip_path}, skipping download.")

    # Extract
    print(f"Extracting to {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # The zip unpacks into a sub-folder called "EuroSAT/2750/<class>/<image>"
    # Flatten so it becomes output_dir/<class>/<image>
    eurosat_inner = output_dir / "EuroSAT" / "2750"
    if eurosat_inner.exists():
        for cls_dir in eurosat_inner.iterdir():
            dest = output_dir / cls_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(cls_dir), str(dest))
        shutil.rmtree(output_dir / "EuroSAT", ignore_errors=True)
    else:
        # Some zip versions unpack differently — try a flat EuroSAT/ folder
        eurosat_flat = output_dir / "EuroSAT"
        if eurosat_flat.exists():
            for cls_dir in eurosat_flat.iterdir():
                dest = output_dir / cls_dir.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(cls_dir), str(dest))
            shutil.rmtree(eurosat_flat, ignore_errors=True)

    # Clean up zip
    zip_path.unlink(missing_ok=True)

    # Verify
    found = [cls for cls in CLASSES if (output_dir / cls).is_dir()]
    print(f"\nExtracted {len(found)}/{len(CLASSES)} classes:")
    for cls in CLASSES:
        cls_dir = output_dir / cls
        count = len(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.tif"))) if cls_dir.exists() else 0
        status = "OK" if cls_dir.exists() else "MISSING"
        print(f"  {status:7s}  {cls} ({count} images)")

    if len(found) < len(CLASSES):
        print("\nWARNING: some classes are missing. Check the zip structure manually.")
    else:
        print(f"\nDataset ready at: {output_dir.resolve()}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eurosat"),
                        help="Where to save and extract the dataset (default: data/eurosat)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-extract even if already present")
    args = parser.parse_args()
    download(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
