from __future__ import annotations

"""Anomaly detection training and inference using the convolutional autoencoder.

Implements the full methodology from:
  V. Srivastava, "Autoencoder Optimization for Anomaly Detection," IEEE IJCNN 2024.

Key design decisions from the paper (all applied here):
  - BCE loss over MSE for feature-rich coloured images
  - [0, 1] normalisation (not ImageNet statistics) for the autoencoder
  - Three architecture variants trained in parallel: CAE-2Conv, CAE-3Conv, CAE-VariedFilter
  - Data augmentation: horizontal + vertical flip only (no rotation — orientation matters)
  - Early stopping with patience = 5 on validation BCE
  - Batch size 256
  - Model selection by AUC-ROC on a held-out normal/anomaly split
  - Anomaly threshold = 95th percentile of normal training reconstruction errors
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import wandb

from src.config import settings
from src.data.preprocessing import assert_safe_image_pixels
from src.models.autoencoder import ARCHITECTURES, _BaseAutoencoder, build_autoencoder


# ---------------------------------------------------------------------------
# [0,1] preprocessing — separate from ImageNet-normalised classifier pipeline
# ---------------------------------------------------------------------------

def _make_transform(image_size: int, augment: bool) -> transforms.Compose:
    ops: list = [transforms.Resize((image_size, image_size))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    ops += [transforms.ToTensor()]  # outputs float32 in [0, 1]
    return transforms.Compose(ops)


def preprocess_for_autoencoder(path: Path, image_size: int = 64) -> torch.Tensor:
    """Read any image and return a [0,1]-normalised CHW tensor (no ImageNet stats)."""
    try:
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read()[:3]  # first 3 bands
            max_val = float(np.iinfo(data.dtype).max) if np.issubdtype(data.dtype, np.integer) else 1.0
            arr = (data.astype(np.float32) / max_val).clip(0, 1)
            pil = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8))
        except Exception:
            pil = Image.open(path)
            assert_safe_image_pixels(*pil.size)
            pil = pil.convert("RGB")
    except Exception:
        pil = Image.open(path)
        assert_safe_image_pixels(*pil.size)
        pil = pil.convert("RGB")

    return _make_transform(image_size, augment=False)(pil)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnomalyDataset(Dataset[torch.Tensor]):
    def __init__(self, image_paths: list[Path], image_size: int = 64, augment: bool = False) -> None:
        self.paths = image_paths
        self.transform = _make_transform(image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            try:
                import rasterio
                with rasterio.open(self.paths[idx]) as src:
                    data = src.read()[:3]
                max_val = float(np.iinfo(data.dtype).max) if np.issubdtype(data.dtype, np.integer) else 1.0
                arr = (data.astype(np.float32) / max_val).clip(0, 1)
                pil = Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8))
            except Exception:
                pil = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            pil = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(pil)


def collect_image_paths(data_root: Path, classes: list[str] | None = None) -> list[Path]:
    root = Path(data_root)
    paths: list[Path] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        if classes and class_dir.name not in classes:
            continue
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            paths.extend(class_dir.glob(ext))
    return paths


# ---------------------------------------------------------------------------
# AUC-ROC evaluation (model selection criterion from the paper)
# ---------------------------------------------------------------------------

def _compute_auc_roc(
    model: _BaseAutoencoder,
    normal_paths: list[Path],
    anomaly_paths: list[Path],
    image_size: int,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []

    for label, paths in ((0, normal_paths), (1, anomaly_paths)):
        ds = AnomalyDataset(paths, image_size=image_size, augment=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                scores = model.reconstruction_error(batch).cpu().numpy()
                all_scores.extend(scores.tolist())
                all_labels.extend([label] * len(scores))

    if len(set(all_labels)) < 2:
        return 0.5  # degenerate case
    return float(roc_auc_score(all_labels, all_scores))


# ---------------------------------------------------------------------------
# Training one architecture (with early stopping)
# ---------------------------------------------------------------------------

def _train_one(
    arch_name: str,
    train_paths: list[Path],
    val_paths: list[Path],
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    device: torch.device,
    patience: int = 5,
) -> tuple[_BaseAutoencoder, float]:
    """Train a single architecture. Returns (model, best_val_loss)."""
    model = build_autoencoder(arch_name, image_size=image_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = AnomalyDataset(train_paths, image_size=image_size, augment=True)
    val_ds = AnomalyDataset(val_paths, image_size=image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    best_val_loss = float("inf")
    best_state: dict = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss += criterion(model(batch), batch).item() * batch.size(0)

        avg_train = train_loss / len(train_ds)
        avg_val = val_loss / len(val_ds)
        wandb.log({f"{arch_name}/train_bce": avg_train, f"{arch_name}/val_bce": avg_val, "epoch": epoch})
        print(f"  [{arch_name}] epoch {epoch}/{epochs}  train={avg_train:.6f}  val={avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{arch_name}] early stop at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_autoencoder(
    data_root: Path,
    checkpoint_dir: Path,
    normal_classes: list[str] | None,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    device: torch.device,
    wandb_project: str,
    wandb_mode: str,
    s3_bucket: str | None = None,
    s3_prefix: str = "checkpoints",
    aws_region: str = "eu-central-1",
    patience: int = 5,
    threshold_percentile: float = 95.0,
) -> Path:
    normal_paths = collect_image_paths(data_root, classes=normal_classes)
    if not normal_paths:
        raise FileNotFoundError(f"No images under {data_root} for classes={normal_classes}")

    # Collect anomaly paths for AUC-ROC model selection
    all_paths = collect_image_paths(data_root)
    anomaly_paths = [p for p in all_paths if p not in set(normal_paths)]

    # 80/20 train/val split on normal images
    rng = torch.Generator().manual_seed(42)
    n_val = max(1, int(0.2 * len(normal_paths)))
    n_train = len(normal_paths) - n_val
    train_ds_tmp, val_ds_tmp = random_split(normal_paths, [n_train, n_val], generator=rng)  # type: ignore[arg-type]
    train_paths = list(train_ds_tmp)
    val_normal_paths = list(val_ds_tmp)

    # Limit anomaly eval set for speed (max 500 images)
    anom_eval = anomaly_paths[:500] if len(anomaly_paths) > 500 else anomaly_paths

    run = wandb.init(
        project=wandb_project,
        mode=wandb_mode,
        config={
            "task": "anomaly_detection",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "image_size": image_size,
            "normal_classes": normal_classes,
            "loss": "BCE",
            "augmentation": "hflip+vflip",
            "patience": patience,
            "threshold_percentile": threshold_percentile,
        },
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Train all 3 architectures, pick by AUC-ROC ---
    best_arch: str = ""
    best_model: _BaseAutoencoder | None = None
    best_auc = -1.0

    for arch_name in ARCHITECTURES:
        print(f"\n=== Training {arch_name} ===")
        model, _ = _train_one(
            arch_name, train_paths, val_normal_paths,
            epochs=epochs, batch_size=batch_size, lr=lr,
            image_size=image_size, device=device, patience=patience,
        )
        auc = _compute_auc_roc(model, val_normal_paths, anom_eval, image_size, device)
        print(f"  [{arch_name}] AUC-ROC = {auc:.4f}")
        wandb.log({f"{arch_name}/auc_roc": auc})

        if auc > best_auc:
            best_auc = auc
            best_arch = arch_name
            best_model = model

    print(f"\nBest architecture: {best_arch}  (AUC-ROC = {best_auc:.4f})")
    assert best_model is not None

    # --- Compute 95th-percentile threshold on all normal training images ---
    best_model.eval()
    all_train_ds = AnomalyDataset(train_paths + val_normal_paths, image_size=image_size, augment=False)
    all_loader = DataLoader(all_train_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    recon_errors: list[float] = []
    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(device)
            errors = best_model.reconstruction_error(batch).cpu().numpy()
            recon_errors.extend(errors.tolist())

    threshold = float(np.percentile(recon_errors, threshold_percentile))
    print(f"Anomaly threshold ({threshold_percentile:.0f}th percentile): {threshold:.6f}")

    # --- Save checkpoint ---
    best_path = checkpoint_dir / "autoencoder_best.pt"
    torch.save(
        {
            "arch": best_arch,
            "model_state_dict": best_model.state_dict(),
            "image_size": image_size,
            "normal_classes": normal_classes,
            "auc_roc": best_auc,
            "threshold": threshold,
            "threshold_percentile": threshold_percentile,
        },
        best_path,
    )
    wandb.summary.update({"best_arch": best_arch, "best_auc_roc": best_auc, "threshold": threshold})

    if s3_bucket:
        from src.storage.s3 import upload_file_to_s3
        key = f"{s3_prefix.rstrip('/')}/autoencoder_best.pt"
        uri = upload_file_to_s3(best_path, s3_bucket, key, aws_region)
        wandb.summary["autoencoder_s3_uri"] = uri
        print(f"Uploaded autoencoder checkpoint to {uri}")

    if run:
        run.finish()
    return best_path


# ---------------------------------------------------------------------------
# Inference helpers (used by the API)
# ---------------------------------------------------------------------------

def load_autoencoder(
    checkpoint_path: Path, device: torch.device
) -> tuple[_BaseAutoencoder, int, float]:
    """Returns (model, image_size, threshold)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "CAE-3Conv")
    image_size = ckpt.get("image_size", 64)
    threshold = ckpt.get("threshold", 0.05)  # sensible default if checkpoint is old
    model = build_autoencoder(arch, image_size=image_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, image_size, threshold


def compute_anomaly_score(
    model: _BaseAutoencoder,
    image_path: Path,
    image_size: int,
    device: torch.device,
    threshold: float | None = None,
) -> dict:
    tensor = preprocess_for_autoencoder(image_path, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        score = float(model.reconstruction_error(tensor).item())
        heatmap = model.anomaly_heatmap(tensor).squeeze(0).cpu().numpy()

    hmin, hmax = float(heatmap.min()), float(heatmap.max())
    heatmap_norm = ((heatmap - hmin) / (hmax - hmin)).tolist() if hmax > hmin else heatmap.tolist()

    result: dict = {"anomaly_score": round(score, 6), "heatmap": heatmap_norm}
    if threshold is not None:
        result["is_anomaly"] = score > threshold
        result["threshold"] = threshold
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train satellite anomaly-detection autoencoder (IEEE IJCNN 2024 methodology).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--normal-classes", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--wandb-project", type=str, default="satellite-anomaly")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--s3-bucket", type=str, default=settings.s3_bucket)
    parser.add_argument("--s3-prefix", type=str, default=settings.s3_prefix)
    parser.add_argument("--aws-region", type=str, default=settings.aws_region)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = train_autoencoder(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        normal_classes=args.normal_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        image_size=args.image_size,
        device=device,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        aws_region=args.aws_region,
        patience=args.patience,
        threshold_percentile=args.threshold_percentile,
    )
    print(f"Best checkpoint saved to {best}")
