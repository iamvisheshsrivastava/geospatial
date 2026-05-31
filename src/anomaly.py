from __future__ import annotations

"""Anomaly detection training and inference using the convolutional autoencoder.

Train the autoencoder on "normal" satellite patches (e.g. Forest class only, or
the full training split). At inference time, a high reconstruction error signals
that the input patch deviates from the learned normal distribution — indicating
deforestation, land-use change, or other anomalies.

Direct extension of published IEEE IJCNN 2024 work on autoencoder optimisation
for anomaly detection, applied to geospatial imagery.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb

from src.config import settings
from src.data.preprocessing import preprocess_image
from src.models.autoencoder import build_autoencoder, SatelliteAutoencoder


# ---------------------------------------------------------------------------
# Dataset — unsupervised: labels are ignored, we only use images
# ---------------------------------------------------------------------------

class UnlabelledImageDataset(Dataset[torch.Tensor]):
    def __init__(self, image_paths: list[Path], image_size: int = 224) -> None:
        self.paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return preprocess_image(self.paths[idx], self.image_size)


def collect_image_paths(data_root: Path, classes: list[str] | None = None) -> list[Path]:
    """Collect all image paths under data_root, optionally filtered by class name."""
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
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(
    data_root: Path,
    checkpoint_dir: Path,
    normal_classes: list[str] | None,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    latent_dim: int,
    device: torch.device,
    wandb_project: str,
    wandb_mode: str,
    s3_bucket: str | None,
    s3_prefix: str,
    aws_region: str,
) -> Path:
    image_paths = collect_image_paths(data_root, classes=normal_classes)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_root} for classes={normal_classes}")

    dataset = UnlabelledImageDataset(image_paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = build_autoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run = wandb.init(
        project=wandb_project,
        mode=wandb_mode,
        config={
            "task": "anomaly_detection",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "image_size": image_size,
            "latent_dim": latent_dim,
            "normal_classes": normal_classes,
            "num_training_images": len(image_paths),
        },
    )

    best_loss = float("inf")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "autoencoder_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        scheduler.step()

        avg_loss = epoch_loss / len(dataset)
        wandb.log({"epoch": epoch, "train/recon_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch}/{epochs}  recon_loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": latent_dim,
                    "image_size": image_size,
                    "normal_classes": normal_classes,
                    "best_loss": best_loss,
                },
                best_path,
            )

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

def load_autoencoder(checkpoint_path: Path, device: torch.device) -> tuple[SatelliteAutoencoder, int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 512)
    image_size = ckpt.get("image_size", 224)
    model = build_autoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, image_size


def compute_anomaly_score(
    model: SatelliteAutoencoder,
    image_path: Path,
    image_size: int,
    device: torch.device,
    threshold: float | None = None,
) -> dict:
    """Return anomaly score, heatmap (as nested list), and optional is_anomaly flag."""
    tensor = preprocess_image(image_path, image_size).unsqueeze(0).to(device)
    score = float(model.reconstruction_error(tensor).item())
    heatmap = model.anomaly_heatmap(tensor).squeeze(0).cpu().numpy()

    # Normalise heatmap to [0, 1] for JSON serialisation
    hmin, hmax = float(heatmap.min()), float(heatmap.max())
    if hmax > hmin:
        heatmap_norm = ((heatmap - hmin) / (hmax - hmin)).tolist()
    else:
        heatmap_norm = heatmap.tolist()

    result: dict = {
        "anomaly_score": round(score, 6),
        "heatmap": heatmap_norm,  # H×W nested list, values in [0,1]
    }
    if threshold is not None:
        result["is_anomaly"] = score > threshold
        result["threshold"] = threshold
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the satellite anomaly-detection autoencoder.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--normal-classes", nargs="*", default=None,
                        help="Class folder names to treat as 'normal'. Default: all classes.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=settings.image_size)
    parser.add_argument("--latent-dim", type=int, default=512)
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
        latent_dim=args.latent_dim,
        device=device,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        aws_region=args.aws_region,
    )
    print(f"Best checkpoint saved to {best}")
