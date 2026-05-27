from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import settings
from src.data.dataset import create_datasets
from src.metrics import classification_metrics
from src.models.resnet import build_resnet50_classifier
from src.storage.s3 import upload_file_to_s3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-50 satellite classifier.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=settings.image_size)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb-project", type=str, default="satellite-deforestation")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--s3-bucket", type=str, default=settings.s3_bucket)
    parser.add_argument("--s3-prefix", type=str, default=settings.s3_prefix)
    parser.add_argument("--aws-region", type=str, default=settings.aws_region)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            total_loss += float(loss.item()) * images.size(0)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(probs.argmax(dim=1).detach().cpu().tolist())
            probabilities.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = classification_metrics(y_true, y_pred, np.asarray(probabilities))
    return avg_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, _, class_names = create_datasets(
        args.data_root,
        image_size=args.image_size,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet50_classifier(
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        config={**vars(args), "num_classes": len(class_names), "class_names": class_names},
    )

    best_f1 = -1.0
    best_path = args.checkpoint_dir / "best_model.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        log_payload = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train/loss": train_loss,
            "val/loss": val_loss,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
        }
        wandb.log(log_payload)
        history.append(log_payload)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(best_path, model, class_names, val_metrics, args)
            wandb.save(str(best_path))

    history_path = args.checkpoint_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if args.s3_bucket:
        s3_key = f"{args.s3_prefix.rstrip('/')}/best_model.pt"
        artifact_uri = upload_file_to_s3(best_path, args.s3_bucket, s3_key, args.aws_region)
        wandb.summary["best_model_s3_uri"] = artifact_uri
        print(f"Uploaded best checkpoint to {artifact_uri}")
    else:
        print("S3_BUCKET not configured; best checkpoint saved locally only.")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
