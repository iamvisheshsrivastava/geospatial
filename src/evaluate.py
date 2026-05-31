from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import settings
from src.data.dataset import create_datasets
from src.metrics import classification_metrics, confusion_matrix_counts
from src.models.resnet import build_resnet50_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a satellite classifier checkpoint.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=settings.model_path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=settings.image_size)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_ds, discovered_class_names = create_datasets(
        args.data_root,
        image_size=args.image_size,
        seed=args.seed,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", discovered_class_names)

    model = build_resnet50_classifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(probs.argmax(dim=1).cpu().tolist())
            probabilities.extend(probs.cpu().numpy())

    metrics = classification_metrics(y_true, y_pred, np.asarray(probabilities))
    cm = confusion_matrix_counts(y_true, y_pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {"metrics": metrics, "class_names": class_names, "confusion_matrix": cm}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    display = ConfusionMatrixDisplay(confusion_matrix=np.asarray(cm), display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
