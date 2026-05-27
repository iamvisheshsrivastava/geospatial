from __future__ import annotations

import torch

from src.models.resnet import build_resnet50_classifier


def test_resnet50_classifier_output_shape():
    model = build_resnet50_classifier(num_classes=10, pretrained=False, freeze_backbone=True)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))

    assert tuple(output.shape) == (2, 10)
