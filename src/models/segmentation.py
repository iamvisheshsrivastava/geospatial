from __future__ import annotations

"""Instance segmentation model for tree crown delineation.

Uses a Mask R-CNN backbone (ResNet-50 + FPN) pretrained on COCO, fine-tuned
(or used zero-shot) to detect and segment individual tree crowns in aerial /
satellite RGB imagery.

This addresses the "tree species detection and instance segmentation" requirement
in forestry ML roles. The model produces per-instance masks, bounding boxes,
and confidence scores — one instance per detected tree crown.

In production:
  - Fine-tune on a labelled aerial dataset (e.g. NEON Tree Crowns, ReforesTree).
  - The pretrained COCO weights already recognise organic blob-shaped objects
    and transfer reasonably well to top-down tree crowns with minimal fine-tuning.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_tree_segmentation_model(
    num_classes: int = 2,  # background + tree
    pretrained_backbone: bool = True,
    hidden_layer: int = 256,
) -> nn.Module:
    """Return a Mask R-CNN model configured for tree crown segmentation.

    Args:
        num_classes:         Number of output classes (default 2: background + tree).
                             Increase if you want per-species classification heads.
        pretrained_backbone: Load COCO-pretrained weights for the backbone.
        hidden_layer:        Hidden dim for the mask predictor head.

    Returns:
        A torchvision MaskRCNN model with replaced prediction heads.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Replace the box predictor head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


# ---------------------------------------------------------------------------
# Inference helper used by the API
# ---------------------------------------------------------------------------

def run_segmentation(
    model: nn.Module,
    image_tensor: torch.Tensor,
    confidence_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run Mask R-CNN inference on a single image tensor.

    Args:
        model:                Loaded Mask R-CNN model in eval mode.
        image_tensor:         CHW float tensor in [0, 1]. NOT ImageNet-normalised
                              (Mask R-CNN handles its own normalisation internally).
        confidence_threshold: Minimum score to include a detection.
        mask_threshold:       Binary threshold applied to soft mask logits.

    Returns:
        JSON-serialisable dict:
          - num_trees:    int
          - detections:   list of {box, score, mask_area_px}
          - masks_shape:  [H, W] of the output mask grid
    """
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor])[0]

    scores = outputs["scores"].cpu().numpy()
    boxes = outputs["boxes"].cpu().numpy()
    masks = (outputs["masks"].cpu().squeeze(1).numpy() > mask_threshold)

    keep = scores >= confidence_threshold
    scores = scores[keep]
    boxes = boxes[keep]
    masks = masks[keep]

    detections = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        detections.append({
            "box": [round(float(v), 1) for v in box],  # [x1, y1, x2, y2]
            "score": round(float(score), 4),
            "mask_area_px": int(masks[i].sum()),
        })

    h = w = 0
    if len(masks):
        h, w = masks[0].shape

    return {
        "num_trees": len(detections),
        "detections": detections,
        "masks_shape": [h, w],
    }


def load_segmentation_model(checkpoint_path: Path, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Load a fine-tuned Mask R-CNN checkpoint."""
    model = build_tree_segmentation_model(num_classes=num_classes, pretrained_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model
