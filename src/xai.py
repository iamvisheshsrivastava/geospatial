"""GradCAM explainability for the ResNet-50 land-cover classifier.

Produces a per-pixel saliency overlay showing which image regions most
influenced the model's prediction — directly implements the XAI pipeline
described in notebooks/04_gradcam_xai.ipynb for production use.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import assert_safe_image_pixels, preprocess_image


def gradcam_explain(
    model: torch.nn.Module,
    image_path: Path,
    image_size: int,
    device: torch.device,
    target_class_idx: int,
) -> str:
    """Run GradCAM on the last ResNet-50 conv block.

    Args:
        model:             Trained ResNet-50 classifier (eval mode).
        image_path:        Path to the input image file.
        image_size:        Spatial size used during training (e.g. 224).
        device:            Torch device.
        target_class_idx:  Class index to explain (usually the predicted class).

    Returns:
        Base64-encoded PNG string of the GradCAM overlay.
        Can be used directly as `<img src="data:image/png;base64,...">`.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Original image for overlay (un-normalised, [0,1] float32)
    orig = Image.open(image_path)
    assert_safe_image_pixels(*orig.size)
    orig = orig.convert("RGB").resize((image_size, image_size))
    rgb_float = np.array(orig, dtype=np.float32) / 255.0

    # ImageNet-normalised tensor for the model
    tensor = preprocess_image(image_path, image_size).unsqueeze(0).to(device)

    # Target: last bottleneck block of ResNet-50 layer4
    # model.layer4 is untouched by build_resnet50_classifier (only fc is replaced)
    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(target_class_idx)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # (H, W)

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # (H, W, 3) uint8

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
