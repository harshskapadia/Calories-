"""Grad-CAM (Selvaraju et al., 2017) for timm classifiers.

Uses timm's ``forward_features`` / ``forward_head`` split, so it works for any timm
backbone (EfficientNet, ConvNeXt, ResNet, MobileNet...) without naming a layer: the
CAM is computed on the last spatial feature map, right before global pooling.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn


def grad_cam(model: nn.Module, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
    """Return a (H, W) CAM in [0, 1] at feature-map resolution and the explained class.

    ``x`` is a normalised (1, 3, H, W) tensor. If ``class_idx`` is None, the top-1 class is used.
    """
    model.eval()
    with torch.enable_grad():
        x = x.detach().requires_grad_(False)
        feats = model.forward_features(x)  # (1, C, h, w)
        if feats.ndim != 4:
            raise ValueError(f"expected NCHW feature map, got {tuple(feats.shape)}")
        feats.retain_grad()
        logits = model.forward_head(feats)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        grads = feats.grad  # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
        cam = torch.relu((weights * feats).sum(dim=1))[0]  # (h, w)
    cam = cam.detach().float().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam, class_idx


def _colormap(cam: np.ndarray) -> np.ndarray:
    """Map [0,1] -> RGB uint8 using matplotlib's 'jet' (no OpenCV dependency needed)."""
    import matplotlib

    return (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Upsample the CAM to the image size and alpha-blend a heatmap on top."""
    image = image.convert("RGB")
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    heat = _colormap(np.asarray(cam_img, dtype=np.float32) / 255.0)
    blended = (1 - alpha) * np.asarray(image, dtype=np.float32) + alpha * heat.astype(np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
