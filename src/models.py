"""Model factory and checkpoint helpers (timm backbones)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

# Short names used in configs -> exact timm pretrained tags (reproducible weights).
BACKBONES = {
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "mobilenetv3_large_100": "mobilenetv3_large_100.ra_in1k",
    "resnet50": "resnet50.a1_in1k",
}


def create_model(backbone: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    name = BACKBONES.get(backbone, backbone)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    head = model.get_classifier()
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: str | Path, model: nn.Module, classes: list[str], cfg: dict[str, Any], **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "config": cfg, **extra}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, list[str], dict[str, Any]]:
    """Rebuild the model from a checkpoint (no pretrained download needed)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = create_model(cfg["model"]["backbone"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], cfg
