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


def early_modules(model: nn.Module, n_stages: int) -> list[nn.Module]:
    """Stem + the first ``n_stages`` stages of a timm ConvNeXt (``stages``) or EfficientNet (``blocks``)."""
    if n_stages <= 0:
        return []
    if hasattr(model, "stages"):
        return [model.stem, *list(model.stages)[:n_stages]]
    if hasattr(model, "blocks"):
        return [m for m in (getattr(model, "conv_stem", None), getattr(model, "bn1", None)) if m is not None] + \
            list(model.blocks)[:n_stages]
    return []


def freeze_early_stages(model: nn.Module, n_stages: int) -> int:
    """Freeze the generic early layers (edges, colours, textures) when fine-tuning a model that already knows food.
    Less backward compute (faster) and less forgetting. Returns the number of frozen parameters."""
    n = 0
    for mod in early_modules(model, n_stages):
        mod.eval()  # keeps BatchNorm statistics fixed in EfficientNet; no-op for ConvNeXt's LayerNorm
        for p in mod.parameters():
            p.requires_grad = False
            n += p.numel()
    return n


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
    model.temperature = float(ckpt.get("temperature", 1.0))  # set by src/calibrate.py; 1.0 = uncalibrated
    model.to(device).eval()
    return model, ckpt["classes"], cfg


def warm_start(model: nn.Module, checkpoint: str | Path, classes: list[str]) -> dict:
    """Initialise ``model`` from an earlier NutriSnap checkpoint whose class list may differ.

    The backbone is copied as is. Classifier rows are copied *by class name*, so dishes the old model knew keep
    what they learned and only new dishes start from scratch. Returns {"copied": n_tensors, "known": n_classes}.
    """
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old_classes, old = ckpt["classes"], ckpt["state_dict"]
    own = model.state_dict()
    head = (getattr(model, "pretrained_cfg", None) or {}).get("classifier", "head.fc")
    pos = {c: i for i, c in enumerate(old_classes)}
    copied = 0
    with torch.no_grad():
        for k, v in old.items():
            if k not in own:
                continue
            v = v.to(own[k].dtype)
            if k.startswith(head + "."):  # classifier: match rows by dish name
                for j, c in enumerate(classes):
                    if c in pos:
                        own[k][j] = v[pos[c]]
                copied += 1
            elif own[k].shape == v.shape:
                own[k] = v
                copied += 1
    model.load_state_dict(own)
    return {"copied": copied, "known": sum(c in pos for c in classes)}
