"""Datasets and augmentations for Food-101 manifests produced by data/prepare_food101.py."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224, train: bool = False, aug: dict[str, Any] | None = None) -> T.Compose:
    """Train: RandomResizedCrop + flip (+ optional RandAugment / ColorJitter / RandomErasing).
    Eval: resize short side to img_size/0.875 then center crop (standard ImageNet protocol)."""
    aug = aug or {}
    if not train:
        return T.Compose(
            [
                T.Resize(int(round(img_size / 0.875))),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    ops: list[Any] = [
        T.RandomResizedCrop(img_size, scale=(aug.get("min_scale", 0.35), 1.0)),
        T.RandomHorizontalFlip(),
    ]
    if aug.get("randaugment"):
        ops.append(T.RandAugment(num_ops=2, magnitude=aug.get("ra_magnitude", 9)))
    cj = aug.get("color_jitter", 0.0)
    if cj:
        ops.append(T.ColorJitter(cj, cj, cj, min(cj / 4, 0.1)))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if aug.get("random_erasing", 0.0):
        ops.append(T.RandomErasing(p=aug["random_erasing"]))
    return T.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization (for visualisation)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


class ManifestDataset(Dataset):
    """Images listed in a ``path,label`` CSV; paths are relative to ``image_root``."""

    def __init__(self, manifest: str | Path, image_root: str | Path, classes: list[str], transform=None):
        self.image_root = Path(image_root)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        with open(manifest, newline="") as f:
            self.samples = [(r["path"], self.class_to_idx[r["label"]]) for r in csv.DictReader(f)]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        rel, y = self.samples[i]
        img = Image.open(self.image_root / rel).convert("RGB")
        return (self.transform(img) if self.transform else img), y


def classes_from_manifest(manifest: str | Path) -> list[str]:
    with open(manifest, newline="") as f:
        return sorted({r["label"] for r in csv.DictReader(f)})


def image_root_for(cfg: dict[str, Any]) -> Path:
    return ROOT / "data" / "cache" / f"food101_{cfg['data'].get('short_side', 256)}"


def manifest_path(cfg: dict[str, Any], split: str) -> Path:
    return ROOT / "data" / "splits" / f"{cfg['data']['name']}_{split}.csv"


def build_loaders(cfg: dict[str, Any]) -> tuple[dict[str, DataLoader], list[str]]:
    """Train/val/test DataLoaders + class list for an experiment config."""
    tcfg = cfg["train"]
    classes = classes_from_manifest(manifest_path(cfg, "train"))
    root = image_root_for(cfg)
    loaders = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"
        ds = ManifestDataset(
            manifest_path(cfg, split),
            root,
            classes,
            build_transforms(tcfg["img_size"], train=is_train, aug=cfg.get("augment")),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=tcfg["batch_size"] * (1 if is_train else 2),
            shuffle=is_train,
            drop_last=is_train,
            num_workers=tcfg.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=tcfg.get("num_workers", 4) > 0,
        )
    return loaders, classes
