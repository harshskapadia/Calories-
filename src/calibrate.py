"""Confidence calibration by temperature scaling (Guo et al., 2017).

A single scalar T divides the logits: softmax(z / T). T is fitted on the validation split by
minimising negative log-likelihood, so it never changes *which* class is predicted, only how
confident the model claims to be. Fitted T is stored in the checkpoint and used by the app.

    python -m src.calibrate --checkpoint checkpoints/gpu_ablation_convnext_tiny/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import ManifestDataset, build_transforms, image_root_for, manifest_path
from src.models import load_checkpoint
from src.train import autocast_ctx, get_device


@torch.no_grad()
def collect_logits(model: nn.Module, loader, device: torch.device, amp: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        with autocast_ctx(device, amp):
            logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return T > 0 minimising NLL of softmax(logits / T) (optimised in log-space for stability)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """ECE: average |confidence - accuracy| over confidence bins, weighted by bin size."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - (pred[m] == labels[m]).mean())
    return float(ece)


def calibrate_checkpoint(checkpoint: Path, num_workers: int = 2, batch_size: int = 64) -> dict[str, float]:
    """Fit T on the checkpoint's validation split and write it into the checkpoint file."""
    device = get_device()
    model, classes, cfg = load_checkpoint(checkpoint, device)
    ds = ManifestDataset(manifest_path(cfg, "val"), image_root_for(cfg), classes,
                         build_transforms(cfg["train"].get("img_size", 224)))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    logits, labels = collect_logits(model, loader, device, cfg["train"].get("amp", True))
    t = fit_temperature(logits, labels)
    y = labels.numpy()
    before = expected_calibration_error(logits.softmax(1).numpy(), y)
    after = expected_calibration_error((logits / t).softmax(1).numpy(), y)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt["temperature"] = t
    torch.save(ckpt, checkpoint)
    out = {"temperature": round(t, 4), "val_ece_before": round(before, 4), "val_ece_after": round(after, 4)}
    print(f"[calibrate] {checkpoint}: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    calibrate_checkpoint(args.checkpoint, args.num_workers)


if __name__ == "__main__":
    main()
