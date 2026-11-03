"""Config-driven fine-tuning of a timm backbone on Food-101.

Features: AdamW + warmup/cosine LR, label smoothing, mixed precision (fp16 on CUDA,
bf16 autocast on CPU), optional backbone freezing for the first N epochs, early
stopping on validation accuracy, best/last checkpoints, CSV metrics + matplotlib
curves, and optional Weights & Biases logging (``--wandb``).

    python -m src.train --config configs/baseline_subset.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from src.data import build_loaders, manifest_path
from src.models import count_params, create_model, save_checkpoint, set_backbone_trainable

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, enabled: bool):
    """fp16 autocast on CUDA, bf16 on CPU; no-op when disabled/unsupported."""
    if not enabled or device.type == "mps":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def lr_lambda(warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device, amp: bool, criterion=None) -> dict[str, float]:
    model.eval()
    n, loss_sum, top1, top5 = 0, 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device, amp):
            logits = model(x)
        logits = logits.float()
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * len(y)
        k = min(5, logits.shape[1])
        topk = logits.topk(k, dim=1).indices
        top1 += (topk[:, 0] == y).sum().item()
        top5 += (topk == y[:, None]).any(dim=1).sum().item()
        n += len(y)
    return {"loss": loss_sum / max(n, 1), "acc": top1 / max(n, 1), "top5": top5 / max(n, 1)}


def plot_history(csv_path: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [float(r["train_loss"]) for r in rows], "o-", label="train")
    ax[0].plot(ep, [float(r["val_loss"]) for r in rows], "o-", label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(ep, [float(r["train_acc"]) for r in rows], "o-", label="train top-1")
    ax[1].plot(ep, [float(r["val_acc"]) for r in rows], "o-", label="val top-1")
    ax[1].plot(ep, [float(r["val_top5"]) for r in rows], "o--", label="val top-5")
    ax[1].set(title="Accuracy", xlabel="epoch", ylim=(0, 1))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def ensure_splits(cfg: dict[str, Any]) -> None:
    """Create the config's train/val/test manifests if they don't exist yet (data must be exported)."""
    if all(manifest_path(cfg, s).exists() for s in ("train", "val", "test")):
        return
    from data.prepare_food101 import splits_from_config  # local import: only needed once

    splits_from_config(cfg["data"])


def train(cfg: dict[str, Any], out_dir: Path, use_wandb: bool = False) -> dict[str, Any]:
    tcfg, mcfg = cfg["train"], cfg["model"]
    ensure_splits(cfg)
    seed_everything(tcfg.get("seed", 42))
    device = get_device()
    amp = bool(tcfg.get("amp", True))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    loaders, classes = build_loaders(cfg)
    model = create_model(mcfg["backbone"], len(classes), mcfg.get("pretrained", True), mcfg.get("drop_rate", 0.0))
    model.to(device)
    print(f"device={device} backbone={mcfg['backbone']} params={count_params(model)/1e6:.2f}M "
          f"classes={len(classes)} train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    # no weight decay on norms/biases
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.get("weight_decay", 0.05)}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg["lr"],
    )
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(int(steps_per_epoch * tcfg.get("warmup_epochs", 0)), total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    run = None
    if use_wandb:
        import wandb

        run = wandb.init(project="nutrisnap", name=cfg.get("experiment"), config=cfg)

    log_path = out_dir / "metrics.csv"
    fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5", "epoch_time_s"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_acc, best_epoch, bad_epochs = -1.0, -1, 0
    freeze_epochs = tcfg.get("freeze_backbone_epochs", 0)
    for epoch in range(1, tcfg["epochs"] + 1):
        set_backbone_trainable(model, epoch > freeze_epochs)
        model.train()
        t0, seen, correct, loss_sum = time.time(), 0, 0, 0.0
        for step, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_ctx(device, amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            if step % 50 == 0:
                print(f"  ep{epoch} step {step}/{steps_per_epoch} loss {loss.item():.3f}", flush=True)

        val = evaluate_loader(model, loaders["val"], device, amp, criterion)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_top5": val["top5"],
            "epoch_time_s": time.time() - t0,
        }
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{row[k]:.6g}" if isinstance(row[k], float) else row[k] for k in fields])
        print(f"epoch {epoch}: train_loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val_loss {val['loss']:.3f} acc {val['acc']:.3f} top5 {val['top5']:.3f} | {row['epoch_time_s']:.0f}s",
              flush=True)
        if run:
            run.log(row)

        save_checkpoint(out_dir / "last.pt", model, classes, cfg, epoch=epoch, val_acc=val["acc"])
        if val["acc"] > best_acc:
            best_acc, best_epoch, bad_epochs = val["acc"], epoch, 0
            save_checkpoint(out_dir / "best.pt", model, classes, cfg, epoch=epoch, val_acc=val["acc"])
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg.get("patience", 5):
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    plot_history(log_path, out_dir / "curves.png")
    from src.calibrate import calibrate_checkpoint  # local import: calibrate imports from this module

    calib = calibrate_checkpoint(out_dir / "best.pt", num_workers=tcfg.get("num_workers", 4) // 2)
    summary = {"experiment": cfg.get("experiment"), "best_val_acc": best_acc, "best_epoch": best_epoch,
               "params": count_params(model), "num_classes": len(classes), **calib}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if run:
        run.finish()
    return summary


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (value parsed as YAML), e.g. ``train.batch_size=128``."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # YAML 1.1 reads "3e-4" (no dot) as a string
            try:
                value = float(value)
            except ValueError:
                pass
        node[leaf] = value
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: checkpoints/<experiment>")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. --set train.num_workers=2 train.epochs=3")
    args = ap.parse_args()
    cfg = apply_overrides(yaml.safe_load(args.config.read_text()), args.set)
    out = args.out or ROOT / "checkpoints" / cfg["experiment"]
    print(json.dumps(train(cfg, out, args.wandb), indent=2))


if __name__ == "__main__":
    main()
