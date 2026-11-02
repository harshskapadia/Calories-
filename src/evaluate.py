"""Evaluate a checkpoint on the held-out test split.

Outputs (under ``results/<experiment>/``):
- ``metrics.json``           top-1, top-5, macro-F1, latency
- ``per_class.csv``          precision / recall / F1 / support per class
- ``confusion_matrix.png``   row-normalised confusion matrix
- ``confused_pairs.csv``     most-confused (true -> predicted) pairs
- ``predictions.csv``        per-image top-5 predictions (input to error analysis)

    python -m src.evaluate --checkpoint checkpoints/baseline_subset/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.data import ManifestDataset, build_transforms, image_root_for, manifest_path
from src.models import count_params, load_checkpoint
from src.train import autocast_ctx, get_device

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def predict_split(model, loader, device, amp: bool) -> np.ndarray:
    """Return softmax probabilities (N, C) for the whole loader."""
    out = []
    for x, _ in loader:
        with autocast_ctx(device, amp):
            logits = model(x.to(device))
        out.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def measure_latency(model, img_size: int, runs: int = 30, threads: int | None = None) -> float:
    """Median single-image CPU latency in ms (batch size 1, fp32)."""
    model = model.to("cpu").eval()
    if threads:
        torch.set_num_threads(threads)
    x = torch.randn(1, 3, img_size, img_size)
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def plot_confusion(cm: np.ndarray, classes: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    n = len(classes)
    size = max(6, n * 0.18) if n > 20 else 7
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ticks = range(n)
    ax.set_xticks(ticks, classes, rotation=90, fontsize=6 if n > 20 else 9)
    ax.set_yticks(ticks, classes, fontsize=6 if n > 20 else 9)
    ax.set(xlabel="predicted", ylabel="true", title="Confusion matrix (row-normalised)")
    if n <= 20:
        for i in range(n):
            for j in range(n):
                if cmn[i, j] >= 0.01:
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def evaluate(checkpoint: Path, split: str = "test", out_dir: Path | None = None,
             batch_size: int = 64, num_workers: int = 4) -> dict:
    device = get_device()
    model, classes, cfg = load_checkpoint(checkpoint, device)
    amp = cfg["train"].get("amp", True)
    img_size = cfg["train"].get("img_size", 224)
    ds = ManifestDataset(manifest_path(cfg, split), image_root_for(cfg), classes, build_transforms(img_size))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    out_dir = out_dir or ROOT / "results" / cfg.get("experiment", checkpoint.parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = predict_split(model, loader, device, amp)
    y = np.array([t for _, t in ds.samples])
    top5 = np.argsort(-probs, 1)[:, :5]
    pred = top5[:, 0]

    metrics = {
        "experiment": cfg.get("experiment"),
        "backbone": cfg["model"]["backbone"],
        "split": split,
        "num_images": int(len(y)),
        "num_classes": len(classes),
        "top1": float((pred == y).mean()),
        "top5": float((top5 == y[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "params_m": round(count_params(model) / 1e6, 2),
        "cpu_latency_ms": round(measure_latency(model, img_size), 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
    with open(out_dir / "per_class.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for i, c in sorted(enumerate(classes), key=lambda t: f[t[0]]):
            w.writerow([c, f"{p[i]:.4f}", f"{r[i]:.4f}", f"{f[i]:.4f}", int(s[i])])

    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    plot_confusion(cm, classes, out_dir / "confusion_matrix.png")
    pairs = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
    pairs.sort(reverse=True)
    with open(out_dir / "confused_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true", "predicted", "count", "rate_of_true_class"])
        for n, a, b in pairs[:30]:
            w.writerow([a, b, int(n), f"{n / cm[classes.index(a)].sum():.3f}"])

    with open(out_dir / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "true", "pred", "conf", "correct", "top5"])
        for (path, t), row, pr in zip(ds.samples, top5, probs):
            w.writerow([path, classes[t], classes[row[0]], f"{pr[row[0]]:.4f}", int(row[0] == t),
                        "|".join(f"{classes[k]}:{pr[k]:.3f}" for k in row)])

    print(json.dumps(metrics, indent=2))
    print("most confused:", [(a, b, int(n)) for n, a, b in pairs[:5]])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    evaluate(args.checkpoint, args.split, args.out, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
