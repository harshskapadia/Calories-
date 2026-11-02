"""Backbone ablation: train + evaluate several configs on identical data, write a results table.

    python -m src.ablation --configs configs/ablation_effnet_b0.yaml configs/ablation_convnext_tiny.yaml

Writes ``results/<name>.csv`` and ``results/<name>.md`` (default name: ablation) (accuracy, top-5, macro-F1,
params, CPU latency, training time per epoch).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from src.evaluate import evaluate
from src.train import apply_overrides, train

ROOT = Path(__file__).resolve().parents[1]


def run(configs: list[Path], skip_trained: bool = True, overrides: list[str] | None = None) -> list[dict]:
    rows = []
    for cfg_path in configs:
        cfg = apply_overrides(yaml.safe_load(cfg_path.read_text()), overrides)
        ckpt_dir = ROOT / "checkpoints" / cfg["experiment"]
        # summary.json is written only when training completes, so a run that was stopped mid-way retrains
        if not (skip_trained and (ckpt_dir / "summary.json").exists()):
            train(cfg, ckpt_dir)
        m = evaluate(ckpt_dir / "best.pt", "test")
        with open(ckpt_dir / "metrics.csv") as f:
            hist = list(csv.DictReader(f))
        m["epochs"] = len(hist)
        m["train_s_per_epoch"] = round(sum(float(r["epoch_time_s"]) for r in hist) / len(hist))
        rows.append(m)
    return rows


def write_table(rows: list[dict], out_dir: Path, name: str = "ablation") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["backbone", "top1", "top5", "macro_f1", "params_m", "cpu_latency_ms", "train_s_per_epoch",
            "epochs", "num_classes", "num_images"]
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = ["| Backbone | Top-1 | Top-5 | Macro-F1 | Params (M) | CPU latency (ms/img) | Train time (s/epoch) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['backbone']} | {r['top1']:.1%} | {r['top5']:.1%} | {r['macro_f1']:.3f} | "
                  f"{r['params_m']:.1f} | {r['cpu_latency_ms']:.0f} | {r['train_s_per_epoch']} |")
    text = "\n".join(md)
    (out_dir / f"{name}.md").write_text(text + "\n")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, nargs="+", required=True)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--name", default="ablation", help="results/<name>.md and .csv")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every config")
    args = ap.parse_args()
    rows = run(args.configs, skip_trained=not args.retrain, overrides=args.set)
    print(write_table(rows, ROOT / "results", args.name))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
