"""Export a training checkpoint to a compact app model (fp16 weights + classes + config).

    python scripts/export_model.py checkpoints/improved_all101/best.pt   # -> app/model/nutrisnap.pt

fp16 halves the file (ConvNeXt-Tiny: 111 MB -> 56 MB, under GitHub's 100 MB limit);
``load_checkpoint`` casts weights back to fp32, so predictions are unchanged up to rounding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def export(src: Path, dst: Path) -> Path:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()}
    keep = {k: ckpt[k] for k in ("classes", "config", "epoch", "val_acc") if k in ckpt}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, **keep}, dst)
    print(f"{src} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB, {len(keep['classes'])} classes)")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "app" / "model" / "nutrisnap.pt")
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
