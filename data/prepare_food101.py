"""Download Food-101 from the Hugging Face Hub and build train/val/test manifests.

Two stages (both idempotent):

1. **export**: stream ``ethz/food101`` once and save every image as a JPEG with its
   short side resized to ``--short-side`` px (default 256) under
   ``data/cache/food101_256/{train,test}/<class>/<idx>.jpg`` plus an ``index.csv``.
   Pre-resizing makes CPU data loading ~5x faster than decoding the 512 px originals.
   The HF ``validation`` split is the official Food-101 *test* split (250 img/class).

2. **split**: from the index, write ``data/splits/<name>_{train,val,test}.csv``:
   - val is a stratified hold-out of the official train split (``val_frac``),
   - test is the official test split (never used for model selection),
   - optional subset: first/random ``num_classes`` and per-class caps for fast runs.

Usage:
    python data/prepare_food101.py --config configs/baseline_subset.yaml
    python data/prepare_food101.py --config configs/full.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
SPLITS = ROOT / "data" / "splits"
HF_DATASET = "ethz/food101"
HF_TO_LOCAL = {"train": "train", "validation": "test"}


# --------------------------------------------------------------------------- export
def _resize_and_save(args: tuple[bytes, str, int]) -> None:
    data, out_path, short_side = args
    img = Image.open(BytesIO(data)).convert("RGB")
    w, h = img.size
    scale = short_side / min(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    img.save(out_path, "JPEG", quality=90)


def export(short_side: int = 256, workers: int = os.cpu_count() or 2) -> Path:
    """Stream Food-101 from the HF Hub and save resized JPEGs + index.csv."""
    from datasets import Image as HFImage, load_dataset

    out_root = CACHE / f"food101_{short_side}"
    index_path = out_root / "index.csv"
    if index_path.exists():
        print(f"[export] already done: {index_path}")
        return out_root

    rows: list[tuple[str, str, str]] = []
    with ProcessPoolExecutor(workers) as pool:
        for hf_split, local_split in HF_TO_LOCAL.items():
            ds = load_dataset(HF_DATASET, split=hf_split, streaming=True)
            names = ds.features["label"].names
            ds = ds.cast_column("image", HFImage(decode=False))  # raw bytes; decode in workers
            for name in names:
                (out_root / local_split / name).mkdir(parents=True, exist_ok=True)
            batch, futures = [], []
            for i, ex in enumerate(ds):
                cls = names[ex["label"]]
                rel = f"{local_split}/{cls}/{i:06d}.jpg"
                batch.append((ex["image"]["bytes"], str(out_root / rel), short_side))
                rows.append((rel, cls, local_split))
                if len(batch) == 256:
                    futures.append(pool.submit(_map_batch, batch))
                    batch = []
                    # backpressure: bound queued image bytes; .result() re-raises worker errors
                    while len(futures) > 4 * workers:
                        futures.pop(0).result()
                if i % 5000 == 0:
                    print(f"[export] {hf_split}: {i} images", flush=True)
            if batch:
                futures.append(pool.submit(_map_batch, batch))
            for f in futures:
                f.result()
            print(f"[export] {hf_split}: done ({sum(r[2] == local_split for r in rows)} images)")

    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    print(f"[export] wrote {len(rows)} rows -> {index_path}")
    return out_root


def _map_batch(batch: list[tuple[bytes, str, int]]) -> None:
    for item in batch:
        _resize_and_save(item)


# ---------------------------------------------------------------------------- split
def make_splits(
    image_root: Path,
    name: str,
    num_classes: int | None = None,
    class_selection: str = "first",
    train_per_class: int | None = None,
    val_per_class: int | None = None,
    test_per_class: int | None = None,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, Path]:
    """Write stratified train/val/test manifests (paths relative to ``image_root``)."""
    by_split: dict[str, dict[str, list[str]]] = {"train": defaultdict(list), "test": defaultdict(list)}
    with open(image_root / "index.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_split[row["split"]][row["label"]].append(row["path"])

    classes = sorted(by_split["train"])
    rng = random.Random(seed)
    if num_classes and num_classes < len(classes):
        classes = sorted(rng.sample(classes, num_classes)) if class_selection == "random" else classes[:num_classes]

    out: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for cls in classes:
        train_paths = sorted(by_split["train"][cls])
        rng.shuffle(train_paths)
        n_val = round(len(train_paths) * val_frac)
        val, train = train_paths[:n_val], train_paths[n_val:]
        test = sorted(by_split["test"][cls])
        rng.shuffle(test)
        out["train"] += [(p, cls) for p in train[:train_per_class]]
        out["val"] += [(p, cls) for p in val[:val_per_class]]
        out["test"] += [(p, cls) for p in test[:test_per_class]]

    SPLITS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, rows in out.items():
        p = SPLITS / f"{name}_{split}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)
        paths[split] = p
        print(f"[split] {name}/{split}: {len(rows)} images, {len({c for _, c in rows})} classes")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["data"]

    image_root = export(cfg.get("short_side", 256), args.workers)
    make_splits(
        image_root,
        name=cfg["name"],
        num_classes=cfg.get("num_classes"),
        class_selection=cfg.get("class_selection", "first"),
        train_per_class=cfg.get("train_per_class"),
        val_per_class=cfg.get("val_per_class"),
        test_per_class=cfg.get("test_per_class"),
        val_frac=cfg.get("val_frac", 0.1),
        seed=cfg.get("seed", 42),
    )


if __name__ == "__main__":
    main()
