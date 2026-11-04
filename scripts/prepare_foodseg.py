"""FoodSeg103 (Apache-2.0) -> YOLO segmentation dataset, for whole-plate multi-food detection.

    python scripts/prepare_foodseg.py                 # all 7,118 images (~1.2 GB download, once)
    python scripts/prepare_foodseg.py --limit 300     # quick subset for a smoke test

FoodSeg103 gives one semantic mask per photo (pixel value = food class 1..103, 0 = background).
Each connected region of a class becomes one YOLO polygon instance; tiny specks are dropped.
Images are stored with their long side at 640 px. Output: data/cache/foodseg103/{images,labels}/{train,val}
and data/cache/foodseg103/data.yaml (class names, YOLO ids 0..102).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cache" / "foodseg103"
NAMES_FILE = ROOT / "data" / "foodseg103_classes.json"
MIN_AREA = 0.0015      # drop regions smaller than 0.15% of the image
MAX_SIDE = 640


def mask_to_polygons(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """[(class_id 1..103, Nx2 normalised polygon)] for every connected region of every class."""
    h, w = mask.shape
    out = []
    for cls in np.unique(mask):
        if cls == 0:
            continue
        binary = (mask == cls).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA * h * w:
                continue
            c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(c) < 3:
                continue
            out.append((int(cls), c.astype(np.float32) / np.array([w, h], np.float32)))
    return out


def main() -> None:
    from datasets import load_dataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="max images per split (smoke tests)")
    args = ap.parse_args()

    names = {int(k): v for k, v in json.loads(NAMES_FILE.read_text(encoding="utf-8")).items()}
    stats = {}
    for hf_split, split in [("train", "train"), ("validation", "val")]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        ds = load_dataset("EduardoPacheco/FoodSeg103", split=hf_split, streaming=True)
        n_img = n_inst = 0
        for i, ex in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            img = ex["image"].convert("RGB")
            mask = np.array(ex["label"].convert("L"))  # pixel = class id
            polys = mask_to_polygons(mask)
            if not polys:
                continue
            s = MAX_SIDE / max(img.size)
            if s < 1:
                img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
            stem = f"{hf_split}_{ex.get('id', i)}"
            img.save(OUT / "images" / split / f"{stem}.jpg", "JPEG", quality=90)
            with open(OUT / "labels" / split / f"{stem}.txt", "w", encoding="utf-8") as f:
                for cls, p in polys:  # YOLO ids are 0-based: FoodSeg class k -> k-1
                    f.write(f"{cls - 1} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in p) + "\n")
            n_img += 1
            n_inst += len(polys)
            if i % 500 == 0:
                print(f"[foodseg] {split}: {i} images", flush=True)
        stats[split] = {"images": n_img, "instances": n_inst}
    data = {"path": str(OUT), "train": "images/train", "val": "images/val",
            "names": {k - 1: names[k] for k in range(1, 104)}}
    (OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps(stats), "->", OUT / "data.yaml", flush=True)


if __name__ == "__main__":
    import os

    main()
    os._exit(0)  # skip interpreter teardown: datasets' streaming threads can crash it after an early stop
