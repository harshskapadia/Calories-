"""Calibrate the "this might be a dish I don't know" warning for the current web model -> docs/model/meta.json "ood".

The app flags a photo when the dish model's strongest raw score (max logit, after mirror averaging) is below the 5th
percentile of that score on photos of dishes it knows. So 5% of known-dish photos get a gentle warning; on 234 phone
photos of 120 dishes it was never trained on, this (plus the existing low-confidence rule) flagged ~65% (v5 model).
Run after every new dish model (after scripts/build_web.py):

    python scripts/calibrate_unknown.py                         # uses the newest test split in data/splits
    python scripts/calibrate_unknown.py --split data/splits/multiworld6_test.csv --n 1500
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "docs" / "model" / "meta.json"


def prep(path: Path, meta: dict) -> np.ndarray:
    """Exactly the browser's preprocessing (docs/app.js toTensor): resize, centre crop, normalise, + mirrored copy."""
    im = Image.open(path).convert("RGB")
    s = meta["resize"] / min(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    S = meta["img_size"]
    left, top = (im.width - S) // 2, (im.height - S) // 2
    a = (np.asarray(im.crop((left, top, left + S, top + S)), np.float32) / 255 - np.array(meta["mean"], np.float32)) / np.array(meta["std"], np.float32)
    a = a.transpose(2, 0, 1)
    return np.stack([a, a[:, :, ::-1]])


def main() -> None:
    import onnxruntime as ort

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, help="test manifest (path,label; paths relative to data/cache)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--percentile", type=float, default=5.0)
    args = ap.parse_args()
    split = args.split or max((ROOT / "data" / "splits").glob("*_test.csv"), key=lambda p: p.stat().st_mtime)
    rows = list(csv.DictReader(open(split, newline="", encoding="utf-8")))
    random.Random(0).shuffle(rows)
    meta = json.loads(META.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(ROOT / "docs" / "model" / "nutrisnap.int8.onnx"), providers=["CPUExecutionProvider"])
    scores = []
    for r in rows:
        p = ROOT / "data" / "cache" / r["path"]
        if not p.exists():
            continue
        logits = sess.run(None, {"image": prep(p, meta)})[0].mean(0)
        scores.append(float(logits.max()))
        if len(scores) >= args.n:
            break
    if len(scores) < 200:
        raise SystemExit(f"only {len(scores)} test photos found from {split}: export the data first")
    th = float(np.percentile(scores, args.percentile))
    meta["ood"] = {"max_logit": round(th, 3), "note": f"{args.percentile:g}th percentile of the max raw logit on "
                   f"{len(scores)} known test photos ({split.name})"}
    META.write_text(json.dumps(meta), encoding="utf-8")
    print(f"unknown-dish threshold: max logit < {th:.3f} ({len(scores)} photos from {split.name})")


if __name__ == "__main__":
    main()
