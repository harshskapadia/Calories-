"""Download Food-101 (and other datasets in data/sources.yaml) from the Hugging Face Hub and build
train/val/test manifests.

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


# --------------------------------------------------------------------------- export
def _resize_and_save(args: tuple[bytes, str, int]) -> None:
    data, out_path, short_side = args
    img = Image.open(BytesIO(data)).convert("RGB")
    w, h = img.size
    scale = short_side / min(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    img.save(out_path, "JPEG", quality=90)


SOURCES_FILE = ROOT / "data" / "sources.yaml"


def load_sources() -> dict[str, dict]:
    return yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))


def export(short_side: int = 256, workers: int = os.cpu_count() or 2, source: str = "food101") -> Path:
    """Stream one HF image-classification dataset and save resized JPEGs + index.csv.

    index.csv paths are relative to the source folder (data/cache/<dir>/). Labels may be a
    ClassLabel (int) or a plain string column; both are stored as strings.
    """
    from datasets import Image as HFImage, load_dataset

    spec = load_sources()[source]
    out_root = CACHE / spec["dir"].replace("_256", f"_{short_side}")
    index_path = out_root / "index.csv"
    if index_path.exists():
        print(f"[export] {source}: already done ({index_path})")
        return out_root
    if spec.get("type") == "yolo_crops":
        return export_yolo_crops(spec, out_root, short_side)

    rows: list[tuple[str, str, str]] = []
    with ProcessPoolExecutor(workers) as pool:
        for hf_split, local_split in spec["splits"].items():
            ds = load_dataset(spec["hf"], split=hf_split, streaming=True)
            feat = ds.features["label"]
            names = getattr(feat, "names", None)  # ClassLabel -> list of names; string column -> None
            ds = ds.cast_column("image", HFImage(decode=False))  # raw bytes; decode in workers
            batch, futures = [], []
            made: set[str] = set()
            for i, ex in enumerate(ds):
                cls = names[ex["label"]] if names else str(ex["label"]).strip().lower().replace(" ", "_")
                if cls not in made:
                    (out_root / local_split / cls).mkdir(parents=True, exist_ok=True)
                    made.add(cls)
                # prefix with the HF split so two HF splits mapped to one local split never collide
                rel = f"{local_split}/{cls}/{hf_split}_{i:06d}.jpg"
                batch.append((ex["image"]["bytes"], str(out_root / rel), short_side))
                rows.append((rel, cls, local_split))
                if len(batch) == 256:
                    futures.append(pool.submit(_map_batch, batch))
                    batch = []
                    # backpressure: bound queued image bytes; .result() re-raises worker errors
                    while len(futures) > 4 * workers:
                        futures.pop(0).result()
                if i % 5000 == 0:
                    print(f"[export] {source}/{hf_split}: {i} images", flush=True)
            if batch:
                futures.append(pool.submit(_map_batch, batch))
            for f in futures:
                f.result()
            print(f"[export] {source}/{hf_split}: done")

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    print(f"[export] {source}: wrote {len(rows)} rows -> {index_path}")
    return out_root


def export_yolo_crops(spec: dict, out_root: Path, short_side: int) -> Path:
    """Detection dataset (YOLO txt boxes, Roboflow layout) -> one classification image per kept box.

    Each box of a class listed in ``spec["keep"]`` is cropped with a 10% margin, resized and saved under
    ``<split>/<our label>/``. Small boxes (side < min_box of the photo) are skipped; dishes with fewer than
    ``min_images`` crops are dropped.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download, snapshot_download

    keep, min_box = spec["keep"], spec.get("min_box", 0.12)
    repo, kind = spec["hf"], "dataset"
    names = yaml.safe_load(Path(hf_hub_download(repo, "data.yaml", repo_type=kind)).read_text(encoding="utf-8"))["names"]
    # 1) the small box files only (~21k); 2) only the photos that contain a big enough box of a dish we keep.
    # Files are fetched one by one from the Hub, so many parallel requests matter more than bandwidth.
    print(f"[export] {repo}: downloading box files…", flush=True)
    root = Path(snapshot_download(repo, repo_type=kind, allow_patterns=["*/labels*/*.txt"], max_workers=32))
    jobs: dict[str, list[tuple[int, str, list[float]]]] = {}  # image path in repo -> [(k, label, box)]
    for split_dir in spec["splits"]:
        for label_file in sorted(root.glob(f"*/{split_dir}/labels*/*.txt")):
            boxes = []
            for k, ln in enumerate(l.split() for l in label_file.read_text(encoding="utf-8").splitlines() if l.strip()):
                if len(ln) >= 5 and names[int(ln[0])] in keep:
                    cx, cy, bw, bh = map(float, ln[1:5])
                    if bw >= min_box and bh >= min_box:
                        boxes.append((k, keep[names[int(ln[0])]], [cx, cy, bw, bh]))
            if boxes:
                rel_lbl = label_file.relative_to(root).as_posix()
                img_rel = rel_lbl.replace("/labels", "/images", 1).rsplit(".", 1)[0] + ".jpg"
                jobs[img_rel] = boxes
    print(f"[export] {repo}: {len(jobs)} photos contain dishes we keep; downloading them…", flush=True)

    def fetch(rel: str) -> tuple[str, str | None]:
        try:
            return rel, hf_hub_download(repo, rel, repo_type=kind)
        except Exception as e:  # a missing/renamed photo shouldn't stop the export
            print(f"[export]   skipped {rel}: {e}", flush=True)
            return rel, None

    local: dict[str, str | None] = {}
    with ThreadPoolExecutor(32) as pool:
        for i, (rel, path) in enumerate(pool.map(fetch, sorted(jobs)), 1):
            local[rel] = path
            if i % 1000 == 0:
                print(f"[export]   {i}/{len(jobs)} photos", flush=True)

    rows: list[tuple[str, str, str]] = []
    for img_rel, boxes in sorted(jobs.items()):
        split_dir = img_rel.split("/")[1]
        if not local.get(img_rel) or split_dir not in spec["splits"]:
            continue
        local_split = spec["splits"][split_dir]
        img = Image.open(local[img_rel]).convert("RGB")
        W, H = img.size
        stem = Path(img_rel).stem
        for k, label, (cx, cy, bw, bh) in boxes:
            m = 0.1 * max(bw * W, bh * H)
            box = (max(0, (cx - bw / 2) * W - m), max(0, (cy - bh / 2) * H - m),
                   min(W, (cx + bw / 2) * W + m), min(H, (cy + bh / 2) * H + m))
            crop = img.crop(tuple(round(v) for v in box))
            sc = short_side / min(crop.size)
            if sc < 1:
                crop = crop.resize((round(crop.width * sc), round(crop.height * sc)), Image.BICUBIC)
            rel = f"{local_split}/{label}/{split_dir}_{stem[:40]}_{k}.jpg"
            (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_root / rel, "JPEG", quality=90)
            rows.append((rel, label, local_split))
    print(f"[export] {repo}: {len(rows)} dish crops", flush=True)
    counts: dict[str, int] = defaultdict(int)
    for _, label, _ in rows:
        counts[label] += 1
    small = {c for c, n in counts.items() if n < spec.get("min_images", 40)}
    rows = [r for r in rows if r[1] not in small]
    print(f"[export] crops per dish: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}"
          + (f"; dropped (too few): {sorted(small)}" if small else ""), flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    print(f"[export] {spec['hf']}: wrote {len(rows)} crops -> {out_root / 'index.csv'}")
    return out_root


def build_combined_index(sources: list[str], short_side: int, name: str) -> Path:
    """Merge several exported sources into data/cache/index_<name>.csv (paths relative to data/cache),
    mapping labels through each source's aliases so shared dishes become one class."""
    specs = load_sources()
    out = CACHE / f"index_{name}.csv"
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["path", "label", "split", "source"])
        for src in sources:
            d = specs[src]["dir"].replace("_256", f"_{short_side}")
            aliases = specs[src].get("aliases") or {}
            drop = set(specs[src].get("drop") or [])
            with open(CACHE / d / "index.csv", newline="", encoding="utf-8") as fi:
                for row in csv.DictReader(fi):
                    if row["label"] in drop:
                        continue
                    w.writerow([f"{d}/{row['path']}", aliases.get(row["label"], row["label"]), row["split"], src])
                    n += 1
    print(f"[index] {'+'.join(sources)}: {n} images -> {out}")
    return out


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
    index_file: Path | None = None,
    min_test: int = 20,
) -> dict[str, Path]:
    """Write stratified train/val/test manifests (paths relative to ``image_root``).

    Classes with fewer than ``min_test`` test images get a test set held out of their train pool,
    so every class is evaluated. Food-101 (250 test images per class) is unaffected.
    """
    by_split: dict[str, dict[str, list[str]]] = {"train": defaultdict(list), "test": defaultdict(list)}
    with open(index_file or image_root / "index.csv", newline="", encoding="utf-8") as f:
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
        test = sorted(by_split["test"][cls])
        rng.shuffle(test)
        if len(test) < min_test:
            # the source dataset has (almost) no test images for this dish: hold some out of its train pool
            n_test = max(min_test, round(len(train_paths) * val_frac))
            test, train_paths = test + train_paths[:n_test], train_paths[n_test:]
        n_val = round(len(train_paths) * val_frac)
        val, train = train_paths[:n_val], train_paths[n_val:]
        out["train"] += [(p, cls) for p in train[:train_per_class]]
        out["val"] += [(p, cls) for p in val[:val_per_class]]
        out["test"] += [(p, cls) for p in test[:test_per_class]]

    SPLITS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, rows in out.items():
        p = SPLITS / f"{name}_{split}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)
        paths[split] = p
        print(f"[split] {name}/{split}: {len(rows)} images, {len({c for _, c in rows})} classes")
    return paths


def splits_from_config(d: dict, export_missing: bool = False, workers: int = os.cpu_count() or 2) -> dict[str, Path]:
    """Make the train/val/test manifests for a config's ``data`` section.

    Single-source configs (no ``sources`` key) keep the original Food-101 layout. Multi-source
    configs export any missing source (if ``export_missing``), merge them into one index with
    shared labels, and write manifests whose paths are relative to data/cache/.
    """
    short = d.get("short_side", 256)
    sources = d.get("sources")
    kwargs = dict(name=d["name"], num_classes=d.get("num_classes"), class_selection=d.get("class_selection", "first"),
                  train_per_class=d.get("train_per_class"), val_per_class=d.get("val_per_class"),
                  test_per_class=d.get("test_per_class"), val_frac=d.get("val_frac", 0.1), seed=d.get("seed", 42))
    if not sources:
        root = export(short, workers) if export_missing else CACHE / f"food101_{short}"
        if not (root / "index.csv").exists():
            raise FileNotFoundError("Food-101 not exported yet: run `python run_all.py --steps data` first")
        return make_splits(root, **kwargs)
    specs = load_sources()
    for src in sources:
        idx = CACHE / specs[src]["dir"].replace("_256", f"_{short}") / "index.csv"
        if not idx.exists():
            if not export_missing:
                raise FileNotFoundError(f"dataset {src!r} not exported yet: run `python run_all.py --steps data` first")
            export(short, workers, source=src)
    index = build_combined_index(sources, short, d["name"])
    return make_splits(CACHE, index_file=index, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    splits_from_config(yaml.safe_load(args.config.read_text(encoding="utf-8"))["data"], export_missing=True, workers=args.workers)


if __name__ == "__main__":
    main()
