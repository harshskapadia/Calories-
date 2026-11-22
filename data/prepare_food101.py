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
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

# Must be set before huggingface_hub is imported. The Xet transfer backend has a large per-file cost: for the tens of
# thousands of tiny files of the Central Asian dataset, plain HTTP downloads measured ~4.5x faster (58 vs 13 files/s).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows warning, printed once per file

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
    if load_sources()[source].get("type") == "local_folder":  # your own photos: re-read every time (you may add more)
        return export_local_folder(load_sources()[source], out_root, short_side)
    if index_path.exists():
        print(f"[export] {source}: already done ({index_path})")
        return out_root
    if spec.get("type") == "yolo_crops":
        return export_yolo_crops(spec, out_root, short_side)
    if spec.get("type") == "local_seg_crops":
        return export_local_seg_crops(spec, out_root, short_side)
    if spec.get("type") == "mmfood_names":
        return export_mmfood_names(spec, out_root, short_side)
    if spec.get("type") == "commons_thumbs":
        return export_commons_thumbs(spec, out_root, short_side)

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


def export_local_seg_crops(spec: dict, out_root: Path, short_side: int) -> Path:
    """A YOLO segmentation dataset already on disk (polygon labels, e.g. the Food Recognition 2022 copy made by
    scripts/prepare_foodrec.py) -> one classification image per kept item, cropped around its polygon.

    Items smaller than ``min_box`` of the photo are skipped; at most ``max_per_class`` crops per dish and split.
    """
    src = CACHE / spec["root"]
    if not (src / "images").exists():
        raise FileNotFoundError(f"{src} not found: run `python scripts/prepare_foodrec.py` first")
    names_map = yaml.safe_load((CACHE / spec["names_from"]).read_text(encoding="utf-8"))["names"]
    names = [names_map[i] for i in range(len(names_map))]
    keep, min_box, cap = spec["keep"], spec.get("min_box", 0.2), spec.get("max_per_class", 1500)
    rows: list[tuple[str, str, str]] = []
    for split_dir, local_split in spec["splits"].items():
        counts: dict[str, int] = defaultdict(int)
        files = sorted((src / "labels" / split_dir).glob("*.txt"))
        random.Random(0).shuffle(files)  # the per-class cap then takes a random sample, not the first photos
        for label_file in files:
            img = None
            for k, line in enumerate(label_file.read_text(encoding="utf-8").splitlines()):
                vals = line.split()
                if len(vals) < 7:
                    continue
                label = keep.get(names[int(vals[0])])
                if not label or counts[label] >= cap:
                    continue
                xs, ys = [float(v) for v in vals[1::2]], [float(v) for v in vals[2::2]]
                bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                if bw < min_box or bh < min_box:
                    continue
                if img is None:
                    img_path = src / "images" / split_dir / (label_file.stem + ".jpg")
                    if not img_path.exists():
                        break
                    img = Image.open(img_path).convert("RGB")
                W, H = img.size
                m = 0.1 * max(bw * W, bh * H)
                box = (max(0, min(xs) * W - m), max(0, min(ys) * H - m), min(W, max(xs) * W + m), min(H, max(ys) * H + m))
                crop = img.crop(tuple(round(v) for v in box))
                sc = short_side / min(crop.size)
                if sc < 1:
                    crop = crop.resize((round(crop.width * sc), round(crop.height * sc)), Image.BICUBIC)
                rel = f"{local_split}/{label}/{split_dir}_{label_file.stem[:40]}_{k}.jpg"
                (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
                crop.save(out_root / rel, "JPEG", quality=90)
                rows.append((rel, label, local_split))
                counts[label] += 1
        print(f"[export] {spec['root']}/{split_dir}: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}", flush=True)
    small = {c for c in {r[1] for r in rows} if sum(r[1] == c for r in rows) < spec.get("min_images", 40)}
    rows = [r for r in rows if r[1] not in small]
    if small:
        print(f"[export] dropped (too few photos): {sorted(small)}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    print(f"[export] {spec['root']}: wrote {len(rows)} crops -> {out_root / 'index.csv'}")
    return out_root


def export_mmfood_names(spec: dict, out_root: Path, short_side: int) -> Path:
    """MM-Food-100K (a CSV of photo URLs + dish names) -> photos of the chosen dish names, downloaded and resized.

    ``names`` maps our label to their dish names (lower-cased, exact). At most ``max_per_class`` photos per label
    (random, fixed seed); a stable ``test_frac`` of each label (by URL hash) becomes test. Already-downloaded photos
    are skipped, so a stopped export can simply be run again.
    """
    import hashlib
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_csv(hf_hub_download(spec["hf"], spec["csv"], repo_type="dataset"), usecols=["image_url", "dish_name"])
    df["dish"] = df.dish_name.astype(str).str.lower().str.strip()
    groups = {**spec["names"], **(spec.get("extra_names") or {})}
    to_label = {n.lower(): lab for lab, names in groups.items() for n in names}
    df = df[df.dish.isin(to_label)].drop_duplicates("image_url").copy()
    df["label"] = df.dish.map(to_label)
    # random cap per dish (same shuffle every run, so a smaller cap reuses photos already downloaded)
    df = df.sample(frac=1, random_state=0)
    rank = df.groupby("label").cumcount()
    cap = df.label.map(lambda l: spec.get("max_per_class_extra", 100) if l in (spec.get("extra_names") or {})
                       else spec.get("max_per_class", 300))
    df = df[rank < cap]
    jobs = []
    for url, label in zip(df.image_url, df.label):
        h = hashlib.md5(url.encode()).hexdigest()
        split = "test" if int(h[:8], 16) % 1000 < spec.get("test_frac", 0.1) * 1000 else "train"
        jobs.append((url, label, f"{split}/{label}/{h[:16]}.jpg", split))
    print(f"[export] {spec['hf']}: {len(jobs)} photos for {df.label.nunique()} dishes; downloading…", flush=True)

    def fetch(job) -> tuple | None:
        url, label, rel, split = job
        dst = out_root / rel
        if dst.exists():
            return (rel, label, split)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "NutriSnap-dataset-prep"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                img = Image.open(BytesIO(data)).convert("RGB")
                sc = short_side / min(img.size)
                if sc < 1:
                    img = img.resize((round(img.width * sc), round(img.height * sc)), Image.BICUBIC)
                dst.parent.mkdir(parents=True, exist_ok=True)
                img.save(dst, "JPEG", quality=90)
                return (rel, label, split)
            except Exception:
                if attempt == 2:
                    return None
        return None

    rows, seen, dupes = [], set(), 0
    with ThreadPoolExecutor(32) as pool:
        for i, r in enumerate(pool.map(fetch, jobs), 1):
            if r:
                # the same photo is often listed under several URLs: keep one copy, or it would leak into the test set
                digest = hashlib.md5((out_root / r[0]).read_bytes()).hexdigest()
                if digest in seen:
                    dupes += 1
                else:
                    seen.add(digest)
                    rows.append(r)
            if i % 2000 == 0:
                print(f"[export]   {i}/{len(jobs)} photos ({len(rows)} ok, {dupes} duplicates skipped)", flush=True)
    counts: dict[str, int] = defaultdict(int)
    for _, label, _ in rows:
        counts[label] += 1
    small = {c for c, n in counts.items() if n < spec.get("min_images", 60)}
    rows = [r for r in rows if r[1] not in small]
    print(f"[export] photos per dish: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}"
          + (f"; dropped (too few): {sorted(small)}" if small else ""), flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    print(f"[export] {spec['hf']}: wrote {len(rows)} photos ({dupes} duplicates skipped) -> {out_root / 'index.csv'}")
    return out_root


def dish_names() -> dict[str, str]:
    """Normalised dish name -> class key, for every dish the app has nutrition for: model classes (key and display name)
    and the Wikidata world dishes (wd_<QID>) in docs/data/foods.json."""
    import json
    import re
    import unicodedata

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
        return re.sub(r"[^a-z0-9]+", " ", s).strip()

    out = {}
    foods = json.loads((ROOT / "docs" / "data" / "foods.json").read_text(encoding="utf-8"))
    for r in foods:
        if r[0].startswith("wd_"):
            out.setdefault(norm(r[1]), r[0])
            out.setdefault(norm(r[0]), r[0])
    for r in foods:
        if not r[0].startswith(("wd_", "fndds_")):
            out[norm(r[1])] = r[0]
            out[norm(r[0])] = r[0]
    return out


def export_local_folder(spec: dict, out_root: Path, short_side: int) -> Path:
    """Your own photos: data/my_photos/<dish>/*.jpg, where <dish> is a dish the app knows (its name as shown in the app,
    e.g. "Halal snack pack" or "halal_snack_pack", or a class key like "pani_puri"). Photos are resized; every 5th
    photo (by content) is kept for testing. Re-read on every run, so you can keep adding photos."""
    import hashlib
    import re

    from PIL import ImageOps

    src = ROOT / spec["path"]
    names = dish_names()
    rows, unknown = [], []
    folders = sorted(p for p in src.iterdir() if p.is_dir()) if src.exists() else []
    for folder in folders:
        key = names.get(re.sub(r"[^a-z0-9]+", " ", folder.name.lower().replace("_", " ")).strip())
        if not key:
            unknown.append(folder.name)
            continue
        files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        for f in files:
            data = f.read_bytes()
            h = hashlib.md5(data).hexdigest()
            split = "test" if int(h[:8], 16) % 5 == 0 else "train"
            rel = f"{split}/{key}/{h[:16]}.jpg"
            dst = out_root / rel
            if not dst.exists():
                try:
                    img = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")  # phone photos: upright
                except Exception:  # noqa: BLE001
                    print(f"[export]   skipped {f} (not an image PIL can read)")
                    continue
                sc = short_side / min(img.size)
                if sc < 1:
                    img = img.resize((round(img.width * sc), round(img.height * sc)), Image.BICUBIC)
                dst.parent.mkdir(parents=True, exist_ok=True)
                img.save(dst, "JPEG", quality=90)
            rows.append((rel, key, split))
    if unknown:
        print(f"[export] my photos: no dish called {unknown} — rename the folder to a dish name from the app's search "
              "(e.g. 'Halal snack pack'); those photos are skipped")
    counts: dict[str, int] = defaultdict(int)
    for _, k, _ in rows:
        counts[k] += 1
    few = {k for k, n in counts.items() if n < spec.get("min_images", 15)}
    if few:
        print(f"[export] my photos: fewer than {spec.get('min_images', 15)} photos of {sorted(few)} — skipped for now")
    rows = sorted(set(r for r in rows if r[1] not in few))
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    print(f"[export] my photos: {len(rows)} photos of {len(counts) - len(few)} dishes ({dict(counts)})")
    return out_root


def export_commons_thumbs(spec: dict, out_root: Path, short_side: int) -> Path:
    """Wikimedia Commons dish photos (list: data/commons_dishes.csv.gz, made by scripts/build_commons_dishes.py, which
    runs first if the list is missing) -> small thumbnails, resized. Polite: a few parallel downloads, waits when the
    server says so. Already-downloaded photos are skipped, so a stopped export can simply be run again."""
    import gzip
    import hashlib
    import sys
    import time
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_commons_dishes import UA, ensure_manifest

    with gzip.open(ensure_manifest(), "rt", newline="", encoding="utf-8") as f:
        jobs = [(r["url"], r["label"], r["split"]) for r in csv.DictReader(f)]
    print(f"[export] Wikimedia Commons: {len(jobs)} photos of {len({j[1] for j in jobs})} dishes "
          f"(~{len(jobs) * 30 / 1e6:.1f} GB of small thumbnails); downloading…", flush=True)
    pause = {"until": 0.0}

    def fetch(job) -> tuple | None:
        url, label, split = job
        rel = f"{split}/{label}/{hashlib.md5(url.encode()).hexdigest()[:16]}.jpg"
        dst = out_root / rel
        if dst.exists():
            return (rel, label, split)
        for attempt in range(6):
            time.sleep(max(0.0, pause["until"] - time.time()))
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                    data = r.read()
                img = Image.open(BytesIO(data)).convert("RGB")
                sc = short_side / min(img.size)
                if sc < 1:
                    img = img.resize((round(img.width * sc), round(img.height * sc)), Image.BICUBIC)
                dst.parent.mkdir(parents=True, exist_ok=True)
                img.save(dst, "JPEG", quality=90)
                return (rel, label, split)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                # 429 / 5xx: everybody waits (Retry-After if given), then tries again
                pause["until"] = time.time() + min(int(e.headers.get("Retry-After") or 0) or 5 * (attempt + 1), 120)
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        return None

    rows, failed = [], 0
    with ThreadPoolExecutor(spec.get("workers", 6)) as pool:
        for i, r in enumerate(pool.map(fetch, jobs), 1):
            if r:
                rows.append(r)
            else:
                failed += 1
            if i % 2000 == 0:
                print(f"[export]   {i}/{len(jobs)} photos ({failed} failed)", flush=True)
    if failed > 0.2 * len(jobs):
        raise SystemExit(f"[export] {failed} of {len(jobs)} Commons photos failed: the server may be limiting downloads. "
                         "Wait a while and run the same command again (finished photos are kept).")
    counts: dict[str, int] = defaultdict(int)
    for _, label, _ in rows:
        counts[label] += 1
    small = {c for c, n in counts.items() if n < spec.get("min_images", 8) and c not in (spec.get("keep") or [])}
    rows = [r for r in rows if r[1] not in small]
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    print(f"[export] Wikimedia Commons: wrote {len(rows)} photos of {len(counts) - len(small)} dishes ({failed} failed) "
          f"-> {out_root / 'index.csv'}")
    return out_root


def build_combined_index(sources: list[str], short_side: int, name: str) -> Path:
    """Merge several exported sources into data/cache/index_<name>.csv (paths relative to data/cache),
    mapping labels through each source's aliases so shared dishes become one class."""
    specs = load_sources()
    out = CACHE / f"index_{name}.csv"
    # duplicate classes -> their parent dish, vague/non-dish classes left out, for every source (data/class_merges.yaml)
    merges = yaml.safe_load((ROOT / "data" / "class_merges.yaml").read_text(encoding="utf-8"))
    merge, gone = merges.get("merge") or {}, set(merges.get("drop") or [])
    # photos a vision model found are not the dish they are labelled as (scripts/clean_photos.py)
    sys.path.insert(0, str(ROOT / "scripts"))
    from clean_photos import rejected
    bad = rejected()
    n = skipped = 0
    with open(out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["path", "label", "split", "source"])
        for src in sources:
            d = specs[src]["dir"].replace("_256", f"_{short_side}")
            aliases = specs[src].get("aliases") or {}
            drop = set(specs[src].get("drop") or [])
            with open(CACHE / d / "index.csv", newline="", encoding="utf-8") as fi:
                for row in csv.DictReader(fi):
                    label = aliases.get(row["label"], row["label"])
                    label = merge.get(label, label)
                    if row["label"] in drop or label in gone:
                        continue
                    if (src, row["path"]) in bad:
                        skipped += 1
                        continue
                    w.writerow([f"{d}/{row['path']}", label, row["split"], src])
                    n += 1
    print(f"[index] {'+'.join(sources)}: {n} images -> {out}"
          + (f" ({skipped} photos left out: not the dish they are labelled as)" if skipped else ""))
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
    keep_splits_from: str | None = None,
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
        if len(test) < min_test and len(test) < 0.15 * (len(test) + len(train_paths)):
            # the source dataset has (almost) no test images for this dish: hold some out of its train pool
            n_test = max(min_test, round(len(train_paths) * val_frac))
            if len(train_paths) < 3 * n_test:  # a rare dish (a few dozen photos): keep two thirds for training
                n_test = len(train_paths) // 3
            test, train_paths = test + train_paths[:n_test], train_paths[n_test:]
        n_val = round(len(train_paths) * val_frac)
        val, train = train_paths[:n_val], train_paths[n_val:]
        out["train"] += [(p, cls) for p in train[:train_per_class]]
        out["val"] += [(p, cls) for p in val[:val_per_class]]
        out["test"] += [(p, cls) for p in test[:test_per_class]]

    if keep_splits_from:
        # a model warm-started from an earlier run must never be tested on photos that run trained on: every photo the
        # earlier run had keeps its split (merging or removing classes would otherwise reshuffle the held-out photos)
        prev = {}
        for split in ("train", "val", "test"):
            f = SPLITS / f"{keep_splits_from}_{split}.csv"
            if not f.exists():
                raise SystemExit(f"{f} not found: run the earlier model's data step first, or remove keep_splits_from")
            with open(f, newline="", encoding="utf-8") as fh:
                prev.update({r["path"]: split for r in csv.DictReader(fh)})
        moved = 0
        fixed: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
        for split, rows in out.items():
            for path, cls in rows:
                to = prev.get(path, split)
                moved += to != split
                fixed[to].append((path, cls))
        out = fixed
        print(f"[split] kept the {keep_splits_from} split for {sum(p in prev for r in out.values() for p, _ in r)} photos "
              f"({moved} moved back)")

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
                  test_per_class=d.get("test_per_class"), val_frac=d.get("val_frac", 0.1), seed=d.get("seed", 42),
                  keep_splits_from=d.get("keep_splits_from"))
    if not sources:
        root = export(short, workers) if export_missing else CACHE / f"food101_{short}"
        if not (root / "index.csv").exists():
            raise FileNotFoundError("Food-101 not exported yet: run `python run_all.py --steps data` first")
        return make_splits(root, **kwargs)
    specs = load_sources()
    for src in sources:
        idx = CACHE / specs[src]["dir"].replace("_256", f"_{short}") / "index.csv"
        if specs[src].get("type") == "local_folder" or not idx.exists():
            if not export_missing and specs[src].get("type") != "local_folder":
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
