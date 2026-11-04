r"""Food Recognition 2022 (MyFoodRepo; Dataset Ninja release, CC0 / original CC-BY-4.0) -> extra YOLO training data.

    python scripts/prepare_foodrec.py              # download (~ a few GB, once), convert, write the mixed dataset
    python scripts/prepare_foodrec.py --limit 200  # quick check on 200 images per split
    python scripts/prepare_foodrec.py --tar "C:\Users\me\Downloads\food-recognition-2022-DatasetNinja.tar"  # browser download

Real smartphone meal photos with a polygon per food item. Their 498 classes are mapped onto the app's 103 FoodSeg103 classes
(scripts/foodrec_map.py), so the app and its nutrition table stay the same. Zero-energy items (water, ice) are dropped.

Writes data/cache/foodrec2022/{images,labels}/{train,val} and data/cache/plate_mix/data.yaml
(train = FoodSeg103 + Food Recognition; val = FoodSeg103, so scores stay comparable with the first model).
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import tarfile
import urllib.request
import zlib
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from foodrec_map import map_name  # noqa: E402

CACHE = ROOT / "data" / "cache"
RAW = CACHE / "foodrec2022" / "raw"
OUT = CACHE / "foodrec2022"
SEG = CACHE / "foodseg103"
MIX = CACHE / "plate_mix"
URL = ("https://www.dropbox.com/scl/fi/9esq3k4r8two9mhqizyvf/food-recognition-2022-DatasetNinja.tar"
       "?rlkey=s28sauo9frx6dkdrql3i6kn10&dl=1")
MIN_AREA = 0.0015
MAX_SIDE = 640


def use_system_certificates() -> None:
    """Trust the same certificates as Windows/macOS (and the browser). Antivirus HTTPS scanning or a school/office
    network re-signs traffic with its own root certificate, which Python's bundled list doesn't know; verification stays on."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        print("[foodrec] tip: `pip install truststore` if the download fails with CERTIFICATE_VERIFY_FAILED", flush=True)


def download(dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1e8:
        print(f"[foodrec] already downloaded: {dst}", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    print("[foodrec] downloading Food Recognition 2022 from Dataset Ninja (a few GB, once)…", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "NutriSnap-dataset-prep"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, step = 0, 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if got >= step:
                print(f"[foodrec]   {got / 1e9:.2f}" + (f" / {total / 1e9:.2f} GB" if total else " GB"), flush=True)
                step += 250 << 20
    tmp.replace(dst)


def safe_extract(tf: tarfile.TarFile, dst: Path) -> None:
    """Extract without letting any member escape dst (Python >= 3.11.4 has filter="data"; older ones need a manual check)."""
    try:
        tf.extractall(dst, filter="data")
        return
    except TypeError:  # older Python without extraction filters
        pass
    root = dst.resolve()
    members = []
    for m in tf.getmembers():
        target = (dst / m.name).resolve()
        if not (m.isfile() or m.isdir()) or (root != target and root not in target.parents):
            print(f"[foodrec] skipped unsafe archive entry: {m.name}", flush=True)
            continue
        members.append(m)
    tf.extractall(dst, members=members)


def bitmap_mask(obj: dict, h: int, w: int) -> np.ndarray:
    """Supervisely 'bitmap': base64(zlib(PNG)) placed at origin [x, y]."""
    png = zlib.decompress(base64.b64decode(obj["bitmap"]["data"]))
    im = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    m = im[:, :, 3] if im.ndim == 3 and im.shape[2] == 4 else (im if im.ndim == 2 else im[:, :, 0])
    x, y = obj["bitmap"]["origin"]
    full = np.zeros((h, w), np.uint8)
    mh, mw = min(m.shape[0], h - y), min(m.shape[1], w - x)
    full[y:y + mh, x:x + mw] = m[:mh, :mw] > 0
    return full


def object_polygons(obj: dict, h: int, w: int) -> list[np.ndarray]:
    """Normalised Nx2 polygons for one annotated object (polygon or bitmap geometry)."""
    kind = obj.get("geometryType")
    if kind == "polygon":
        pts = np.array(obj["points"]["exterior"], np.float32)
        if len(pts) < 3 or cv2.contourArea(pts) < MIN_AREA * h * w:
            return []
        return [pts / np.array([w, h], np.float32)]
    if kind == "bitmap":
        mask = bitmap_mask(obj, h, w)
        out = []
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < MIN_AREA * h * w:
                continue
            c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(c) >= 3:
                out.append(c.astype(np.float32) / np.array([w, h], np.float32))
        return out
    return []


def convert(project: Path, names: list[str], limit: int | None) -> dict:
    ids = {n: i for i, n in enumerate(names)}
    stats, unmapped = {}, {}
    ann_dirs = sorted(p for p in project.rglob("ann") if p.is_dir())
    if not ann_dirs:
        raise SystemExit(f"no Supervisely 'ann' folders under {project}: send this folder listing to the maintainer")
    for ann_dir in ann_dirs:
        ds = ann_dir.parent.name.lower()
        split = "val" if ds.startswith(("val", "test")) else "train"
        img_dir = ann_dir.parent / "img"
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_img = n_inst = 0
        for i, ann_path in enumerate(sorted(ann_dir.glob("*.json"))):
            if limit and i >= limit:
                break
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            h, w = ann["size"]["height"], ann["size"]["width"]
            lines = []
            for obj in ann.get("objects", []):
                target = map_name(obj.get("classTitle", ""))
                if target == "skip":
                    continue
                if target == "other ingredients":
                    unmapped[obj.get("classTitle", "")] = unmapped.get(obj.get("classTitle", ""), 0) + 1
                for poly in object_polygons(obj, h, w):
                    poly = np.clip(poly, 0, 1)
                    lines.append(f"{ids[target]} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in poly))
            if not lines:
                continue
            img_path = img_dir / ann_path.name[:-5]  # "x.jpg.json" -> "x.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            s = MAX_SIDE / max(img.size)
            if s < 1:
                img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
            stem = f"frc_{ds}_{Path(img_path.stem).stem}"
            img.save(OUT / "images" / split / f"{stem}.jpg", "JPEG", quality=90)
            (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_img += 1
            n_inst += len(lines)
            if i % 2000 == 0:
                print(f"[foodrec] {ds}: {i} images", flush=True)
        prev = stats.get(split, {"images": 0, "instances": 0})
        stats[split] = {"images": prev["images"] + n_img, "instances": prev["instances"] + n_inst}
    if unmapped:
        top = sorted(unmapped.items(), key=lambda kv: -kv[1])[:20]
        print("[foodrec] classes sent to 'other ingredients':", top, flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="max images per split folder (quick check)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded archive and extracted files")
    ap.add_argument("--tar", type=Path, default=None,
                    help="use an archive you downloaded yourself (e.g. in the browser) instead of downloading it")
    args = ap.parse_args()
    seg_yaml = SEG / "data.yaml"
    if not seg_yaml.exists():
        raise SystemExit("run `python scripts/prepare_foodseg.py` first (the mixed dataset uses its class list)")
    names_map = yaml.safe_load(seg_yaml.read_text(encoding="utf-8"))["names"]
    names = [names_map[i] for i in range(len(names_map))]

    if args.tar:
        tar_path = args.tar.expanduser().resolve()
        if not tar_path.exists():
            raise SystemExit(f"not found: {tar_path}")
    else:
        tar_path = RAW / "food-recognition-2022.tar"
        use_system_certificates()
        download(tar_path)
    project = RAW / "extracted"
    already = project.exists() and any(project.rglob("ann"))
    if not already:
        print("[foodrec] extracting…", flush=True)
        with tarfile.open(tar_path) as tf:  # archive from Dataset Ninja (Supervisely project)
            safe_extract(tf, project)
    stats = convert(project, names, args.limit)

    MIX.mkdir(parents=True, exist_ok=True)
    mix = {"path": str(CACHE), "train": ["foodseg103/images/train", "foodrec2022/images/train"],
           "val": "foodseg103/images/val", "names": {i: n for i, n in enumerate(names)}}
    (MIX / "data.yaml").write_text(yaml.safe_dump(mix, sort_keys=False, allow_unicode=True), encoding="utf-8")
    frc_val = {"path": str(OUT), "train": "images/train", "val": "images/val", "names": mix["names"]}
    (OUT / "val.yaml").write_text(yaml.safe_dump(frc_val, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps(stats), "->", MIX / "data.yaml", flush=True)
    if not args.keep_raw and not args.limit:
        shutil.rmtree(RAW, ignore_errors=True)  # only our own download/extraction, never a --tar file elsewhere  # frees several GB; the converted copy is all training needs
        print("[foodrec] removed the raw download (use --keep-raw to keep it)", flush=True)


if __name__ == "__main__":
    main()
