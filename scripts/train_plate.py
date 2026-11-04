"""Train the whole-plate food segmenter (Ultralytics YOLO26-seg, AGPL-3.0) on FoodSeg103 and export it for the web app.

    python scripts/prepare_foodseg.py                               # once: dataset (~1.2 GB download)
    python scripts/train_plate.py                                   # GPU: yolo26s-seg, 60 epochs (~3-5 h on a 4 GB RTX 3050 Ti)
    python scripts/train_plate.py --model yolo26n-seg --epochs 40   # faster/smaller
    python scripts/train_plate.py --smoke                           # CPU pipeline check: 1 epoch, tiny images

Writes runs/plate/<name>/ (Ultralytics outputs + metrics), docs/model/plate_seg.onnx and docs/model/plate_meta.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cache" / "foodseg103" / "data.yaml"
DOCS = ROOT / "docs" / "model"


def main() -> None:
    import torch
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26s-seg", help="yolo26n-seg (fast) | yolo26s-seg (default) | yolo26m-seg")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4, help="4 fits a 4 GB GPU at 640 px with yolo26s-seg; lower if out of memory")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="1 epoch at 256 px on CPU to test the pipeline")
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit("dataset missing: run `python scripts/prepare_foodseg.py` first")

    if args.smoke:
        args.model = "yolo26n-seg"
    name = "smoke" if args.smoke else args.model
    device = 0 if torch.cuda.is_available() and not args.smoke else "cpu"
    model = YOLO(f"{args.model}.pt")
    model.train(
        data=str(DATA), epochs=1 if args.smoke else args.epochs, imgsz=256 if args.smoke else args.imgsz,
        batch=args.batch, workers=args.workers, device=device, project=str(ROOT / "runs" / "plate"), name=name,
        exist_ok=True, patience=20, cos_lr=True, close_mosaic=10, seed=0,
        # food-photo augmentation: colour/lighting, flips, scale; no vertical flip of plates needed but harmless
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=10, scale=0.5, fliplr=0.5, flipud=0.1, plots=not args.smoke,
    )
    run = ROOT / "runs" / "plate" / name
    best = run / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(DATA), imgsz=256 if args.smoke else args.imgsz, device=device,
                                  plots=False, verbose=False)
    res = {"model": args.model, "epochs": args.epochs if not args.smoke else 1,
           "box_map50": round(float(metrics.box.map50), 4), "box_map50_95": round(float(metrics.box.map), 4),
           "mask_map50": round(float(metrics.seg.map50), 4), "mask_map50_95": round(float(metrics.seg.map), 4)}
    print(json.dumps(res, indent=2))
    (run / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # export for the browser: fixed-size ONNX (NMS-free YOLO26 end-to-end output), simplified, opset 17
    size = 256 if args.smoke else args.imgsz
    onnx_path = Path(YOLO(str(best)).export(format="onnx", imgsz=size, opset=17, simplify=True, dynamic=False))
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, DOCS / "plate_seg.onnx")
    names = yaml.safe_load(DATA.read_text(encoding="utf-8"))["names"]
    (DOCS / "plate_meta.json").write_text(json.dumps({
        "imgsz": size, "names": [names[i] for i in range(len(names))], "metrics": res, "smoke": args.smoke,
        "model": f"Ultralytics {args.model} (AGPL-3.0) trained on FoodSeg103 (Apache-2.0)"}, indent=1), encoding="utf-8")
    print(f"exported -> {DOCS / 'plate_seg.onnx'} ({(DOCS / 'plate_seg.onnx').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
