"""Train the whole-plate food segmenter (Ultralytics YOLO26-seg, AGPL-3.0) on FoodSeg103 and export it for the web app.

    python scripts/prepare_foodseg.py                               # once: dataset (~1.2 GB download)
    python scripts/train_plate.py                                   # GPU: yolo26s-seg, 60 epochs (~3-5 h on a 4 GB RTX 3050 Ti)
    python scripts/train_plate.py --model yolo26n-seg --epochs 40   # faster/smaller
    python scripts/train_plate.py --smoke                           # CPU pipeline check: 1 epoch, tiny images
    python scripts/train_plate.py --resume                          # continue a stopped run from its last epoch
    python scripts/train_plate.py --mix --hours 8                   # fine-tune on FoodSeg103 + Food Recognition 2022
                                                                    # (scripts/prepare_foodrec.py), fits in 8 hours

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
MIX = ROOT / "data" / "cache" / "plate_mix" / "data.yaml"
FRC_VAL = ROOT / "data" / "cache" / "foodrec2022" / "val.yaml"
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
    ap.add_argument("--resume", action="store_true", help="continue runs/plate/<model>/weights/last.pt")
    ap.add_argument("--mem-frac", type=float, default=0.9,
                    help="cap on the GPU's own memory; stops Windows spilling into (7x slower) shared RAM")
    ap.add_argument("--smoke", action="store_true", help="1 epoch at 256 px on CPU to test the pipeline")
    ap.add_argument("--mix", action="store_true",
                    help="train on FoodSeg103 + Food Recognition 2022, starting from the FoodSeg-trained model (much faster)")
    ap.add_argument("--hours", type=float, default=None,
                    help="time budget: the learning-rate schedule is fitted so training ends within this many hours")
    ap.add_argument("--force-export", action="store_true", help="export even if the new model scores lower")
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit("dataset missing: run `python scripts/prepare_foodseg.py` first")
    if args.mix and not MIX.exists():
        raise SystemExit("mixed dataset missing: run `python scripts/prepare_foodrec.py` first")

    if args.smoke:
        args.model = "yolo26n-seg"
    name = "smoke" if args.smoke else f"{args.model}-mix" if args.mix else args.model
    data = MIX if args.mix else DATA
    base = ROOT / "runs" / "plate" / args.model / "weights" / "best.pt"  # the first (FoodSeg103-only) model
    device = 0 if torch.cuda.is_available() and not args.smoke else "cpu"
    if device == 0:
        # Past the card's memory, the Windows driver silently borrows system RAM and training slows ~7x.
        # With a cap, PyTorch frees its cache and reuses memory instead of growing past the card.
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
        print(f"GPU: {torch.cuda.get_device_name(0)}, using at most {args.mem_frac:.0%} of its memory")
    last = ROOT / "runs" / "plate" / name / "weights" / "last.pt"
    if args.resume:
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} not found")
        model = YOLO(str(last))
        model.add_callback("on_train_epoch_start", lambda _t: torch.cuda.empty_cache())
        model.train(resume=True, batch=args.batch, workers=args.workers, device=device)
    else:
        # --mix starts from the model that already knows the 103 foods: it needs a fraction of the epochs
        start = str(base) if args.mix and base.exists() else f"{args.model}.pt"
        print(f"starting from {start}")
        model = YOLO(start)
        model.add_callback("on_train_epoch_start", lambda _t: torch.cuda.empty_cache())
        extra = {"optimizer": "AdamW", "lr0": 0.0005, "warmup_epochs": 0.5, "close_mosaic": 2} if args.mix else {}
        if args.hours:
            extra["time"] = args.hours  # Ultralytics fits the epoch count and schedule into this budget
        model.train(**{
            "data": str(data), "epochs": 1 if args.smoke else (20 if args.mix and args.epochs == 60 else args.epochs),
            "imgsz": 256 if args.smoke else args.imgsz, "batch": args.batch, "workers": args.workers, "device": device,
            "project": str(ROOT / "runs" / "plate"), "name": name, "exist_ok": True, "patience": 20, "cos_lr": True,
            "close_mosaic": 10, "seed": 0,
            # food-photo augmentation: colour/lighting, flips, scale
            "hsv_h": 0.015, "hsv_s": 0.6, "hsv_v": 0.4, "degrees": 10, "scale": 0.5, "fliplr": 0.5, "flipud": 0.1,
            "plots": not args.smoke, **extra,
        })
    run = ROOT / "runs" / "plate" / name
    best = run / "weights" / "best.pt"
    torch.cuda.empty_cache() if device == 0 else None
    size = 256 if args.smoke else args.imgsz

    def evaluate(weights: Path, yaml_path: Path):
        m = YOLO(str(weights)).val(data=str(yaml_path), imgsz=size, device=device, batch=args.batch, plots=False, verbose=False)
        return m

    metrics = evaluate(best, DATA)  # always scored on FoodSeg103 val, comparable across runs
    csv = run / "results.csv"  # one row per finished epoch (fewer than --epochs if early stopping kicked in)
    epochs = len(csv.read_text(encoding="utf-8").strip().splitlines()) - 1 if csv.exists() else args.epochs
    res = {"model": args.model, "epochs": epochs, "data": "FoodSeg103 + Food Recognition 2022" if args.mix else "FoodSeg103",
           "box_map50": round(float(metrics.box.map50), 4), "box_map50_95": round(float(metrics.box.map), 4),
           "mask_map50": round(float(metrics.seg.map50), 4), "mask_map50_95": round(float(metrics.seg.map), 4)}
    better = True
    if args.mix and FRC_VAL.exists():  # real phone photos: the test that matters for the app
        new = evaluate(best, FRC_VAL)
        res["phone_photos_mask_map50"] = round(float(new.seg.map50), 4)
        if base.exists():
            old = evaluate(base, FRC_VAL)
            res["phone_photos_mask_map50_before"] = round(float(old.seg.map50), 4)
            better = new.seg.map50 >= old.seg.map50
    print(json.dumps(res, indent=2))
    (run / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    if not better and not args.force_export:
        raise SystemExit("the new model is not better on phone photos: kept the current app model (use --force-export to override)")

    # export for the browser: fixed-size ONNX (NMS-free YOLO26 end-to-end output), simplified, opset 17
    onnx_path = Path(YOLO(str(best)).export(format="onnx", imgsz=size, opset=17, simplify=True, dynamic=False))
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, DOCS / "plate_seg.onnx")
    names = yaml.safe_load(DATA.read_text(encoding="utf-8"))["names"]
    (DOCS / "plate_meta.json").write_text(json.dumps({
        "imgsz": size, "names": [names[i] for i in range(len(names))], "metrics": res, "smoke": args.smoke,
        "model": f"Ultralytics {args.model} (AGPL-3.0) trained on FoodSeg103 (Apache-2.0)"
                 + (" + Food Recognition 2022 (MyFoodRepo, CC-BY-4.0)" if args.mix else "")}, indent=1), encoding="utf-8")
    print(f"exported -> {DOCS / 'plate_seg.onnx'} ({(DOCS / 'plate_seg.onnx').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
