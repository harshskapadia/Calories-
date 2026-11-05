"""Build the in-browser NutriSnap site in docs/ (served free by GitHub Pages or a static HF Space).

    python scripts/build_web.py                      # uses app/model/nutrisnap.pt
    python -m http.server -d docs 8000               # preview at http://localhost:8000

What it writes:
- docs/model/nutrisnap.int8.onnx  the classifier exported to ONNX with int8 weights in every MatMul/Gemm
                                   (dynamic quantization: ~3x smaller, ~1.7x faster on CPU, same accuracy)
- docs/model/meta.json             classes, display names, temperature, preprocessing constants
- docs/data/foods.json             nutrition for every dish + all USDA FNDDS foods (for search)

The ONNX Runtime Web files in docs/ort/ are vendored from the npm package onnxruntime-web@1.30.0
(dist/ort.wasm.min.js, ort-wasm-simd-threaded.{mjs,wasm}) so the page needs no CDN.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def export_onnx(checkpoint: Path, out: Path) -> dict:
    """Export the checkpoint to ONNX (fp32) and quantize MatMul/Gemm weights to int8."""
    import sys

    sys.path.insert(0, str(ROOT))
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from src.models import load_checkpoint

    model, classes, cfg = load_checkpoint(checkpoint, "cpu")
    model.float().eval()
    size = cfg["train"].get("img_size", 224)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        # two outputs: dish logits, and the pooled image embedding just before the classifier ("fingerprint" used by
        # "Learns from you" to recognise photos similar to ones the user corrected; computed anyway, so it is free)
        class WithEmbedding(torch.nn.Module):
            def __init__(self, m: torch.nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor):
                feats = self.m.forward_features(x)
                emb = self.m.forward_head(feats, pre_logits=True)
                return self.m.get_classifier()(emb), emb

        torch.onnx.export(WithEmbedding(model), (torch.randn(1, 3, size, size),), fp32, input_names=["image"],
                          output_names=["logits", "embedding"], opset_version=17, dynamo=False,
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}})
        quantize_dynamic(fp32, out, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    print(f"ONNX int8 model -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return {"classes": classes, "temperature": float(getattr(model, "temperature", 1.0)),
            "img_size": size, "backbone": cfg["model"]["backbone"]}


def build_foods(dish_classes: list[str]) -> list[list]:
    """Compact nutrition rows: [key, name, kcal, protein, carbs, fat (per 100 g), serving_g, match, source]."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.nutrition import load_full_db

    db = load_full_db()
    rows = []
    for key in db.classes:
        f = db.get(key)
        rows.append([key, f.display_name, f.kcal_per_100g, f.protein_per_100g, f.carbs_per_100g,
                     f.fat_per_100g, f.serving_g, f.match, f.source if f.match != "usda" else ""])
    # recognisable dishes first, then USDA foods alphabetically
    rows.sort(key=lambda r: (r[0].startswith("fndds_"), r[1].lower()))
    missing = [c for c in dish_classes if c not in db]
    if missing:
        raise SystemExit(f"classes without nutrition: {missing}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "app" / "model" / "nutrisnap.pt")
    args = ap.parse_args()

    meta = export_onnx(args.checkpoint, DOCS / "model" / "nutrisnap.int8.onnx")
    import sys

    sys.path.insert(0, str(ROOT))
    from src.data import IMAGENET_MEAN, IMAGENET_STD
    from src.nutrition import load_db

    db = load_db()
    meta.update({
        "display_names": [db.get(c).display_name for c in meta["classes"]],
        "resize": int(round(meta["img_size"] / 0.875)), "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
        "low_confidence": 0.4, "portion_multipliers": {"small": 0.6, "medium": 1.0, "large": 1.5},
    })
    (DOCS / "model" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    foods = build_foods(meta["classes"])
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    # keep the Wikidata world dishes (wd_*) that scripts/build_world_dishes.py appended: "Identify anything" links to them
    foods_path = DOCS / "data" / "foods.json"
    if foods_path.exists():
        have = {r[0] for r in foods}
        foods += [r for r in json.loads(foods_path.read_text(encoding="utf-8")) if r[0].startswith("wd_") and r[0] not in have]
    foods_path.write_text(json.dumps(foods, separators=(",", ":")), encoding="utf-8")
    print(f"foods.json: {len(foods)} foods")

    print(f"site ready in {DOCS}")


if __name__ == "__main__":
    main()
