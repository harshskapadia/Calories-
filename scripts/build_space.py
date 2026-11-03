"""Assemble a self-contained Hugging Face Space in build/space/.

    python scripts/build_space.py                      # uses app/model/nutrisnap.pt
    huggingface-cli upload <user>/nutrisnap build/space . --repo-type space

The Space keeps the repo layout (app/app.py imports from src/), so ``app_file`` points
at app/app.py. Only inference code, the nutrition table, the model and examples are
copied: no training data, notebooks or caches.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gradio

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: NutriSnap
emoji: 🍽️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: {gradio_version}
app_file: app/app.py
pinned: false
license: mit
short_description: Food photo to dish, portion, calories and macros
---

# NutriSnap 🍽️

Upload or snap a photo of a meal: the model recognises the dish (139 Western and Indian dishes,
ConvNeXt-Tiny, 88.8% top-1 on 26,836 test images), shows a Grad-CAM heatmap of what it looked at,
and estimates calories and macros for an adjustable portion using USDA FoodData Central (FNDDS).
Anything it doesn't recognise can be looked up in a search over 5,500+ foods.

Training data: Food-101 (research use) and the SohlHealth Enhanced Indian Food Classification
dataset (CC-BY-4.0).

**All values are estimates. Not medical or dietary advice.**

Source code, training pipeline and evaluation: https://github.com/harshskapadia/Calories-
"""

SPACE_REQUIREMENTS = """# CPU-only PyTorch wheels: much smaller and faster to install than the default CUDA build
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.3
torchvision>=0.18
timm>=1.0
numpy>=1.26
pillow>=10.0
opencv-python-headless>=4.9
pandas>=2.1
matplotlib>=3.8
pyyaml>=6.0
"""

SRC_FILES = ["__init__.py", "data.py", "gradcam.py", "models.py", "nutrition.py", "portion.py", "predict.py"]


def build(out: Path, ckpt: Path) -> Path:
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt} (train one, or pass --checkpoint)")
    if out.exists():
        shutil.rmtree(out)
    (out / "src").mkdir(parents=True)
    (out / "app" / "model").mkdir(parents=True)
    (out / "data").mkdir()
    for f in SRC_FILES:
        shutil.copy2(ROOT / "src" / f, out / "src" / f)
    shutil.copy2(ROOT / "app" / "app.py", out / "app" / "app.py")
    shutil.copy2(ckpt, out / "app" / "model" / "nutrisnap.pt")
    if (ROOT / "app" / "examples").exists():
        shutil.copytree(ROOT / "app" / "examples", out / "app" / "examples")
    for f in ("nutrition.csv", "food_search.csv"):  # dishes + the 5,400-food search table
        shutil.copy2(ROOT / "data" / f, out / "data" / f)
    (out / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__), encoding="utf-8")
    (out / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Space assembled in {out} ({size:.1f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "app" / "model" / "nutrisnap.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "space")
    args = ap.parse_args()
    build(args.out, args.checkpoint)


if __name__ == "__main__":
    main()
