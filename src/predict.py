"""Single-image inference: photo -> top-k dishes -> Grad-CAM -> nutrition estimate.

    python -m src.predict path/to/meal.jpg --checkpoint checkpoints/best.pt --portion large
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

from src.data import build_transforms
from src.gradcam import grad_cam, overlay
from src.models import load_checkpoint
from src.nutrition import NutritionDB, NutritionEstimate, load_db

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "app" / "model" / "nutrisnap.pt"


@dataclass
class Prediction:
    label: str
    display_name: str
    confidence: float


@dataclass
class AnalysisResult:
    predictions: list[Prediction]
    nutrition: NutritionEstimate
    cam_image: Image.Image | None = None
    low_confidence: bool = False
    extra: dict = field(default_factory=dict)


class Predictor:
    """Loads a checkpoint once and runs inference on PIL images."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CKPT, device: str = "cpu",
                 nutrition_db: NutritionDB | None = None, tta: bool = True):
        self.model, self.classes, self.cfg = load_checkpoint(checkpoint, device)
        self.device = torch.device(device)
        self.img_size = self.cfg["train"].get("img_size", 224)
        self.transform = build_transforms(self.img_size, train=False)
        self.db = nutrition_db or load_db()
        self.tta = tta
        missing = [c for c in self.classes if c not in self.db]
        if missing:
            raise ValueError(f"classes without nutrition data: {missing}")

    @staticmethod
    def load_image(image: Image.Image | str | Path) -> Image.Image:
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        return ImageOps.exif_transpose(img).convert("RGB")  # phone photos carry EXIF rotation

    @torch.inference_mode()
    def probabilities(self, image: Image.Image) -> torch.Tensor:
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        if self.tta:  # horizontal-flip test-time augmentation
            logits = (logits + self.model(torch.flip(x, dims=[3]))) / 2
        logits = logits / getattr(self.model, "temperature", 1.0)  # calibrated confidence
        return logits.softmax(1)[0].float().cpu()

    def topk(self, image: Image.Image, k: int = 3) -> list[Prediction]:
        probs = self.probabilities(image)
        conf, idx = probs.topk(min(k, len(self.classes)))
        return [
            Prediction(self.classes[i], self.db.get(self.classes[i]).display_name, float(c))
            for c, i in zip(conf.tolist(), idx.tolist())
        ]

    def explain(self, image: Image.Image, label: str | None = None) -> Image.Image:
        """Grad-CAM overlay for ``label`` (default: top-1) on the center-cropped view the model saw."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        class_idx = self.classes.index(label) if label else None
        cam, _ = grad_cam(self.model, x, class_idx)
        # Reproduce the eval resize + center crop so the heatmap aligns with what the model saw.
        view = T.Compose([T.Resize(int(round(self.img_size / 0.875))), T.CenterCrop(self.img_size)])(image)
        return overlay(view, cam)

    def analyze(self, image: Image.Image | str | Path, portion: str | float = "medium",
                k: int = 3, with_cam: bool = True, low_conf_threshold: float = 0.4) -> AnalysisResult:
        img = self.load_image(image)
        preds = self.topk(img, k)
        top = preds[0]
        return AnalysisResult(
            predictions=preds,
            nutrition=self.db.estimate(top.label, portion),
            cam_image=self.explain(img, top.label) if with_cam else None,
            low_confidence=top.confidence < low_conf_threshold,
        )



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--portion", default="medium", help="small|medium|large or grams")
    ap.add_argument("--cam-out", type=Path, default=None, help="save Grad-CAM overlay here")
    args = ap.parse_args()
    portion: str | float = float(args.portion) if args.portion.replace(".", "", 1).isdigit() else args.portion
    p = Predictor(args.checkpoint)
    res = p.analyze(args.image, portion, with_cam=args.cam_out is not None)
    if args.cam_out and res.cam_image:
        res.cam_image.save(args.cam_out)
    print(json.dumps({
        "predictions": [vars(x) for x in res.predictions],
        "nutrition": res.nutrition.as_dict(),
        "low_confidence": res.low_confidence,
    }, indent=2))


if __name__ == "__main__":
    main()
