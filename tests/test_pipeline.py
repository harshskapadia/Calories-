"""Tests for the prediction pipeline: transforms, checkpoint round-trip, top-k, Grad-CAM, analyze().

A randomly initialised model is saved to a temp checkpoint, so these tests need no
dataset and no network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import build_transforms
from src.gradcam import grad_cam, overlay
from src.models import create_model, load_checkpoint, save_checkpoint, set_backbone_trainable
from src.predict import Predictor

CLASSES = ["apple_pie", "pizza", "sushi", "waffles"]
IMG = 64


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    cfg = {"experiment": "test", "model": {"backbone": "efficientnet_b0"}, "train": {"img_size": IMG}}
    model = create_model("efficientnet_b0", len(CLASSES), pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(path, model, CLASSES, cfg)
    return path


@pytest.fixture(scope="module")
def predictor(ckpt: Path) -> Predictor:
    return Predictor(ckpt)


@pytest.fixture
def photo() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (120, 90, 3), dtype=np.uint8))


def test_transforms_shapes(photo: Image.Image) -> None:
    for train in (True, False):
        aug = {"randaugment": True, "color_jitter": 0.2, "random_erasing": 0.25} if train else None
        x = build_transforms(IMG, train=train, aug=aug)(photo)
        assert x.shape == (3, IMG, IMG)


def test_checkpoint_roundtrip(ckpt: Path) -> None:
    model, classes, cfg = load_checkpoint(ckpt)
    assert classes == CLASSES and cfg["model"]["backbone"] == "efficientnet_b0"
    assert not model.training


def test_freeze_backbone_keeps_head_trainable() -> None:
    m = create_model("efficientnet_b0", 3, pretrained=False)
    set_backbone_trainable(m, False)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("classifier") for n in trainable)
    set_backbone_trainable(m, True)
    assert all(p.requires_grad for p in m.parameters())


def test_topk_is_sorted_probability_distribution(predictor: Predictor, photo: Image.Image) -> None:
    preds = predictor.topk(photo, k=3)
    assert len(preds) == 3
    confs = [p.confidence for p in preds]
    assert confs == sorted(confs, reverse=True)
    assert all(0 <= c <= 1 for c in confs)
    assert len({p.label for p in preds}) == 3
    probs = predictor.probabilities(photo)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_topk_clamped_to_num_classes(predictor: Predictor, photo: Image.Image) -> None:
    assert len(predictor.topk(photo, k=10)) == len(CLASSES)


def test_gradcam_range_and_overlay(predictor: Predictor, photo: Image.Image) -> None:
    x = predictor.transform(photo).unsqueeze(0)
    cam, idx = grad_cam(predictor.model, x)
    assert cam.ndim == 2 and 0 <= cam.min() and cam.max() <= 1
    assert 0 <= idx < len(CLASSES)
    over = overlay(photo, cam)
    assert over.size == photo.size and over.mode == "RGB"


def test_explain_matches_model_view(predictor: Predictor, photo: Image.Image) -> None:
    assert predictor.explain(photo, "sushi").size == (IMG, IMG)


def test_analyze_end_to_end(predictor: Predictor, photo: Image.Image, tmp_path: Path) -> None:
    path = tmp_path / "meal.jpg"
    photo.save(path)
    res = predictor.analyze(path, portion=250)
    top = res.predictions[0]
    assert res.nutrition.name == top.label
    assert res.nutrition.grams == 250
    expected = predictor.db.get(top.label).kcal_per_100g * 2.5
    assert res.nutrition.kcal == pytest.approx(expected, abs=0.2)
    assert res.cam_image is not None
    # random weights -> near-uniform probabilities -> must be flagged as low confidence
    assert res.low_confidence


def test_analyze_portion_presets(predictor: Predictor, photo: Image.Image) -> None:
    small = predictor.analyze(photo, "small", with_cam=False).nutrition
    large = predictor.analyze(photo, "large", with_cam=False).nutrition
    assert small.name == large.name and small.kcal < large.kcal


def test_exif_rotation_is_applied() -> None:
    img = Image.new("RGB", (40, 20))
    exif = img.getexif()
    exif[0x0112] = 6  # orientation: rotate 90 CW
    import io

    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    buf.seek(0)
    assert Predictor.load_image(Image.open(buf)).size == (20, 40)


def test_config_overrides() -> None:
    from src.train import apply_overrides

    cfg = {"train": {"epochs": 5, "lr": 1e-3}, "data": {"name": "x"}}
    out = apply_overrides(cfg, ["train.epochs=2", "train.lr=3e-4", "model.backbone=convnext_tiny", "data.num_classes=null"])
    assert out["train"]["epochs"] == 2 and out["train"]["lr"] == pytest.approx(3e-4)
    assert out["model"]["backbone"] == "convnext_tiny" and out["data"]["num_classes"] is None
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["train.epochs"])


def test_fp16_export_roundtrip(ckpt: Path, tmp_path: Path, photo: Image.Image) -> None:
    from scripts.export_model import export

    out = export(ckpt, tmp_path / "app.pt")
    assert out.stat().st_size < ckpt.stat().st_size * 0.6
    full, small = Predictor(ckpt, tta=False), Predictor(out, tta=False)
    assert next(small.model.parameters()).dtype == torch.float32  # cast back on load
    assert torch.allclose(full.probabilities(photo), small.probabilities(photo), atol=1e-2)


def test_temperature_scaling_fixes_overconfidence() -> None:
    from src.calibrate import expected_calibration_error, fit_temperature

    torch.manual_seed(0)
    n, c = 4000, 5
    labels = torch.randint(0, c, (n,))
    clean = torch.randn(n, c)
    clean[torch.arange(n), labels] += 1.5          # informative but uncertain logits
    overconfident = clean * 4                       # same ranking, 4x too sharp
    t = fit_temperature(overconfident, labels)
    assert 2.5 < t < 6                              # recovers roughly the 4x scaling
    y = labels.numpy()
    before = expected_calibration_error(overconfident.softmax(1).numpy(), y)
    after = expected_calibration_error((overconfident / t).softmax(1).numpy(), y)
    assert after < before / 2
    assert ((overconfident / t).argmax(1) == overconfident.argmax(1)).all()  # never changes the prediction


def test_temperature_is_applied_by_predictor(ckpt: Path, tmp_path: Path, photo: Image.Image) -> None:
    raw = torch.load(ckpt, weights_only=False)
    raw["temperature"] = 5.0
    hot = tmp_path / "hot.pt"
    torch.save(raw, hot)
    base, soft = Predictor(ckpt, tta=False), Predictor(hot, tta=False)
    assert soft.model.temperature == 5.0
    assert soft.probabilities(photo).max() <= base.probabilities(photo).max() + 1e-6
    assert soft.probabilities(photo).argmax() == base.probabilities(photo).argmax()


def test_phone_augmentations(photo: Image.Image) -> None:
    from src.data import RandomJPEG

    x = build_transforms(IMG, train=True, aug={"phone_aug": True})(photo)
    assert x.shape == (3, IMG, IMG)
    out = RandomJPEG(p=1.0)(photo)
    assert out.size == photo.size and out.mode == "RGB"
