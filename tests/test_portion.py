"""Tests for the reference-object portion estimator on synthetic top-down scenes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from src.portion import (
    CARD_MM,
    DEFAULT_G_PER_CM2,
    G_PER_CM2,
    QUARTER_MM,
    detect_card,
    detect_coin,
    estimate_grams_from_reference,
)

PX_PER_CM = 20.0  # synthetic scale


def _scene(card: bool = True, coin: bool = False, food_radius_cm: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((600, 800, 3), 235, np.uint8)  # light table
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    # food: a brown textured disc in the middle-right
    cv2.circle(img, (470, 300), int(food_radius_cm * PX_PER_CM), (140, 80, 40), -1)
    if card:
        w, h = CARD_MM[0] / 10 * PX_PER_CM, CARD_MM[1] / 10 * PX_PER_CM
        cv2.rectangle(img, (40, 40), (int(40 + w), int(40 + h)), (30, 60, 160), -1)
    if coin:
        cv2.circle(img, (100, 500), int(QUARTER_MM / 20 * PX_PER_CM), (120, 120, 120), -1)
    return img


def test_detect_card_scale() -> None:
    found = detect_card(_scene())
    assert found is not None
    assert found[0] == pytest.approx(PX_PER_CM, rel=0.06)


def test_detect_coin_scale() -> None:
    found = detect_coin(_scene(card=False, coin=True))
    assert found is not None
    assert found[0] == pytest.approx(PX_PER_CM, rel=0.15)


def test_no_reference_returns_none() -> None:
    img = Image.fromarray(_scene(card=False))
    assert estimate_grams_from_reference(img, "pizza", "credit card") is None


def test_area_and_grams_with_card() -> None:
    res = estimate_grams_from_reference(Image.fromarray(_scene()), "steak", "credit card")
    assert res is not None
    true_area = np.pi * 6.0**2  # ~113 cm²
    assert res.food_area_cm2 == pytest.approx(true_area, rel=0.15)
    assert res.grams == pytest.approx(res.food_area_cm2 * DEFAULT_G_PER_CM2)
    assert res.debug_image.size[0] > 0


def test_class_specific_density() -> None:
    img = Image.fromarray(_scene())
    soup = estimate_grams_from_reference(img, "ramen")
    salad = estimate_grams_from_reference(img, "greek_salad")
    assert soup and salad
    assert soup.grams / salad.grams == pytest.approx(G_PER_CM2["ramen"] / G_PER_CM2["greek_salad"], rel=0.05)
