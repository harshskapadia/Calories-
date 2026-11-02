"""Area-based portion estimate using a reference object of known size (stretch goal).

Pipeline (top-down photo, reference object lying flat next to the food):

1. **Reference detection** (OpenCV):
   - credit card: largest 4-sided contour whose aspect ratio is close to 85.60 x 53.98 mm (1.586)
   - coin (US quarter, 24.26 mm): Hough circle in a plausible radius range, picking the
     most "coin-like" small circle (plates and bowls are large circles and are ignored)
   -> gives pixels-per-cm.
2. **Food segmentation**: GrabCut initialised with a centred rectangle, with the reference
   object forced to background -> food area in pixels -> cm².
3. **Area -> grams**: ``grams = area_cm2 * g_per_cm2`` where ``g_per_cm2`` is a coarse
   per-dish "areal density" (typical thickness x density). Soups/bowls are deep, salads are
   airy, cakes are thick. These factors are rough assumptions and are documented below.

This is intentionally simple and transparent; it is NOT a substitute for depth sensing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

CARD_MM = (85.60, 53.98)
QUARTER_MM = 24.26

# grams of food per cm² of visible top-down area. Default ~1.5 g/cm² ≈ 1.5 cm thick at ~1 g/cm³.
DEFAULT_G_PER_CM2 = 1.5
G_PER_CM2: dict[str, float] = {
    # deep bowls: liquid ~4 cm deep
    **{c: 4.0 for c in ["clam_chowder", "french_onion_soup", "hot_and_sour_soup", "lobster_bisque",
                         "miso_soup", "pho", "ramen", "bibimbap"]},
    # thick items
    **{c: 3.0 for c in ["cheesecake", "chocolate_cake", "carrot_cake", "red_velvet_cake", "tiramisu",
                         "lasagna", "hamburger", "club_sandwich", "pulled_pork_sandwich", "lobster_roll_sandwich",
                         "strawberry_shortcake", "ice_cream", "frozen_yogurt", "bread_pudding", "apple_pie"]},
    # thin / flat items
    **{c: 0.9 for c in ["pizza", "beef_carpaccio", "crepes", "pancakes", "waffles", "french_toast",
                         "chicken_quesadilla", "nachos", "bruschetta", "garlic_bread", "sashimi"]},
    # airy salads / leaves
    **{c: 0.6 for c in ["caesar_salad", "greek_salad", "beet_salad", "caprese_salad", "seaweed_salad"]},
}


@dataclass
class ReferenceResult:
    grams: float
    food_area_cm2: float
    px_per_cm: float
    debug_image: Image.Image


def _resize_for_cv(img: Image.Image, max_side: int = 800) -> tuple[np.ndarray, float]:
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        arr = cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return arr, s


def detect_card(rgb: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Return (px_per_cm, 4x2 corner array) for the most card-like rectangle, or None."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = rgb.shape[0] * rgb.shape[1]
    target = CARD_MM[0] / CARD_MM[1]
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if not (0.005 * img_area < area < 0.35 * img_area):
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(approx)
        if min(w, h) == 0:
            continue
        aspect = max(w, h) / min(w, h)
        err = abs(aspect - target) / target
        if err < 0.12 and (best is None or err < best[0]):
            best = (err, max(w, h) / (CARD_MM[0] / 10), approx.reshape(4, 2))
    return (best[1], best[2]) if best else None


def detect_coin(rgb: np.ndarray) -> tuple[float, tuple[int, int, int]] | None:
    """Return (px_per_cm, (x, y, r)) for a coin-sized circle, or None."""
    gray = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 5)
    m = min(gray.shape)
    # HOUGH_GRADIENT_ALT: param2 is a circle "perfectness" score in [0, 1]; fewer false positives
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=m * 0.05, param1=300, param2=0.85,
                               minRadius=int(m * 0.015), maxRadius=int(m * 0.09))
    if circles is None:
        return None
    x, y, r = (int(round(v)) for v in circles[0][0])  # best-scoring circle
    return (2 * r) / (QUARTER_MM / 10), (x, y, r)


def segment_food(rgb: np.ndarray, exclude: np.ndarray | None = None, iters: int = 4) -> np.ndarray:
    """GrabCut foreground mask (bool) from a centred rectangle; ``exclude`` pixels forced to background."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    if exclude is not None and exclude.any():
        mask[exclude] = cv2.GC_BGD
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if exclude is not None:
        fg &= ~exclude
    # keep the largest connected component (the plate of food), drop specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
    if n <= 1:
        return fg
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def estimate_grams_from_reference(image: Image.Image, label: str, reference: str = "credit card") -> ReferenceResult | None:
    rgb, _ = _resize_for_cv(image)
    exclude = np.zeros(rgb.shape[:2], np.uint8)
    dbg = rgb.copy()
    if reference.startswith("coin"):
        found = detect_coin(rgb)
        if not found:
            return None
        px_per_cm, (x, y, r) = found
        cv2.circle(exclude, (x, y), int(r * 1.3), 1, -1)
        cv2.circle(dbg, (x, y), r, (255, 215, 0), 3)
    else:
        found = detect_card(rgb)
        if not found:
            return None
        px_per_cm, quad = found
        cv2.fillConvexPoly(exclude, cv2.convexHull(quad.astype(np.int32)), 1)
        exclude = cv2.dilate(exclude, np.ones((9, 9), np.uint8))
        cv2.polylines(dbg, [quad.astype(np.int32)], True, (255, 215, 0), 3)

    food = segment_food(rgb, exclude.astype(bool))
    area_cm2 = float(food.sum()) / (px_per_cm**2)
    grams = area_cm2 * G_PER_CM2.get(label, DEFAULT_G_PER_CM2)

    overlay = dbg.copy()
    overlay[food] = (0.5 * overlay[food] + 0.5 * np.array([42, 157, 143])).astype(np.uint8)
    return ReferenceResult(grams=grams, food_area_cm2=area_cm2, px_per_cm=px_per_cm,
                           debug_image=Image.fromarray(overlay))
