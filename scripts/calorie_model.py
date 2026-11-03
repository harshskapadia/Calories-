"""Calories & macros directly from a photo: a small head on CLIP image features, trained on MM-Food-100K.

    python scripts/calorie_model.py            # train on seen dishes, evaluate on UNSEEN dish names, export for the web

Compares, on photos of dishes whose names never appear in training:
  1. constant baseline   - always predict the training median
  2. app pipeline        - CLIP open-vocabulary top-1 food -> its per-100 g nutrition x typical serving
  3. photo -> nutrition  - ridge regression / MLP on the CLIP embedding (predicts log kcal + macro shares)
Metrics are agreement with the dataset's labels (reviewed GPT-4o/Qwen estimates), not weighed ground truth.
Exports docs/model/calorie_head.json (weights for the browser) and results/calorie_model.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mmfood"
TARGETS = ["calories_kcal", "protein_g", "carbohydrate_g", "fat_g"]


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = np.abs(pred - true)
    rel = err / np.maximum(true, 1)
    return {"mae_kcal": round(float(err.mean()), 1), "median_abs_pct": round(float(np.median(rel) * 100), 1),
            "within_25pct": round(float((rel <= 0.25).mean() * 100), 1), "within_50pct": round(float((rel <= 0.5).mean() * 100), 1)}


def app_pipeline(E: np.ndarray) -> np.ndarray:
    """kcal the current app would show: open-vocabulary top-1 food x its typical serving."""
    docs = ROOT / "docs"
    meta = json.loads((docs / "model" / "clip_vocab.json").read_text(encoding="utf-8"))
    bank = np.fromfile(docs / "model" / "clip_text.bin", dtype=np.int8).reshape(len(meta["keys"]), meta["dim"]).astype(np.float32) / 127
    foods = {r[0]: r for r in json.loads((docs / "data" / "foods.json").read_text(encoding="utf-8"))}
    top = (E @ bank.T).argmax(1)
    return np.array([foods[meta["keys"][j]][2] * foods[meta["keys"][j]][6] / 100 for j in top])


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    return W[:-1], W[-1]


def targets(df: pd.DataFrame) -> np.ndarray:
    """log kcal + energy shares of protein/carbs/fat (shares keep macros consistent with kcal)."""
    k = df.calories_kcal.to_numpy()
    e = np.stack([4 * df.protein_g, 4 * df.carbohydrate_g, 9 * df.fat_g], 1).astype(float)
    shares = e / np.maximum(e.sum(1, keepdims=True), 1)
    return np.hstack([np.log(np.maximum(k, 10))[:, None], shares])


def decode(Y: np.ndarray) -> dict:
    kcal = np.exp(Y[:, 0])
    s = np.clip(Y[:, 1:], 0.01, None)
    s = s / s.sum(1, keepdims=True)
    return {"calories_kcal": kcal, "protein_g": kcal * s[:, 0] / 4, "carbohydrate_g": kcal * s[:, 1] / 4, "fat_g": kcal * s[:, 2] / 9}


def main() -> None:
    V = np.load(RAW / "features.npy").astype(np.float32)          # [N, views, 512]
    df = pd.read_parquet(RAW / "features_meta.parquet")
    tr, te = (df.split == "train").to_numpy(), (df.split == "test_unseen").to_numpy()
    Y = targets(df)
    # test-time (and browser): mean of original + mirrored view, re-normalised
    E = V[:, :2].mean(1)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    n_views = V.shape[1]
    # validation for lambda: hold out 15% of *training dish names*
    rng = np.random.default_rng(0)
    tr_dishes = df.dish[tr].unique()
    val_d = set(rng.choice(tr_dishes, int(len(tr_dishes) * 0.15), replace=False))
    va = tr & df.dish.isin(val_d).to_numpy()
    fit = tr & ~va
    best = None
    for lam in [0.1, 0.3, 1, 3, 10, 30]:
        W, b = fit_ridge(V[fit].reshape(-1, V.shape[2]), np.repeat(Y[fit], n_views, 0), lam)
        m = metrics(decode(E[va] @ W + b)["calories_kcal"], df.calories_kcal.to_numpy()[va])
        if best is None or m["median_abs_pct"] < best[1]["median_abs_pct"]:
            best = (lam, m)
    lam = best[0]
    W, b = fit_ridge(V[tr].reshape(-1, V.shape[2]), np.repeat(Y[tr], n_views, 0), lam)
    # robustness check: spread of the estimate across the 4 views of each unseen photo
    per_view = np.stack([np.exp(V[te][:, k] @ W[:, 0] + b[0]) for k in range(n_views)], 1)
    view_spread = float(np.median((per_view.max(1) - per_view.min(1)) / per_view.mean(1)) * 100)
    pred = decode(E[te] @ W + b)
    true = df[te]
    res = {
        "photos_train": int(tr.sum()), "photos_test_unseen_dishes": int(te.sum()),
        "dishes_test_unseen": int(df.dish[te].nunique()), "ridge_lambda": lam,
        "constant_median": metrics(np.full(te.sum(), np.median(df.calories_kcal[tr])), true.calories_kcal.to_numpy()),
        "app_pipeline_name_then_table": metrics(app_pipeline(E[te]), true.calories_kcal.to_numpy()),
        "photo_to_nutrition_ridge": metrics(pred["calories_kcal"], true.calories_kcal.to_numpy()),
        "median_spread_across_views_pct": round(view_spread, 1),
        "macro_mae_g": {k: round(float(np.abs(pred[k] - true[k].to_numpy()).mean()), 1) for k in TARGETS[1:]},
        "label_note": "labels are GPT-4o/Qwen estimates reviewed by crowd annotators (dataset card), not weighed food",
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "calorie_model.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    # export for the browser: y = e @ W + b ; kcal = exp(y0); shares = y1..3 (clipped, renormalised)
    out = {"W": np.round(W.T, 6).tolist(), "b": np.round(b, 6).tolist(), "trained_on": int(tr.sum()),
           "metrics_unseen": res["photo_to_nutrition_ridge"]}
    (ROOT / "docs" / "model" / "calorie_head.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
