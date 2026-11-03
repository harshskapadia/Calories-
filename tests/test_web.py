"""Consistency checks for the in-browser site in docs/ (built by scripts/build_web.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
pytestmark = pytest.mark.skipif(not (DOCS / "model" / "meta.json").exists(), reason="web site not built")


def test_meta_and_foods_are_consistent() -> None:
    meta = json.loads((DOCS / "model" / "meta.json").read_text(encoding="utf-8"))
    foods = json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))
    keys = {row[0] for row in foods}
    assert len(meta["classes"]) == len(meta["display_names"]) == 139
    assert set(meta["classes"]) <= keys  # every recognisable dish has nutrition in the browser
    assert len(foods) > 5000 and all(len(r) == 9 and r[6] > 0 for r in foods)
    assert 0.3 < meta["temperature"] < 5 and meta["img_size"] == 224 and meta["resize"] == 256


def test_site_files_present() -> None:
    for rel in ["index.html", "app.js", "style.css", "model/nutrisnap.int8.onnx", "ort/ort.wasm.min.js",
                "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm", ".nojekyll",
                "products.js", "zxing/zxing-reader.js", "zxing/zxing_reader.wasm"]:
        assert (DOCS / rel).exists(), rel
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'src="ort/ort.wasm.min.js"' in html and 'src="app.js"' in html and 'src="products.js"' in html
    js = (DOCS / "products.js").read_text(encoding="utf-8")
    assert "world.openfoodfacts.org" in js and "Open Food Facts" in html and "ODbL" in html  # attribution


def test_open_vocabulary_bank() -> None:
    meta = json.loads((DOCS / "model" / "clip_vocab.json").read_text(encoding="utf-8"))
    foods = {row[0] for row in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    n, d = len(meta["keys"]), meta["dim"]
    assert n == len(meta["names"]) > 5000 and d == 512
    assert set(meta["keys"]) <= foods  # every open-vocabulary match links to nutrition
    assert (DOCS / "model" / "clip_text.bin").stat().st_size == n * d
    assert (DOCS / "model" / "clip_visual.int8.onnx").stat().st_size < 100e6  # GitHub file limit
    assert any("milk shake" in x for x in meta["names"]) and any("smoothie" in x for x in meta["names"])


def test_world_dishes_and_regions() -> None:
    meta = json.loads((DOCS / "model" / "clip_vocab.json").read_text(encoding="utf-8"))
    foods = {r[0]: r for r in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    assert len(meta["regions"]) == len(meta["keys"]) and len(meta["region_list"]) >= 15
    world = [k for k in meta["keys"] if k.startswith("wd_")]
    assert len(world) > 5000
    for k in world[:200]:  # world dishes carry approximate nutrition + provenance (source|desc|qid|similarity)
        row = foods[k]
        assert row[7] == "approx" and row[2] > 0 and len(row[8].split("|")) == 4


def test_calorie_head_shape() -> None:
    head = json.loads((DOCS / "model" / "calorie_head.json").read_text(encoding="utf-8"))
    dim = json.loads((DOCS / "model" / "clip_vocab.json").read_text(encoding="utf-8"))["dim"]
    assert len(head["W"]) == len(head["b"]) == 4 and all(len(r) == dim for r in head["W"])
    assert head["trained_on"] > 1000 and 0 < head["metrics_unseen"]["median_abs_pct"] < 100


def test_new_layout_hooks() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    for el in ["tab-meal", "tab-pack", "panel-meal", "panel-pack", "donut", "ov-btn", "photo-est", "scan-btn", "chip-count"]:
        assert f'id="{el}"' in html, el
    assert "examples" not in html and not (DOCS / "examples").exists()
